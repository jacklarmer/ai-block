// AI Block detector — runs the real-vs-AI classifier fully in-browser via
// ONNX Runtime Web (WebGPU EP) with a WASM CPU fallback. No network, no server,
// no uploads.
//
// Preprocessing MUST mirror the training pipeline:
//   Resize shortest side to 288 -> CenterCrop 256x256 -> RGB -> float32
//   Normalize by ImageNet mean/std -> NCHW [1,3,256,256]
//
// We deliberately use a SINGLE deterministic center-crop (no test-time
// augmentation): measurement on a held-out generalization set (an unseen
// generator + unseen real faces) showed a 5-crop x 2-flip TTA HURT balanced
// accuracy (0.887 vs 0.921) and reduced coverage, because it dilutes the
// calibrated center-crop signal. Single-pass is both faster and more accurate.
//
// Model input name is "input"; output is a 2-class logits tensor where
// index 1 = "AI / fake".

(function (global) {
  "use strict";

  const IMG_SIZE = 256;
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
  // Resolve model URL. Extension: chrome.runtime; plain page: relative, or an
  // explicit override (used by the test harness).
  const MODEL_URL = (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL)
    ? chrome.runtime.getURL("model/detector.onnx")
    : (typeof LOCALLENS_MODEL_URL !== "undefined" && LOCALLENS_MODEL_URL) || "model/detector.onnx";
  const WASM_DIRS = (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.getURL)
    ? chrome.runtime.getURL("lib/")
    : "/lib/";
  // Model build tag for the bytes actually bundled as model/detector.onnx.
  // Keep in sync with background.js MODEL_VERSION / README Metrics. This is
  // surfaced on each detect() result for debugging only — the popup reads the
  // authoritative version string from background.js.
  const MODEL_VERSION = "v14 · fresh real diversity";

  let sessionPromise = null;
  let ort = null;

  function getOrt() {
    return global.ort || global.onnx || null;
  }

  async function initOrt() {
    if (!ort) {
      ort = getOrt();
      if (!ort) {
        throw new Error("onnxruntime-web not loaded");
      }
      ort.env.wasm.wasmPaths = WASM_DIRS;
      // Content scripts on ordinary pages (e.g. twitter.com) do NOT get COOP/COEP
      // cross-origin-isolation headers, so SharedArrayBuffer is unavailable and the
      // threaded wasm build cannot initialize. Force single-thread execution — a
      // fully in-browser, CPU-only path that needs no SharedArrayBuffer and works
      // on any page. (Per-image cost stays tiny.)
      ort.env.wasm.numThreads = 1;
    }
    return ort;
  }

  // Load (and cache) the ONNX inference session with the WebGPU execution
  // provider, falling back to the WASM CPU provider when WebGPU is unavailable.
  async function loadSession(force = false) {
    if (sessionPromise && !force) return sessionPromise;
    sessionPromise = (async () => {
      const o = await initOrt();
      const epPrefs = ["webgpu", "wasm"];
      let lastErr = null;
      for (const ep of epPrefs) {
        try {
          const opts = {
            executionProviders: [ep],
            graphOptimizationLevel: "all",
            logSeverityLevel: 3,
          };
          return await o.InferenceSession.create(MODEL_URL, opts);
        } catch (e) {
          lastErr = e;
          console.warn("[AI Block] EP", ep, "failed:", e && e.message);
        }
      }
      throw lastErr;
    })();
    return sessionPromise;
  }

  // Downscale + center-crop + normalize an HTMLImageElement into a Float32Array
  // laid out as NCHW [1,3,256,256]. Matches the torchvision val pipeline used
  // at evaluation time.
  function preprocess(img) {
    const canvas = document.createElement("canvas");
    canvas.width = IMG_SIZE;
    canvas.height = IMG_SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    // Match torchvision val: Resize(288) = scale so the SHORTEST side becomes
    // 288 (long side may exceed 288), then CenterCrop(256).
    const srcW = img.naturalWidth || img.width;
    const srcH = img.naturalHeight || img.height;
    const scale = Math.max(288 / srcW, 288 / srcH);
    const fittedW = Math.max(1, Math.round(srcW * scale));
    const fittedH = Math.max(1, Math.round(srcH * scale));

    const work = document.createElement("canvas");
    work.width = fittedW;
    work.height = fittedH;
    const wctx = work.getContext("2d", { willReadFrequently: true });
    wctx.drawImage(img, 0, 0, fittedW, fittedH);

    // Center crop 256 from the resized image
    const offX = Math.max(0, Math.round((fittedW - IMG_SIZE) / 2));
    const offY = Math.max(0, Math.round((fittedH - IMG_SIZE) / 2));
    ctx.drawImage(work, offX, offY, IMG_SIZE, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

    const id = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    const px = id.data; // RGBA
    const data = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
    let p = 0;
    for (let c = 0; c < 3; c++) {
      const mean = MEAN[c];
      const std = STD[c];
      for (let i = 0; i < px.length; i += 4) {
        data[p++] = (px[i + c] / 255.0 - mean) / std;
      }
    }
    return {
      data,
      name: "input",
      dims: [1, 3, IMG_SIZE, IMG_SIZE],
      type: "float32",
    };
  }

  // Run inference on an ImageBitmap / HTMLImageElement / canvas-sourced image.
  // Returns { fake: <0..1>, real: <0..1>, label: "computer-generated"|"Real", ms }
  async function detect(img, opts = {}) {
    const o = await initOrt();
    const session = await loadSession(opts.forceReload);

    const input = preprocess(img);
    const feeds = {};
    feeds[input.name] = new o.Tensor(input.type, input.data, input.dims);

    const t0 = performance.now();
    const results = await session.run(feeds);
    const ms = performance.now() - t0;

    // Find the output tensor — grab by common names, else the first.
    let out = results[session.outputNames[0]] || results["output"] || results["logits"];
    if (!out && typeof results === "object") {
      const keys = Object.keys(results);
      if (keys.length) out = results[keys[0]];
    }
    const vals = Array.from(out.data);
    // Model outputs 2-class logits -> softmax to get calibrated probabilities.
    // index 1 = "fake" (computer-generated).
    let p0, p1;
    if (vals.length >= 2) {
      // numerically stable softmax over the two logits
      const m = Math.max(vals[0], vals[1]);
      const e0 = Math.exp(vals[0] - m);
      const e1 = Math.exp(vals[1] - m);
      const s = e0 + e1;
      p0 = e0 / s;
      p1 = e1 / s;
    } else {
      p1 = vals[0];
      p0 = 1 - vals[0];
    }
    const fake = p1;
    const real = p0;
    return {
      fake: fake,
      real: real,
      label: fake >= 0.5 ? "computer-generated" : "Real",
      ms: Math.round(ms),
      modelVersion: MODEL_VERSION,
    };
  }

  global.LocalLensDetector = {
    detect,
    loadSession,
    preprocess,
    IMG_SIZE,
    MODEL_URL,
  };
})(typeof self !== "undefined" ? self : this);
