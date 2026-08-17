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
      ort.env.wasm.numThreads = 1; // see note below
      // Strict-CSP sites (e.g. reddit.com) block ORT's own `fetch()` /
      // instantiateStreaming() of the .wasm from the page world, so on those
      // pages the model silently never loads. Fix: pre-fetch the required wasm
      // bytes OURSELVES (extension origin, immune to page CSP) and hand them to
      // ORT in-memory via wasmPaths as a byte map. ORT then compiles the bytes
      // directly instead of issuing a CSP-governed fetch.
      const needed = ["ort-wasm-simd-threaded.wasm"];
      const loaded = {};
      const results = await Promise.allSettled(
        needed.map((f) => fetch(WASM_DIRS + f, { credentials: "omit" }).then((r) => r.arrayBuffer()))
      );
      results.forEach((res, i) => {
        if (res.status === "fulfilled") loaded[needed[i]] = res.value;
      });
      if (Object.keys(loaded).length) {
        ort.env.wasm.wasmPaths = loaded;
      } else {
        // fallback: let ORT try its normal path (works on non-CSP pages)
        ort.env.wasm.wasmPaths = WASM_DIRS;
      }
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
    // Prefer the in-page ORT path (fast, no message round-trip). On strict-CSP
    // pages (e.g. reddit.com) the page forbids WebAssembly compilation in the
    // content-script world, so the in-page session cannot initialize; the
    // offscreen-document fallback runs outside page CSP. We also fall back when
    // a cross-origin image taints the content-script canvas (can't read pixels).
    const o = await initOrt().catch(() => null);
    if (o) {
      try {
        const session = await loadSession(opts.forceReload);
        const input = preprocess(img);
        return await runInPage(o, session, input);
      } catch (e) {
        // fall through to offscreen path
      }
    }
    return workerInfer(img, opts);
  }

  async function runInPage(o, session, input) {
    const feeds = {};
    feeds[input.name] = new o.Tensor(input.type, input.data, input.dims);
    const t0 = performance.now();
    const results = await session.run(feeds);
    const ms = performance.now() - t0;
    let out = results[session.outputNames[0]] || results["output"] || results["logits"];
    if (!out && typeof results === "object") { const ks = Object.keys(results); if (ks.length) out = results[ks[0]]; }
    const vals = Array.from(out.data);
    // Model outputs 2-class logits -> stable softmax. index 1 = fake.
    let p0, p1;
    if (vals.length >= 2) {
      const m = Math.max(vals[0], vals[1]);
      const e0 = Math.exp(vals[0] - m), e1 = Math.exp(vals[1] - m), s = e0 + e1;
      p0 = e0 / s; p1 = e1 / s;
    } else { p1 = vals[0]; p0 = 1 - p1; }
    return { fake: p1, real: p0, label: p1 >= 0.5 ? "computer-generated" : "Real", ms: Math.round(ms), modelVersion: MODEL_VERSION, via: "inline" };
  }

  // Fallback inference in the offscreen document (extension context, immune to
  // page CSP AND CORS-read restrictions via host permission). For http(s)/data
  // images we pass the URL and let the offscreen fetch+decode+preprocess+infer,
  // which sidesteps canvas-taint entirely. Only blob: (page-scoped) URLs need
  // the preprocessed tensor sent over.
  async function workerInfer(img, opts) {
    // Prefer the original <img> element (carries the real URL) when provided;
    // fall back to whatever image-like object we were handed.
    const src = (opts && opts.original) || img;
    const url = (src && (src.currentSrc || src.src)) || "";
    let payload;
    if (/^(https?:|data:)/i.test(url)) {
      payload = { imageUrl: url };
    } else {
      const input = preprocess(img); // blob: decode fine in-page (no taint)
      payload = { tensor: input.data, dims: input.dims };
    }
    const resp = await chrome.runtime.sendMessage({ type: "locallens:infer", ...payload });
    if (!resp || !resp.ok) {
      throw new Error("inference failed (inline + offscreen): " + ((resp && resp.error) || "no response"));
    }
    return { fake: resp.fake, real: resp.real, label: resp.fake >= 0.5 ? "computer-generated" : "Real", ms: -1, modelVersion: MODEL_VERSION, via: "offscreen" };
  }

  global.LocalLensDetector = {
    detect,
    loadSession,
    preprocess,
    IMG_SIZE,
    MODEL_URL,
  };
})(typeof self !== "undefined" ? self : this);
