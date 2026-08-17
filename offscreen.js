// AI Block — offscreen-document inference engine.
//
// Runs ONNX Runtime Web (WebGPU, with WASM CPU fallback) in a hidden extension
// document that is immune to any page's Content-Security-Policy. It both (a)
// compiles/runs wasm without page CSP, and (b) fetches image bytes with the
// extension's <all_urls> host permission — bypassing both the page's CSP AND
// CORS-read restrictions, so it works even on sites that refuse to serve their
// images cross-origin (e.g. reddit.com).
//
// The content script sends either a ready [1,3,256,256] float32 tensor
// (data:/blob images it already decoded) or an http(s) image URL. When given a
// URL we fetch, decode, and preprocess here — same torchvision-val pipeline the
// model was trained/measured with (Resize shortest→288, CenterCrop 256, ImageNet
// normalize).

"use strict";

const IMG_SIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortLib = null;
let session = null;
let initPromise = null;

function loadOrt() {
  if (ortLib) return Promise.resolve(ortLib);
  // window.ort is set by the ORT UMD <script> in offscreen.html (note: must not
  // shadow with a local `let ort` — that collides with the global and crashes).
  if (!window.ort) return Promise.reject(new Error("onnxruntime-web not loaded in offscreen doc"));
  ortLib = window.ort;
  // No COOP/COEP cross-origin isolation here -> no SharedArrayBuffer -> use the
  // single-thread build so it runs anywhere.
  ortLib.env.wasm.numThreads = 1;
  ortLib.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
  return Promise.resolve(ortLib);
}

async function getSession() {
  if (session) return session;
  const o = await loadOrt();
  const eps = ["webgpu", "wasm"];
  let lastErr = null;
  for (const ep of eps) {
    try {
      session = await o.InferenceSession.create(chrome.runtime.getURL("model/detector.onnx"), {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
        logSeverityLevel: 3,
      });
      return session;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// Preprocess an ImageBitmap into the model's expected NCHW float32 tensor.
// Mirrors detector.js exactly (must stay in sync with the training pipeline).
function preprocess(bmp) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const srcW = bmp.width, srcH = bmp.height;
  // Match detector.js: Resize so the SHORTEST side -> 288 then center-crop 256.
  const RESIZE = 288;
  const fittedW = Math.max(1, Math.round(srcW * Math.max(RESIZE / srcW, RESIZE / srcH)));
  const fittedH = Math.max(1, Math.round(srcH * Math.max(RESIZE / srcW, RESIZE / srcH)));
  const work = document.createElement("canvas");
  work.width = fittedW;
  work.height = fittedH;
  const wctx = work.getContext("2d", { willReadFrequently: true });
  wctx.drawImage(bmp, 0, 0, fittedW, fittedH);
  const offX = Math.max(0, Math.round((fittedW - IMG_SIZE) / 2));
  const offY = Math.max(0, Math.round((fittedH - IMG_SIZE) / 2));
  ctx.drawImage(work, offX, offY, IMG_SIZE, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);
  const id = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const px = id.data;
  const data = new Float32Array(1 * 3 * IMG_SIZE * IMG_SIZE);
  let p = 0;
  for (let c = 0; c < 3; c++) {
    const mean = MEAN[c], std = STD[c];
    for (let i = 0; i < px.length; i += 4) data[p++] = (px[i + c] / 255.0 - mean) / std;
  }
  return { data, name: "input", dims: [1, 3, IMG_SIZE, IMG_SIZE], type: "float32" };
}

async function blobToBitmap(blob) {
  return await createImageBitmap(blob);
}

async function tensorToInput(tensor, dims) {
  // Typed arrays can arrive from a message with a realm-mismatched constructor;
  // ORT validates `instanceof Float32Array` strictly, so copy into a fresh one.
  const data = tensor instanceof Float32Array ? tensor : new Float32Array(tensor);
  return { data, name: "input", dims, type: "float32" };
}

async function runInference(input) {
  const o = await loadOrt();
  const s = await getSession();
  const feeds = { input: new o.Tensor(input.type, input.data, input.dims) };
  const results = await s.run(feeds);
  let out = results[s.outputNames[0]] || results["output"] || results["logits"];
  if (!out) { const ks = Object.keys(results); if (ks.length) out = results[ks[0]]; }
  const vals = Array.from(out.data);
  let p0, p1;
  if (vals.length >= 2) {
    const m = Math.max(vals[0], vals[1]);
    const e0 = Math.exp(vals[0] - m), e1 = Math.exp(vals[1] - m), k = e0 + e1;
    p0 = e0 / k; p1 = e1 / k;
  } else { p1 = vals[0]; p0 = 1 - p1; }
  return { fake: p1, real: p0 };
}

// Serialize inference: ORT `session.run()` is not concurrency-safe (concurrent
// calls throw "Session already started"), and the content script fires many
// detects in parallel. Chaining every run through one promise keeps them ordered.
let runQueue = Promise.resolve();
function enqueue(fn) {
  const next = runQueue.then(fn, fn);
  runQueue = next.catch(() => {}); // keep the chain alive after errors
  return next;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "locallens:infer") {
    initPromise = initPromise || loadOrt();
    initPromise
      // fetch+decode+preprocess are safe in parallel; only the run is serialized
      .then(async () => {
        if (msg.imageUrl && /^(https?:|data:)/i.test(msg.imageUrl)) {
          const resp = await fetch(msg.imageUrl, { credentials: "include" });
          if (!resp.ok) throw new Error("image fetch failed " + resp.status);
          const bmp = await blobToBitmap(await resp.blob());
          const input = preprocess(bmp);
          bmp.close && bmp.close();
          return enqueue(() => runInference(input));
        }
        const input = await tensorToInput(msg.tensor, msg.dims);
        return enqueue(() => runInference(input));
      })
      .then((r) => sendResponse({ ok: true, ...r }))
      .catch((e) => sendResponse({ ok: false, error: String((e && e.message) || e) }));
    return true; // async
  }
});

// Warm the ORT load immediately so the first inference stays fast.
initPromise = loadOrt().catch(() => { /* lazily retried on first message */ });
