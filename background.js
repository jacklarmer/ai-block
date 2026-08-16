// AI Block background service worker (MV3)
// Responsible for:
//  - locating / caching the ONNX model weights (one-time download, then offline)
//  - exposing model availability + settings to the popup
//  - handling a few orchestration messages

const MODEL_SRC = chrome.runtime.getURL("model/detector.onnx");
// Bump this cache name whenever the shipped weights change, so a stale cached
// model is invalidated and the fresh bundled on-disk model is served.
const MODEL_CACHE = "locallens-model-v4";
// Human-readable model build — surfaced in the popup so the installed weights
// are always identifiable without guessing. Keep in sync with the shipped
// model/detector.onnx (see README Metrics / git).
const MODEL_VERSION = "v13 · broadened real";

// Small helper: report state to popup / any sender
async function modelState() {
  const cache = await caches.open(MODEL_CACHE);
  const resp = await cache.match(MODEL_SRC);
  const bundled = await isBundledModelPresent();
  return {
    cached: !!resp,
    bundled,
    ready: !!(resp || bundled),
    version: MODEL_VERSION,
  };
}

function isBundledModelPresent() {
  return fetch(MODEL_SRC, { method: "HEAD" })
    .then((r) => r.ok)
    .catch(() => false);
}

async function ensureModel() {
  // If the model is already bundled (shipped inside the extension package),
  // there is nothing to download — content scripts load it directly.
  const bundled = await isBundledModelPresent();
  if (bundled) {
    return { ok: true, bundled: true, action: "bundled" };
  }
  // Otherwise cache a remotely-hosted build once, then serve offline from cache.
  const cache = await caches.open(MODEL_CACHE);
  const hit = await cache.match(MODEL_SRC);
  if (hit) return { ok: true, bundled: false, action: "cached" };
  try {
    const resp = await fetch(MODEL_SRC);
    if (!resp.ok) throw new Error(`model fetch failed ${resp.status}`);
    await cache.put(MODEL_SRC, resp);
    return { ok: true, bundled: false, action: "downloaded" };
  } catch (e) {
    return { ok: false, error: String(e && e.message || e) };
  }
}

chrome.runtime.onInstalled.addListener(() => {
  // warm nothing; model is lazily fetched on first use
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "locallens:state") {
    modelState().then((s) => sendResponse(s));
    return true; // async
  }
  if (msg && msg.type === "locallens:ensureModel") {
    ensureModel().then((s) => sendResponse(s));
    return true;
  }
});

// Keep the service worker alive briefly while large model files are fetched.
chrome.runtime.onSuspend && chrome.runtime.onSuspend.addListener(() => {});
