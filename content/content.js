// LocalLens content script — finds <img> elements, runs on-device detection,
// and overlays a small confidence badge. Also powers click-to-inspect on the
// active image (popup + context menu via messaging).
(function () {
  "use strict";

  const THRESHOLD_DEFAULT = 0.65; // confidence at which we call "AI"
  const MIN_IMG_AREA = 48 * 48; // ignore tiny icons / trackers
  const debounceMs = 250;

  let enabled = true;
  let threshold = THRESHOLD_DEFAULT;
  let scanned = new WeakSet();
  let queue = [];
  let running = false;
  let badgeStyle = {};

  // ---------- settings ----------
  function loadSettings() {
    try {
      chrome.storage.local.get(["locallens_enabled", "locallens_threshold", "locallens_badge"], (s) => {
        enabled = s.locallens_enabled !== false;
        threshold = typeof s.locallens_threshold === "number" ? s.locallens_threshold : THRESHOLD_DEFAULT;
        badgeStyle = s.locallens_badge || {};
        if (enabled) schedule();
      });
    } catch (e) {
      /* storage may be unavailable */
    }
  }

  // ---------- image harvesting ----------
  function collectImages() {
    const out = [];
    const imgs = document.querySelectorAll("img");
    for (const img of imgs) {
      if (scanned.has(img)) continue;
      if (!img.complete && !img.src) continue;
      if (!img.src || img.src.startsWith("data:image/svg")) continue;
      // natural size check
      const w = img.naturalWidth || img.width || 0;
      const h = img.naturalHeight || img.height || 0;
      if (w * h < MIN_IMG_AREA) {
        scanned.add(img);
        continue;
      }
      // already has a badge or is our own badge
      if (img.dataset.locallens !== undefined) continue;
      out.push(img);
    }
    return out;
  }

  // ---------- badge drawing ----------
  function drawBadge(img, result) {
    if (document.hidden) return;
    const label = result.label;
    const conf = Math.round(result.label === "AI-generated" ? result.fake * 100 : result.real * 100);
    const color = result.label === "AI-generated" ? "rgba(214,40,40,0.92)" : "rgba(34,139,51,0.92)";

    // Wrap the img in a positioned container so we can overlay a corner tag.
    let wrap = img.closest("[data-locallens-wrap]");
    if (!wrap) {
      if (img.parentElement) {
        wrap = img.parentElement;
      } else {
        return;
      }
    }
    let style = "position:relative;display:inline-block;";
    try {
      wrap.setAttribute("style", (wrap.getAttribute("style") || "") + style);
    } catch (e) {}
    wrap.setAttribute("data-locallens-wrap", "1");

    const badge = document.createElement("div");
    badge.className = "locallens-badge";
    badge.textContent = `${label} · ${conf}%`;
    badge.style.cssText = [
      "position:absolute",
      "top:4px",
      "left:4px",
      "z-index:2147483647",
      "background:" + color,
      "color:#fff",
      "font:600 11px/1.4 -apple-system,Segoe UI,Roboto,sans-serif",
      "padding:2px 6px",
      "border-radius:4px",
      "pointer-events:none",
      "box-shadow:0 1px 3px rgba(0,0,0,0.4)",
    ].join(";");
    badge.dataset.locallens = img.dataset.locallens || "";
    // Mark the image as already processed
    try {
      img.setAttribute("data-locallens", result.label);
    } catch (e) {}
    wrap.appendChild(badge);
    img.__locallensBadge = badge;
  }

  // ---------- processing loop (serialized) ----------
  function schedule() {
    if (running) return;
    running = true;
    process();
  }

  async function process() {
    try {
      // warm the detector session once
      await LocalLensDetector.loadSession();
    } catch (e) {
      console.error("[LocalLens] model load failed:", e);
      running = false;
      return;
    }

    while (enabled) {
      collectImages().forEach((im) => enqueue(im));
      const img = queue.shift();
      if (!img) break;

      try {
        // Load the image pixel data respecting CORS. Many image hosts (e.g.
        // Twitter's pbs.twimg.com) serve images with Access-Control-Allow-Origin,
        // so we try to fetch the bytes and decode from a Blob — this avoids the
        // "canvas tainted by cross-origin image" failure that would otherwise
        // skip every social photo. Fall back to the in-DOM <img> when CORS blocks
        // the fetch (that may taint the canvas; handled by try/catch below).
        let loaded = null;
        const url = img.currentSrc || img.src;
        if (url && /^(https?:|data:)/.test(url)) {
          try {
            const resp = await fetch(url, { credentials: "omit", mode: "cors" });
            if (resp.ok) {
              const blob = await resp.blob();
              loaded = await createImageBitmap(blob);
              img.__locallensBmp = loaded;
            }
          } catch (e) {
            loaded = null;
          }
        }
        if (!loaded && img.complete && img.naturalWidth > 0) {
          try {
            loaded = await createImageBitmap(img);
            img.__locallensBmp = loaded;
          } catch (e) {
            loaded = img;
          }
        }
        if (!loaded) {
          scanned.add(img);
          continue;
        }
        const result = await LocalLensDetector.detect(loaded, {});
        if (result.label === "AI-generated" && result.fake < threshold) {
          // below threshold -> treat as real/unflagged but still show soft badge
          result.label = "Real";
        }
        drawBadge(img, result);
        if (img.__locallensBmp) {
          img.__locallensBmp.close();
          img.__locallensBmp = null;
        }
      } catch (e) {
        // skip images we cannot decode (e.g. CORS-tainted)
        scanned.add(img);
      } finally {
        scanned.add(img);
      }
      // yield to avoid blocking
      await new Promise((r) => setTimeout(r, 0));
    }
    running = false;
  }

  function enqueue(img) {
    if (scanned.has(img)) return;
    queue.push(img);
  }

  // ---------- MutationObserver for lazy content ----------
  let mo = null;
  function startObserver() {
    if (mo) return;
    mo = new MutationObserver(() => schedule());
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  // ---------- messaging ----------
  try {
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (msg && msg.type === "locallens:scan") {
        schedule();
        sendResponse({ ok: true });
        return false;
      }
      if (msg && msg.type === "locallens:disable") {
        enabled = false;
        sendResponse({ ok: true });
        return false;
      }
      if (msg && msg.type === "locallens:enable") {
        enabled = true;
        loadSettings();
        sendResponse({ ok: true });
        return false;
      }
    });
  } catch (e) {}

  // ---------- init ----------
  loadSettings();
  startObserver();
  // initial scan after a tick
  setTimeout(schedule, 300);
})();
