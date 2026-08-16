// LocalLens content script — finds <img> elements, runs on-device detection,
// and overlays a small confidence badge. Also powers click-to-inspect on the
// active image (popup + context menu via messaging).
(function () {
  "use strict";

  const THRESHOLD_DEFAULT = 0.5; // flag as "AI" at 50% confidence; measured real-photo
  // false-positive rate on the held-out set is ~0.001 even at 0.40, so this is a
  // safe, high-recall point that catches ~82% of unseen-generator AI images
  // versus ~78% at the old 0.65 default. User can tune via the popup slider.
  const MIN_IMG_AREA = 48 * 48; // ignore tiny icons / trackers
  // Skip SMALL images entirely — profile pictures, avatars, icons, tiny
  // thumbnails. We judge "small" by RENDERED (on-screen) size, because an
  // avatar is small even if its source file is 400x400. Real photos worth
  // classifying are almost always >= ~72px on screen in both dimensions.
  const MIN_RENDERED = 72;
  const debounceMs = 250;

  let enabled = true;
  let threshold = THRESHOLD_DEFAULT;
  let scanned = new WeakSet();
  let deferred = new Set(); // far-offscreen imgs: defer until scrolled near view
  let queue = [];
  let running = false;
  let dirty = false; // set by observer/scroll even while the loop is running
  let badgeStyle = {};
  let periodicTimer = null;
  const PERIOD_MS = 1200; // paced re-check so scroll-lazy images are never missed
  // Session-scoped URL -> result cache: avoids re-running the model on images
  // that reappear (new <img> node, same URL — common in scroll/SPA layouts).
  const resultCache = new Map();
  const cacheUrl = (img) => (img.currentSrc || img.src || "").split("#")[0];

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

  // ---------- size filtering ----------
  // True if the image is too small (on screen) to be worth flagging. Uses the
  // RENDERED box so avatars/icons stay clean even when their source file is big;
  // falls back to natural size for images not yet laid out (lazy / offscreen).
  function isTooSmall(img) {
    const r = img.getBoundingClientRect();
    let w = r && r.width, h = r && r.height;
    if (!w || !h) { w = img.naturalWidth || img.width || 0; h = img.naturalHeight || img.height || 0; }
    // small in BOTH dimensions -> an avatar/icon/thumbnail; skip.
    if (w > 0 && h > 0 && w < MIN_RENDERED && h < MIN_RENDERED) return true;
    // intrinsic tiny image (e.g. a real 32x32 favicon)
    if ((img.naturalWidth || 0) * (img.naturalHeight || 0) < MIN_IMG_AREA) return true;
    return false;
  }

  // Viewport-aware: whether this image is near the visible viewport (with a
  // margin) so we only pay for fetch/decode/inference on images the user will
  // actually see soon. Far-offscreen images are deferred until scrolled into
  // range by the periodic rescan — slashes wasted work on huge infinite feeds.
  const vh_ = (window.innerHeight || 1200);
  const VP_MARGIN = vh_ - 50; // ~1 viewport of lookahead
  function isNearViewport(img) {
    const r = img.getBoundingClientRect();
    if (!r) return true;
    const vh = window.innerHeight || 1200;
    // keep a generous margin (above and below) so we prefetch just-ahead-of-view
    const top = -vh, bottom = vh * 2 + VP_MARGIN;
    return r.bottom >= top && r.top <= bottom;
  }

  // ---------- image harvesting ----------
  function collectImages() {
    const out = [];
    const imgs = document.querySelectorAll("img");
    for (const img of imgs) {
      if (scanned.has(img)) continue;
      if (!img.complete && !img.src) continue;
      if (!img.src || img.src.startsWith("data:image/svg")) continue;
      if (isTooSmall(img)) { scanned.add(img); continue; }
      // already has a badge or is our own badge
      if (img.dataset.locallens !== undefined) continue;
      out.push(img);
    }
    // Also scan images inside OPEN shadow roots (some SPAs render their image
    // galleries inside shadow DOM, e.g. certain news/media embeds).
    try {
      const hosts = document.querySelectorAll("*");
      for (const host of hosts) {
        const root = host.shadowRoot;
        if (!root) continue;
        for (const img of root.querySelectorAll("img")) {
          if (scanned.has(img)) continue;
          if (!img.src || img.src.startsWith("data:image/svg")) continue;
          if (isTooSmall(img)) { scanned.add(img); continue; }
          if (img.dataset.locallens !== undefined) continue;
          out.push(img);
        }
      }
    } catch (e) {}
    return out;
  }

  // ---------- badge drawing ----------
  function drawBadge(img, result) {
    if (document.hidden) return;
    const label = result.label;
    const conf = Math.round(result.label === "AI-generated" ? result.fake * 100 : result.real * 100);
    const color = result.label === "AI-generated" ? "rgba(214,40,40,0.92)" : "rgba(34,139,51,0.92)";
    // tooltip so a hover reveals the exact confidence (nice on desktop)
    const pct = result.label === "AI-generated" ? (result.fake * 100).toFixed(1) : (result.real * 100).toFixed(1);

    // Wrap the img in a DEDICATED container we fully control. We must NOT touch
    // the existing parent's style/layout — host pages (e.g. X/Twitter) put
    // critical inline styles (width, aspect-ratio, display) on the image's real
    // parent; mutating them collapses the layout and makes images vanish.
    let wrap = img.closest("[data-locallens-wrap]");
    if (!wrap) {
      if (!img.parentNode) return; // detached image — skip badge
      wrap = document.createElement("span");
      wrap.className = "locallens-badge-wrap";
      wrap.style.cssText = "position:relative;display:inline-block;line-height:0;";
      wrap.setAttribute("data-locallens-wrap", "1");
      // Insert as the img's new parent without disturbing the rest of the tree.
      img.parentNode.insertBefore(wrap, img);
      wrap.appendChild(img);
    }

    const badge = document.createElement("div");
    badge.className = "locallens-badge";
    badge.textContent = `${label} · ${conf}%`;
    badge.title = `${result.kind === "cached" ? "cached · " : ""}${label} confidence ${pct}% (fake=${(result.fake*100).toFixed(1)}%)`;
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
    dirty = true; // always mark pending work, even if a loop is mid-run
    if (running) return; // the running loop re-checks `dirty` each pass
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
      // Always mark loop exit as clean unless something new arrives mid-pass;
      // re-collect every iteration so freshly lazy-loaded (scroll) images are
      // picked up, even if the MutationObserver call landed while we were busy.
      dirty = false;
      const newImgs = collectImages();
      dbg.collected += newImgs.length;
      newImgs.forEach((im) => enqueue(im));
      const img = queue.shift();
      if (!img) {
        // Dirty means a lazy image likely just appeared/loaded -> keep looping
        // so it gets collected on the next pass instead of silently dropping.
        if (dirty) { await new Promise((r) => setTimeout(r, 120)); continue; }
        break;
      }

      // Viewport-aware deferral: skip far-offscreen images for now (add to
      // `scanned`+`deferred` so the normal collect skips them and the periodic
      // timer promotes them back into the queue once they scroll toward view).
      // This avoids fetching/decoding/inferring every image on a huge feed up
      // front — the big win for Google Images / infinite-scroll pages.
      if (!isNearViewport(img)) {
        scanned.add(img);
        deferred.add(img);
        continue;
      }

      try {
        // Load the image pixel data respecting CORS. Many image hosts (e.g.
        // Twitter's pbs.twimg.com) serve images with Access-Control-Allow-Origin,
        // so we try to fetch the bytes and decode from a Blob — this avoids the
        // "canvas tainted by cross-origin image" failure that would otherwise
        // skip every social photo. Fall back to the in-DOM <img> when CORS blocks
        // the fetch (that may taint the canvas; handled by try/catch below).
        let loaded = null;
        const url = img.currentSrc || img.src;
        // Serve a cached verdict instantly if we already classified this URL.
        const ckey = cacheUrl(img);
        if (ckey && resultCache.has(ckey)) {
          const cached = Object.assign({}, resultCache.get(ckey), { kind: "cached" });
          // re-apply the (possibly live-adjusted) threshold
          if (cached.fake != null && cached.fake < threshold) cached.label = "Real";
          drawBadge(img, cached);
          dbg.cached++;
          scanned.add(img);
          continue;
        }
        if (url && /^(https?:|data:)/.test(url)) {
          try {
            const resp = await fetch(url, { credentials: "omit", mode: "cors" });
            if (resp.ok) {
              const blob = await resp.blob();
              loaded = await createImageBitmap(blob);
              img.__locallensBmp = loaded;
              dbg.fetched++;
            }
          } catch (e) {
            loaded = null;
          }
        }
        if (!loaded && img.complete && img.naturalWidth > 0) {
          try {
            loaded = await createImageBitmap(img);
            img.__locallensBmp = loaded;
            dbg.fetched++;
          } catch (e) {
            loaded = img;
          }
        }
        if (!loaded) {
          scanned.add(img);
          dbg.skipped++;
          continue;
        }
        const result = await LocalLensDetector.detect(loaded, {});
        dbg.detected++;
        if (result.label === "AI-generated" && result.fake < threshold) {
          // below threshold -> treat as real/unflagged but still show soft badge
          result.label = "Real";
        }
        drawBadge(img, result);
        dbg.badged++;
        try { if (ckey) resultCache.set(ckey, { fake: result.fake, real: result.real, label: result.label }); } catch (e) {}
        if (img.__locallensBmp) {
          img.__locallensBmp.close();
          img.__locallensBmp = null;
        }
      } catch (e) {
        // skip images we cannot decode (e.g. CORS-tainted)
        dbg.errors.push(String(e && e.message));
        scanned.add(img);
      } finally {
        scanned.add(img);
      }
      // yield to avoid blocking
      await new Promise((r) => setTimeout(r, 0));
    }
    // ----- trailing catch-up -----
    // Some lazy-loading sites insert the <img> before its `src` is set, so an
    // image can be present-but-not-ready at collect time. Give those a beat to
    // populate, then do one final collect so nothing is left undetected after
    // the loop exits (e.g. the tail of a Google Images scroll).
    await new Promise((r) => setTimeout(r, 400));
    if (enabled) {
      dirty = false;
      const late = collectImages();
      if (late.length) {
        dbg.collected += late.length;
        late.forEach((im) => enqueue(im));
        while (enabled) {
          const im = queue.shift();
          if (!im) break;
          try {
            let loaded = null;
            if (im.complete && im.naturalWidth > 0) {
              try { loaded = await createImageBitmap(im); } catch (e) { loaded = im; }
            }
            if (!loaded) { scanned.add(im); continue; }
            const res = await LocalLensDetector.detect(loaded, {});
            dbg.detected++;
            if (res.label === "AI-generated" && res.fake < threshold) res.label = "Real";
            drawBadge(im, res);
            if (loaded && loaded.close) loaded.close();
          } catch (e) { dbg.errors.push(String(e && e.message)); scanned.add(im); }
          finally { scanned.add(im); }
          await new Promise((r) => setTimeout(r, 0));
        }
      }
    }
    running = false;
  }

  function enqueue(img) {
    if (scanned.has(img)) return;
    queue.push(img);
  }

  // Promote deferred (far-offscreen) images that have scrolled near the
  // viewport back into the processing queue, and drop ones that are gone.
  function promoteDeferred() {
    let promoted = false;
    for (const img of deferred) {
      if (!img.isConnected) { deferred.delete(img); continue; }
      if (isNearViewport(img)) {
        // allow re-collection by the main loop
        deferred.delete(img);
        scanned.delete(img);
        enqueue(img);
        promoted = true;
      }
    }
    return promoted;
  }

  // ---------- MutationObserver for lazy content ----------
  let mo = null;
  // Paced re-check: guarantees newly lazy-loaded (scroll) images get detected
  // even if the scroll/MutationObserver signal races with the processing loop.
  // schedule() is cheap & idempotent (collectImages dedups via `scanned`), so
  // this is safe to run periodically while enabled. When no new images arrive,
  // the loop drains and rests; the timer simply nudges it to re-collect.
  function startPeriodic() {
    if (periodicTimer) return;
    periodicTimer = setInterval(() => {
      if (!enabled) return;
      // Only wake the loop when there is genuinely new work — promote deferred
      // (scroll-into-view) images and/or newly collected unscanned images.
      const promoted = promoteDeferred();
      if (promoted || collectImages().length > 0) schedule();
    }, PERIOD_MS);
  }
  function startObserver() {
    if (mo) return;
    mo = new MutationObserver(() => schedule());
    mo.observe(document.documentElement, { childList: true, subtree: true });
    // Belt-and-suspenders: some lazy-loaders swap images on scroll without a
    // childList mutation we can rely on (or batch them). A passive scroll
    // listener guarantees scrolling surfaces new images — and cheaply promotes
    // any deferred (previously offscreen) images into the queue immediately so
    // badges appear as soon as content scrolls into view.
    window.addEventListener("scroll", () => {
      promoteDeferred(); // surfaces offscreen images that scrolled into view
      schedule();
    }, { passive: true, capture: true });
    startPeriodic();
  }

  // ---------- diagnostic counters (visible via console) ----------
  const dbg = { collected: 0, fetched: 0, detected: 0, badged: 0, cached: 0, skipped: 0, errors: [] };
  setInterval(() => {
    if (dbg.collected || dbg.detected || dbg.errors.length) {
      console.log("[LocalLens] dbg", JSON.stringify(dbg));
      if (dbg.errors.length > 3) dbg.errors.length = 3; // avoid unbounded growth
    }
  }, 10000);

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
      if (msg && msg.type === "locallens:threshold" && typeof msg.value === "number") {
        threshold = msg.value;
        if (enabled && !running) schedule();
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
