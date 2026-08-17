/**
 * AI Block — real-world site QA harness.
 *
 * Loads the extension as a real unpacked Chrome extension, drives an actual
 * image-heavy website through scroll/lazy-load, and reports:
 *   - content-script injection + [AI Block] debug counters
 *   - badge coverage over time (does scrolling pick up lazy images?)
 *   - block-mode removals
 *   - console/page errors and unhandled rejections
 *   - no image-load regressions caused by the extension
 *
 * Usage:
 *   node test/site_qa.js <url> [--scroll N] [--mode badge|block] [--wait-ms N]
 *
 * Examples:
 *   node test/site_qa.js "https://www.google.com/search?q=cats&tbm=isch" --scroll 25
 *   node test/site_qa.js "https://imgur.com/t/animals" --mode block --scroll 15
 */
const puppeteer = require("puppeteer-core");
const path = require("path");

const EXT = path.resolve(__dirname, "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
// Chrome for Testing honors --load-extension (stable branded Chrome refuses it
// since ~v137). Prefer a CfT binary if present.
const os = require("os"), fs2 = require("fs");
const CfT = (() => {
  const base = os.homedir() + "/.cache/puppeteer/chrome";
  try {
    const dirs = fs2.readdirSync(base).filter((d) => d.startsWith("mac_arm-"));
    for (const d of dirs.sort().reverse()) {
      const p = path.join(base, d, "chrome-mac-arm64", "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing");
      if (fs2.existsSync(p)) return p;
    }
  } catch (e) {}
  return null;
})();
const BROWSER = CfT || CHROME;

const argv = process.argv.slice(2);
const url = argv.find((a) => a.startsWith("https://") || a.startsWith("http://") || a.startsWith("file://")) || "https://www.google.com/search?q=cats&tbm=isch";
const scrollN = parseInt((argv.find((a, i) => argv[i - 1] === "--scroll") || "0").replace("--scroll=", "") || "0", 10);
const modeArg = argv.find((a, i) => argv[i - 1] === "--mode") || "";
const mode = modeArg.includes("block") ? "block" : "badge";
const waitMs = parseInt((argv.find((a, i) => argv[i - 1] === "--wait-ms") || "0").replace("--wait-ms=", "") || "0", 10);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  console.log(`\n=== AI Block QA: ${url}  [mode=${mode}] [scrolls=${scrollN}] ===`);
  const browser = await puppeteer.launch({
    executablePath: BROWSER,
    headless: false,
    args: [
      "--no-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPUSwiftShader,AllowWebGPUInInsecureContext",
      "--disable-web-security", // only for local harness; ignore CORS from test driver
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
    ],
    defaultViewport: { width: 1440, height: 900 },
  });

  const page = await browser.newPage();

  // ---- instrument the page for errors + extension signal ----
  const consoleErrors = [];
  const pageErrors = [];
  const unhandled = [];
  const aiBlockLog = { badge: 0, scanned: 0, fetched: 0, cached: 0, blocked: 0, real: 0, fake: 0, errored: 0 };
  const networkFails = [];
  let extensionContextId = null;

  page.on("console", (m) => {
    const t = m.type();
    const txt = m.text();
    // capture AI Block debug counters
    const b = txt.match(/\[AI Block\].*?badge[:=]\s*(\d+)/);
    if (b) aiBlockLog.badge = parseInt(b[1], 10);
    const sc = txt.match(/scanned[:=]\s*(\d+)/);
    if (sc) aiBlockLog.scanned = Math.max(aiBlockLog.scanned, parseInt(sc[1], 10));
    const fe = txt.match(/fetched[:=]\s*(\d+)/);
    if (fe) aiBlockLog.fetched = Math.max(aiBlockLog.fetched, parseInt(fe[1], 10));
    const ca = txt.match(/cached[:=]\s*(\d+)/);
    if (ca) aiBlockLog.cached = Math.max(aiBlockLog.cached, parseInt(ca[1], 10));
    const bl = txt.match(/blocked[:=]\s*(\d+)/);
    if (bl) aiBlockLog.blocked = Math.max(aiBlockLog.blocked, parseInt(bl[1], 10));
    const er = txt.match(/errored[:=]\s*(\d+)/);
    if (er) aiBlockLog.errored = Math.max(aiBlockLog.errored, parseInt(er[1], 10));
    if (t === "error" && !/favicon/i.test(txt)) {
      // ignore benign network/resource + synthetic sample errors
      if (!/Failed to load resource|NotAllowedError: play\(\)|Unexpected token/i.test(txt)) {
        consoleErrors.push(txt);
      }
    }
  });
  page.on("pageerror", (e) => {
    const s = String(e.message || e);
    // Only surface errors that originate from the extension's own content script
    // (our script runs under a locallens/AI-block namespace, or is a JS TypeError
    // unrelated to ads/autoplay). Discount site/3rd-party noise.
    if (/locallens|AI Block|isNotIterable|WeakSet|__locallens|drawBadge|blockImage|detector/i.test(s)) {
      pageErrors.push(s);
    }
  });
  page.on("requestfailed", (r) => {
    if (r.resourceType() === "image") networkFails.push(r.url());
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.request().resourceType() === "image" && !/favicon/i.test(r.url())) {
      networkFails.push(`${r.status()} ${r.url()}`);
    }
  });

  // find the extension service-worker context (to read its logs if needed)
  const targets = await browser.targets();
  for (const t of targets) {
    if (t.type() === "service_worker") extensionContextId = t._targetId;
  }

  console.log("navigating…");
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
  } catch (e) {
    console.log("goto warn:", e.message);
  }
  await sleep(4000 + waitMs);

  // If block mode requested, enable it via the extension's own storage so the
  // content script's block flag flips on. Reach into the extension service
  // worker (if reachable) or via a chrome.storage bridge through an extension
  // page. Simplest reliable path: open the extension's background SW target and
  // set storage there; if that fails, fall through (badge mode).
  if (mode === "block") {
    let engaged = false;
    try {
      // The extension's background is a service worker; find ANY service_worker
      // target (the extension SW is the only one in this isolated profile).
      const extTargets = browser.targets().filter((t) => t.type() === "service_worker");
      if (extTargets.length) {
        const sw = await extTargets[0].worker();
        await sw.evaluate(() => chrome.storage.local.set({ locallens_block: true }));
        engaged = true;
        console.log("[harness] block mode ENABLED via extension SW");
        await sleep(500);
        try { await page.reload({ waitUntil: "domcontentloaded" }); } catch (e) {}
        await sleep(4000 + waitMs);
      }
    } catch (e) {
      console.log("[harness] could not enable block mode:", e.message);
    }
    if (!engaged) {
      // fallback: set via a tab navigated to the extension origin
      try {
        const targets = browser.targets();
        let extId = null;
        for (const t of targets) { const u = (t._targetInfo && t._targetInfo.url) || ""; const m = u.match(/^chrome-extension:\/\/([^/]+)\//); if (m) { extId = m[1]; break; } }
        if (extId) {
          const p2 = await browser.newPage();
          await p2.goto(`chrome-extension://${extId}/background.js`, { timeout: 8000 }).catch(() => {});
          await p2.evaluate(() => chrome.storage.local.set({ locallens_block: true })).catch(() => {});
          await p2.close();
          await page.reload({ waitUntil: "domcontentloaded" }).catch(() => {});
          await sleep(4000 + waitMs);
          console.log("[harness] block mode ENABLED via extension page bridge");
        }
      } catch (e) { console.log("[harness] block-mode fallback failed:", e.message); }
    }
  }

  // ---- scroll in steps to trigger lazy loading ----
  let totalHeight = 2000;
  for (let i = 0; i < scrollN; i++) {
    totalHeight += Math.round(700 + Math.random() * 900);
    try {
      await page.evaluate((h) => window.scrollTo(0, h), totalHeight);
    } catch (e) { /* page navigated mid-scroll (consent/redirect) — skip */ }
    await sleep(900);
  }
  await sleep(2500);

  // ---- sample live page state ----
  let live = { total: 0, badged: 0, visible: 0, blockedRefs: 0, smallBadged: 0 };
  try {
    live = await page.evaluate(() => {
    const imgs = Array.from(document.querySelectorAll("img"));
    const badged = imgs.filter((i) => i.dataset.locallens !== undefined).length;
    const total = imgs.length;
    const visible = imgs.filter((i) => {
      const r = i.getBoundingClientRect();
      return r.top < innerHeight && r.bottom > 0 && r.width > 0 && r.height > 0;
    }).length;
    const blockedRefs = imgs.filter((i) => i.style.display === "none" || i.getAttribute("data-ai-blocked") === "1").length;
    // count images that LOOK like avatars/icons (small) that SHOULD NOT be badged
    const smallBadged = imgs.filter((i) => {
      if (i.dataset.locallens === undefined) return false;
      const r = i.getBoundingClientRect();
      const w = r.width || 0, h = r.height || 0;
      return w > 0 && h > 0 && (w < 100 || h < 100);
    }).length;
    // block-mode card removal check: elements expected to be removed (AI) vs kept (real)
    const gone = ["cardAI", "cardAI2"].filter((id) => !document.getElementById(id));
    const stayed = ["cardReal", "cardReal2"].filter((id) => !!document.getElementById(id));
    return { total, badged, visible, blockedRefs, smallBadged, blockGone: gone, blockStayed: stayed };
    });
  } catch (e) { live = { total: 0, badged: 0, visible: 0, blockedRefs: 0, smallBadged: 0 }; }

  let extensionLoading = false;
  try {
    extensionLoading = await page.evaluate(() => {
      // whether our content script's sentinel is present (a badge element exists in DOM)
      return !!document.querySelector('[data-locallens-badge]') || !!document.querySelector('.locallens-badge') ||
        Array.from(document.querySelectorAll('img')).some((i) => i.dataset.locallens !== undefined);
    });
  } catch (e) { extensionLoading = false; }

  // ---- report ----
  console.log("\n── LIVE PAGE SAMPLE ──");
  console.log(`  images in DOM:        ${live.total}`);
  console.log(`  images visible:       ${live.visible}`);
  console.log(`  badged (data-locallens): ${live.badged}`);
  console.log(`  small/badged (should be 0): ${live.smallBadged}`);
  console.log(`  block-removed refs:   ${live.blockedRefs}`);
  if (live.blockGone) console.log(`  block-mode: cards REMOVED (AI) = ${JSON.stringify(live.blockGone)} | cards KEPT (real) = ${JSON.stringify(live.blockStayed)}`);
  console.log(`  content script active: ${extensionLoading ? "YES" : "NO/UNKNOWN"}`);

  console.log("\n── AI BLOCK COUNTERS (from [AI Block] console) ──");
  console.log(`  ${JSON.stringify(aiBlockLog)}`);

  console.log("\n── ERRORS ──");
  console.log(`  console errors:       ${consoleErrors.length}`);
  consoleErrors.slice(0, 15).forEach((e) => console.log("    ✗", e.slice(0, 200)));
  console.log(`  page errors:          ${pageErrors.length}`);
  pageErrors.slice(0, 10).forEach((e) => console.log("    ✗", e.slice(0, 200)));
  console.log(`  unhandled rejections (page-level, if any): ${unhandled.length}`);
  console.log(`  image network fails:  ${networkFails.length}`);
  networkFails.slice(0, 10).forEach((f) => console.log("    •", f.slice(0, 120)));

  // Classify errors: a FAIL marks only *genuine extension* problems. Benign/
  // expected noise is reported but not counted:
  //  - inline content-script wasm compile errors on strict-CSP pages (the cue to
  //    fall back to offscreen inference), e.g. reddit.com
  //  - site/3rd-party ad/auth noise (token refresh, guest data, doubleclick, …)
  const BENIGN = /wasm streaming compile|initWasm|Aborted\(CompileError|falling back to ArrayBuffer|failed to asynchronously prepare wasm|GSI_LOGGER|tokenRefresh|User not logged in|guest data|unexpected end of json|play\(\) failed|identity provider|doubleclick|rmkt|getuid|Scope is not present|FedCM|403\) was received|SentimentEarnClient/i;
  const realConsole = consoleErrors.filter((e) => !BENIGN.test(e));

  let verdict = "PASS";
  const problems = [];
  if (pageErrors.length) { verdict = "FAIL"; problems.push("page JS errors"); }
  if (realConsole.length) { verdict = "FAIL"; problems.push(`${realConsole.length} genuine console errors`); }
  if (!extensionLoading && live.total > 0) { verdict = "WARN"; problems.push("content script sentinel not detected (may still work)"); }
  if (live.smallBadged > 0) { verdict = "WARN"; problems.push(`${live.smallBadged} small/icon images got badged (should be filtered)`); }
  if (consoleErrors.length && !realConsole.length && verdict === "PASS") { problems.push(`${consoleErrors.length} benign/site console msgs (wasm fallback, ads, auth)`); }

  console.log(`\n── VERDICT: ${verdict}${problems.length ? " — " + problems.join("; ") : ""} ──`);
  await browser.close();
})().catch((e) => { console.error("HARNESS FAILED:", e); process.exit(1); });
