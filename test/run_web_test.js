/**
 * LocalLens end-to-end WebGPU test harness.
 *
 * Serves the extension's onnxruntime-web + detector artifacts over a local HTTP
 * server, opens a real Chrome tab (headless, WebGPU enabled), then uses the SAME
 * detector.js preprocessing + inference path to classify a labeled set of sample
 * images. Prints per-image predicted fake-probability and the aggregate balanced
 * accuracy at the 65% confidence threshold.
 *
 * Usage:
 *   node test/run_web_test.js [path-to-model.onnx] [samplesDir]
 */
const fs = require("fs");
const path = require("path");
const http = require("http");
const puppeteer = require("puppeteer-core");

const EXT = path.resolve(__dirname, "..");
const MODEL = path.resolve(process.argv[2] || path.join(EXT, "model", "detector.onnx"));
const SAMPLES = path.resolve(process.argv[3] || path.join(EXT, "samples"));
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".css": "text/css", ".json": "application/json",
};

// --- tiny static server rooted at EXT ---
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url.endsWith(".wasm") || url.endsWith(".onnx") || url.endsWith(".js") || url.includes("detector")) console.log("[req]", req.url);
  let p = path.normalize(path.join(EXT, url));
  if (!p.startsWith(EXT)) { res.writeHead(403); return res.end(); }
  if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) { if(url.endsWith('.wasm')) console.log("[404]", url, "->", p); res.writeHead(404); return res.end(); }
  const ext = path.extname(p);
  // NOTE: We deliberately do NOT send COOP/COEP here. Real pages (twitter.com)
  // don't have cross-origin-isolation headers, so this reproduces the actual
  // content-script environment and proves the single-threaded wasm path works
  // without SharedArrayBuffer.
  res.writeHead(200, {
    "Content-Type": MIME[ext] || "application/octet-stream",
    "Access-Control-Allow-Origin": "*",
  });
  fs.createReadStream(p).pipe(res);
});

function serve() {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

async function gatherSamples() {
  const real = [];
  const fake = [];
  for (const dir of fs.readdirSync(SAMPLES)) {
    const full = path.join(SAMPLES, dir);
    if (!fs.statSync(full).isDirectory()) continue;
    for (const f of fs.readdirSync(full)) {
      if (/\.(png|jpe?g)$/i.test(f)) {
        const rec = `/samples/${dir}/${f}`;
        (dir.startsWith("real") || dir === "real" ? real : fake).push(rec);
      }
    }
  }
  return { real, fake };
}

(async () => {
  const port = await serve();
  const { real, fake } = await gatherSamples();
  console.log(`real samples: ${real.length}, fake samples: ${fake.length}`);
  if (real.length === 0 || fake.length === 0) {
    console.error("Need both samples/real and samples/fake images.");
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: false, // WebGPU requires a real GPU surface; headless lacks navigator.gpu
    args: [
      "--no-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan,WebGPUSwiftShader,AllowWebGPUInInsecureContext",
      "--enable-webgpu-developer-features",
    ],
  });
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
  page.on("response", (r) => { if (r.status() >= 400) console.log("[404]", r.status(), r.url()); });

  const modelUrl = `http://127.0.0.1:${port}/model/${path.basename(MODEL)}`;
  const pageUrl = `http://127.0.0.1:${port}/test/web_runner.html?model=${encodeURIComponent(modelUrl)}`;

  await page.goto(pageUrl, { waitUntil: "networkidle0", timeout: 60000 });

  // warm up + classify
  const results = await page.evaluate(async ({ real, fake }) => {
    const out = { real: [], fake: [] };
    const classifyOne = async (kind, src) => {
      const img = new Image();
      img.src = src;
      await img.decode();
      const r = await window.runDetect(img);
      out[kind].push({ src, fake: r.fake, label: r.label, ms: r.ms });
    };
    try {
      for (const s of real) { await classifyOne("real", s); }
      for (const s of fake) { await classifyOne("fake", s); }
      return { out };
    } catch (e) {
      return { error: String(e && e.stack || e) };
    }
  }, { real, fake });

  if (results.error) {
    console.error("detection failed:", results.error);
    await browser.close();
    process.exit(1);
  }

  // --- aggregate: balanced accuracy at 65% confidence threshold ---
  const { real: rr, fake: ff } = results.out;
  const thr = 0.65;
  const preds = [];
  for (const r of rr) preds.push({ label: 0, fake: r.fake });
  for (const f of ff) preds.push({ label: 1, fake: f.fake });

  // choose decisions at the 65% confidence gate (only flag AI when fake>=thr)
  let tp = 0, fp = 0, tn = 0, fn = 0, covered = 0;
  for (const p of preds) {
    const callAI = p.fake >= thr;
    const conf = Math.max(p.fake, 1 - p.fake);
    if (conf >= thr) covered++;               // "confident" sample
    if (callAI && p.label === 1) tp++;
    else if (callAI && p.label === 0) fp++;
    else if (!callAI && p.label === 0) tn++;
    else if (!callAI && p.label === 1) fn++;
  }
  const tpr = tp / Math.max(1, (tp + fn));
  const tnr = tn / Math.max(1, (tn + fp));
  const baccAll = (tpr + tnr) / 2;
  console.log(`\n[real] n=${rr.length}  [fake] n=${ff.length}`);
  console.log(`decisions (thr=${thr}): TP=${tp} FP=${fp} TN=${tn} FN=${fn}`);
  console.log(`balanced acc @65% = ${baccAll.toFixed(4)}  (all samples)`);
  console.log(`coverage = ${(covered / preds.length).toFixed(3)} (${covered}/${preds.length})`);

  // also dump per-image for a few
  const sample = 8;
  console.log("\n-- sample real predictions --");
  rr.slice(0, sample).forEach((r) => console.log(`  fake=${r.fake.toFixed(3)} ${r.label.padEnd(10)} ${r.src} (${r.ms}ms)`));
  console.log("-- sample fake predictions --");
  ff.slice(0, sample).forEach((f) => console.log(`  fake=${f.fake.toFixed(3)} ${f.label.padEnd(10)} ${f.src} (${f.ms}ms)`));

  await browser.close();
  server.close();
})().catch((e) => { console.error(e); process.exit(1); });
