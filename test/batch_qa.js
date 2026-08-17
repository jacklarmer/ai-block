// Batch-run site_qa across multiple URLs, print just verdict + key numbers.
const { spawn } = require("child_process");
const path = require("path");
const qa = path.join(__dirname, "site_qa.js");

const cases = process.argv.slice(2);
if (!cases.length) {
  console.log("usage: node test/batch_qa.js <url> [<url> ...]");
  process.exit(1);
}

function run(url) {
  return new Promise((resolve) => {
    const p = spawn("node", [qa, url, "--scroll", "8", "--mode", "badge", "--wait-ms", "1500"], {
      cwd: path.join(__dirname, ".."), stdio: ["ignore", "pipe", "pipe"],
    });
    let out = "";
    p.stdout.on("data", (d) => (out += d.toString()));
    p.stderr.on("data", (d) => (out += d.toString()));
    p.on("close", () => {
      // extract key lines
      const get = (re) => { const m = out.match(re); return m ? m[1].trim() : "?"; };
      const verdict = get(/VERDICT:\s*(\S+)/);
      const images = get(/images in DOM:\s+(\d+)/);
      const badged = get(/badged \(data-locallens\):\s+(\d+)/);
      const small = get(/small\/badged \(should be 0\):\s+(\d+)/);
      const consErr = get(/console errors:\s+(\d+)/);
      const pageErr = get(/page errors:\s+(\d+)/);
      const netF = get(/image network fails:\s+(\d+)/);
      resolve({ url, verdict, images, badged, small, consErr, pageErr, netF });
    });
  });
}

(async () => {
  const results = [];
  for (const url of cases) {
    const r = await run(url);
    results.push(r);
    console.log(`[${r.verdict}] ${r.url}  imgs=${r.images} badged=${r.badged} smallBadged=${r.small} consErr=${r.consErr} pageErr=${r.pageErr} netFails=${r.netF}`);
  }
  console.log("\n=== SUMMARY ===");
  for (const r of results) console.log(`  ${r.verdict.padEnd(5)} ${r.url}`);
})().catch((e) => { console.error(e); process.exit(1); });
