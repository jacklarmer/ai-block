/**
 * AI Block logo generator (emoji version — per Jack: just the robot emoji 🤖
 * alone, no NO/prohibition overlay).
 *
 * Renders the real robot emoji (🤖) using Chrome's native emoji font, then
 * exports crisp PNGs at 16/48/128.
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const EXT = path.resolve(__dirname, "..");
const ICONS = path.join(EXT, "icons");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true, // we only need canvas rendering, WebGPU not required
    args: ["--no-sandbox", "--enable-emoji-rendering"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 512, height: 512, deviceScaleFactor: 8 }); // 4096 supersample

  const pngs = await page.evaluate(() => {
    const HI = 4096; // supersampled canvas (8x of 512)
    const cv = document.createElement("canvas");
    cv.width = HI;
    cv.height = HI;
    const ctx = cv.getContext("2d");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Robot emoji alone, filling most of the canvas
    ctx.font = `${Math.round(HI * 0.94)}px "Apple Color Emoji", "Segoe UI Emoji", sans-serif`;
    ctx.fillText("\u{1F916}", HI / 2, HI / 2); // 🤖 robot

    // export at the 3 target sizes
    const out = {};
    for (const s of [16, 48, 128]) {
      const tmp = document.createElement("canvas");
      tmp.width = s;
      tmp.height = s;
      const c2 = tmp.getContext("2d");
      c2.imageSmoothingEnabled = true;
      c2.imageSmoothingQuality = "high";
      c2.drawImage(cv, 0, 0, s, s);
      out[s] = tmp.toDataURL("image/png").split(",")[1];
    }
    return out;
  });

  fs.mkdirSync(ICONS, { recursive: true });
  for (const s of [16, 48, 128]) {
    const p = path.join(ICONS, `icon${s}.png`);
    fs.writeFileSync(p, Buffer.from(pngs[s], "base64"));
    console.log("wrote", p, s + "x" + s);
  }
  await browser.close();
  console.log("done");
})().catch((e) => { console.error(e); process.exit(1); });

