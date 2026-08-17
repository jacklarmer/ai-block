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
    const HI = 2048; // large drawing canvas
    const cv = document.createElement("canvas");
    cv.width = HI;
    cv.height = HI;
    const ctx = cv.getContext("2d");
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    // draw big, in the middle-lower so the emoji (which rises above baseline)
    // stays fully on-canvas regardless of font metrics
    ctx.font = `${Math.round(HI * 1.4)}px "Apple Color Emoji", "Segoe UI Emoji", sans-serif`;
    ctx.fillText("\u{1F916}", HI / 2, HI * 0.85); // 🤖 robot

    // find the true ink bounding box (non-transparent pixels)
    const data = ctx.getImageData(0, 0, HI, HI).data;
    let minX = HI, minY = HI, maxX = -1, maxY = -1;
    for (let y = 0; y < HI; y++) {
      for (let x = 0; x < HI; x++) {
        if (data[(y * HI + x) * 4 + 3] > 8) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    const inkW = maxX - minX + 1, inkH = maxY - minY + 1;
    let side = Math.max(inkW, inkH);
    side = Math.ceil(side * 1.28); // 14% margin around the emoji, even on all sides

    // Composite the ink region baked with its own background-free pixels into a
    // square of `side` px, centered on the ink box. Everything outside is
    // transparent — by construction the emoji is dead-center with even margins.
    const hic = document.createElement("canvas");
    hic.width = side;
    hic.height = side;
    const hctx = hic.getContext("2d");
    hctx.drawImage(cv, minX, minY, inkW, inkH, (side - inkW) / 2, (side - inkH) / 2, inkW, inkH);

    // export at the 3 target sizes
    const out = {};
    for (const s of [16, 48, 128]) {
      const t = document.createElement("canvas");
      t.width = s;
      t.height = s;
      const c2 = t.getContext("2d");
      c2.imageSmoothingEnabled = true;
      c2.imageSmoothingQuality = "high";
      c2.drawImage(hic, 0, 0, s, s);
      out[s] = t.toDataURL("image/png").split(",")[1];
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

