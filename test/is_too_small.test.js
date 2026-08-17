/**
 * AI Block — unit test for the `isTooSmall` size/icon predicate.
 *
 * Regression test for a real bug: a lazy-loaded image that is collected BEFORE
 * it has decoded reports naturalWidth===0. The old intrinsic-size check
 * `(naturalWidth||0)*(naturalHeight||0) < MIN_IMG_AREA` treated that as a tiny
 * icon and returned `true` — which permanently flushed the image into the
 * `scanned` set (never re-evaluated), so a large real photo that was mid-load
 * would be silently skipped and never badged.
 *
 * The fix gates the intrinsic-tiny test on `img.complete`: we only rule an
 * image "too small" by its intrinsic size once it has actually decoded. Until
 * then the fetch + createImageBitmap step in the processing loop classifies it
 * with its real dimensions.
 *
 * This test evaluates the ACTUAL `isTooSmall` source from content/content.js so
 * it can never drift from what ships.
 *
 * Run: node test/is_too_small.test.js
 */
const fs = require("fs");
const path = require("path");

const SRC = path.resolve(__dirname, "..", "content", "content.js");
const code = fs.readFileSync(SRC, "utf8");

function grab(name) {
  // Match `const NAME = <expr>;` up to the terminating semicolon (no template
  // literals/nested semis in our constants) or `function NAME(...) {...}`.
  const c = new RegExp(`const\\s+${name}\\s*=\\s*([^;]+);`).exec(code);
  if (c) return c[1];
  const f = new RegExp(`function\\s+${name}[\\s\\S]*?\\n\\s*}`).exec(code);
  if (f) return f[0];
  throw new Error(`could not extract ${name}`);
}

// Mini sandbox: define the constants isTooSmall closes over, then eval it.
const fn = new Function("MIN_RENDERED", "MIN_CONTENT_AREA", "MIN_IMG_AREA",
  "return (" + grab("isTooSmall") + ");");
const isTooSmall = fn(
  eval(grab("MIN_CONTENT_DIM")),   // MIN_RENDERED = MIN_CONTENT_DIM
  eval(grab("MIN_CONTENT_AREA")),
  eval(grab("MIN_IMG_AREA"))
);

let pass = 0, fail = 0;
function t(name, got, want) {
  const ok = got === want;
  ok ? pass++ : fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: got=${got} want=${want}`);
}

// ---- case 1 (THE BUG): large lazy photo collected before decode ----
// on-screen 400x400 CSS box (so rect checks pass), but natural size still 0
// because it hasn't decoded yet. Must NOT be treated as too small.
t("large lazy photo, not yet decoded (naturalW=0) -> NOT too small",
  isTooSmall({ complete: false, naturalWidth: 0, naturalHeight: 0,
               getBoundingClientRect: () => ({ width: 400, height: 400 }) }),
  false);

// ---- case 2: same image AFTER decode (now the real size is visible) ----
// 400x400 decoded, not an icon -> still not too small.
t("large photo after decode -> NOT too small",
  isTooSmall({ complete: true, naturalWidth: 800, naturalHeight: 500,
               getBoundingClientRect: () => ({ width: 400, height: 400 }) }),
  false);

// ---- case 3: genuine tiny favicon, decoded (16x16) -> too small ----
t("decoded 16x16 favicon -> too small",
  isTooSmall({ complete: true, naturalWidth: 16, naturalHeight: 16,
               getBoundingClientRect: () => ({ width: 16, height: 16 }) }),
  true);

// ---- case 4: on-screen tiny icon (rendered 24x24) even if not decoded ----
// rect is genuinely small (24x24 < 100 in both) -> too small regardless of
// decode state (catches icons by their on-screen box, not just intrinsic size).
t("tiny on-screen 24x24, not decoded -> too small (by rect)",
  isTooSmall({ complete: false, naturalWidth: 0, naturalHeight: 0,
               getBoundingClientRect: () => ({ width: 24, height: 24 }) }),
  true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
