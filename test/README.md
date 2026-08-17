# AI Block — real-world QA harness

Automated end-to-end QA for the extension against real, image-heavy websites.
Loads the extension as a real unpacked extension in a Chrome-for-Testing browser,
scrolls the page (triggering lazy loading), then verifies detection + badging +
block-mode behavior and reports any errors.

> Stable branded Chrome refuses `--load-extension` (since ~Chrome 137), so the
> harness requires a **Chrome for Testing** binary. Install one with:
> ```bash
> node -e "require('@puppeteer/browsers').install({browser:'chrome',buildId:'stable',cacheDir:require('os').homedir()+'/.cache/puppeteer'})"
> ```
> The harness auto-detects the newest Chrome-for-Testing in
> `~/.cache/puppeteer/chrome/mac_arm-*/`.

## Usage

Serve the local fixtures (a tiny static server that adds CORS headers so the
content script can read the served images):

```bash
node test/_server.cjs            # serves test/ on http://127.0.0.1:8899
```

Run one site (badge mode is default; `--mode block` enables the block toggle):

```bash
node test/site_qa.js "https://www.reddit.com/r/pics/" --scroll 6 --mode badge
node test/site_qa.js "http://127.0.0.1:8899/test/fixtures/infinite_scroll.html" --scroll 8
node test/site_qa.js "http://127.0.0.1:8899/test/fixtures/block_mode.html" --mode block
```

Run a batch and print just verdicts:

```bash
node test/batch_qa.js "https://en.wikipedia.org/wiki/Cat" "https://www.bbc.com/news"
```

## What it checks

- Content script injects and its sentinel is present
- `badged` count grows as the page is scrolled (lazy-load coverage)
- `small/badged` == 0 (avatars/icons/favicons must NOT be badged)
- Block mode removes AI-classified images (DOM image count drops) while real
  images stay
- No console / page errors and no unhandled rejections from the *extension*
  (site/3rd-party ad noise is reported but not a failure)

## Unit tests (no browser needed)

- `test/is_too_small.test.js` — regression test for the image-harvesting size
  predicate (`isTooSmall`) extracted from `content/content.js`. Verifies that a
  lazy-loaded image collected before it has decoded (natural size still 0) is
  not mistaken for a tiny icon and permanently skipped, while genuinely tiny
  decoded icons/favicons still are. Run: `node test/is_too_small.test.js`

## Fixtures

- `test/fixtures/infinite_scroll.html` — deterministic infinite-scroll feed with
  lazy-loaded images + avatars/icons that must not be badged
- `test/fixtures/block_mode.html` — cards carrying real AI-sample images
  (`samples/fake/flux_*.jpg`) + real images; verifies block-mode card removal

## Architecture note

Inference runs in an **offscreen document** (`offscreen.html`/`offscreen.js`),
not the page's content-script world. Why: some sites (e.g. reddit.com) send a
Content-Security-Policy that forbids WebAssembly compilation inside a content
script, and also refuse to serve their images cross-origin (canvas taint). The
offscreen document compiles wasm under the extension's own CSP
(`'wasm-unsafe-eval'`) and fetches images using the extension's `<all_urls>`
host permission, bypassing both the page CSP and CORS-read restrictions.
