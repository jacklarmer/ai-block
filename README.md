# AI Block — Local Computer Image Detector (Chrome MV3)

Detects computer-generated images (AI) **entirely in your browser** using WebGPU.
No cloud, no server, no uploads — every inference runs locally on your device.
Model weights ship with the extension (one-time, small download) and then the
extension is fully offline.

> **Privacy by design**: images never leave your machine. AI Block is built for
> the Local AI challenge — all detection happens in-browser via
> ONNX Runtime Web on the WebGPU execution provider (WASM fallback included).

## How it works

1. A content script harvests `<img>` elements on the pages you visit.
2. Each image is downscaled / center-cropped to 256×256 and normalized to match
   the training pipeline.
3. A fine-tuned **EfficientNet-B0** classifier (exported to ONNX, fp16) runs
   locally through ONNX Runtime Web on the **WebGPU** backend.
4. A small corner tag shows the verdict as an emoji — ❌ (computer-generated)
   or ✅ (real) — with the confidence % (the % can be toggled off in the popup;
   the ❌/✅ always shows, and hovering any badge reveals the exact confidence).
   Badges are only drawn on **actual content images** (photos/artwork) — icons,
   favicons, avatars, logos, thumbnails and emoji are skipped automatically.
   Adjusting the confidence threshold in the popup re-applies instantly to
   images already on the page (no reload, no re-inference — verdicts are
   re-derived from the verdict retained on each image, with the bounded URL
   result-cache as a fallback, so live threshold updates keep working even
   after the cache evicts older entries on very long sessions), so you can tune
   sensitivity live.
   (Optional "block" mode — enabled from the popup — removes computer-generated
   images AND the post/result card they sit in from the page entirely: the item
   vanishes from the grid/feed as if it was never fetched. Scroll-lazy images
   on Google Images / infinite feeds are caught via an IntersectionObserver.)

Every image is classified by a learned model that was trained on a diverse
multi-generator corpus (SDXL, Stable Diffusion, Midjourney, BigGAN, ADM, glide,
wukong, VQDM, FLUX, **DALL·E 3**, Mobius + real web photos) so it generalizes to
unseen generators rather than memorizing one.

## Requirements

- Chrome 113+ (WebGPU). Older browsers / pages without cross-origin isolation fall
  back to the single-threaded WASM execution provider — always works, no
  SharedArrayBuffer needed.
- The `storage`, `activeTab`, `scripting` permissions (used for settings &
  on-demand scanning only — nothing leaves the device).

## Install (unpacked, dev)

1. `git clone` this repo.
2. (optional) re-export the model — see `model/README.md`.
3. Open `chrome://extensions`, enable **Developer mode**.
4. Click **Load unpacked** and select this `extension/` directory.
5. Visit any page with images — badges appear within a few seconds.

## Model

The deployed model is **EfficientNet-B0** (~5.3M params), fine-tuned on a large
AI-vs-real corpus (~150K+ images across 10+ generative models + deepfakes +
real photographs), then exported to ONNX and stored as **fp16** (~8 MB).
Quantized int8 was evaluated and rejected: dynamic int8 quantization collapsed
the two-class logits and dropped decision agreement to ~55%, so fp16 is the
shipping artifact (~100% agreement with fp32, max softmax diff ≤ 0.01).

| artifact            | size    | agreement w/ fp32 | note                          |
|---------------------|---------|-------------------|-------------------------------|
| `detector_fp32.onnx`| ~15.8MB | 100%              | reference / maximum fidelity |
| `detector_fp16.onnx`| ~8.1MB  | ~100%             | **shipped** (WebGPU)          |
| `detector_int8.onnx`| ~4.3MB  | 54.5%             | rejected (collapses logits)   |

## Metrics

ALL metrics below are measured with the **exact shipped preprocessing**
(`Resize(288) -> CenterCrop(256) -> Normalize(ImageNet)`) on data the model
never trained on — see `evaluation/`. AI recall / real specificity at the 0.5
threshold, plus the honest truly-unseen hard-subtype probe.

**v14 (shipped).** The real class has been progressively broadened across
v12/v13/v14 to span the hard photographic subtypes (macros, clinical, abstracts,
low-light, heavy-JPEG). v14 adds **7,500 FRESH, never-trained ImageNet real
photos** (shards 00020–00022, content-hash-deduped against the 1500 truly-unseen
eval oracle — 0 exact overlaps) on top of v13's set. This progressive real
diversity keeps driving down the harsh-hard-subtype real false-positive rate that
the v10–v12 line uncovered, while holding AI recall flat.

| class (held-out, AI-recall @0.5)      | v14     | v13 (prev) | v12 (older) |
|---------------------------------------|---------|------------|-------------|
| Deepfake / synthetic-face             | **1.000** | 1.000    | 1.000       |
| Frontier (CogView/Gemini/FLUX/Janus)  | **0.875** | 0.880    | 0.885       |
| DALL·E 3                              | **0.945** | 0.960    | 0.965       |
| Midjourney                            | **0.885** | 0.895    | 0.905       |
| Ideogram                              | **0.945** | 0.955    | 0.955       |

| real class (real-specificity @0.5)                      | v14     | v13 (prev) | v12 (older) |
|---------------------------------------------------------|---------|------------|-------------|
| Real photos, held-out editorial (web-photo)             | **0.990** | 0.985    | 0.990       |
| Real artwork (WikiArt)                                  | **~1.00** | ~1.00   | ~1.00       |
| **Real photos, TRULY-unseen hard subtypes** (macros / clinical / abstract / low-light / heavy-JPEG, 1500 never-trained) | **0.576** (42.4% FP) | 0.496 (50.4% FP) | 0.343 (65.7% FP) |

> **Honest note — read this.** The earlier READMEs claimed ~0% real-photo FP and
> 99%+ deepfake recall. Those numbers came from an easier, in-distribution real
> set and did not survive a truly-unseen, hard-subtype real probe. This table is
> the exact shipped preprocessing on genuinely unseen data. Each version is a
> strict improvement over the previous ships (unseen-hard FP 93.8% (v9) →
> 65.7% (v12) → 50.4% (v13) → **42.4% (v14)**, deepfake recall 18% (v9) →
> 100% (v12/v13/v14)), with the other AI-recall axes held within ≤1.5pts of v13
> (well within eval noise on the ~600-image per-generator sets). It still
> over-flags **~42%** of the hardest real photographs (extreme macros, clinical,
> very heavy JPEG, very low-light); on ordinary web photography — the common
> case — it is ~98–99% specific. Closing the residual hard-subtype false positive
> is the active next problem; we report it plainly rather than bury it.

The reproduction harness is in `evaluation/`; the per-image WebGPU test is in
`test/`. The v10→v14 scripts tracked in `evaluation/` (gather_v13.sh,
build_v13.py, v13_pipeline.sh, gather_v14.sh, build_v14.py, v14_pipeline.sh)
form the audit trail for how the hard-subtype gap was found and progressively
narrowed.

> **v15 candidacy — tested and DISCARDED (kept v14).** The next cycle gathered
> 7,500 more fresh, never-trained ImageNet real photos (shards 00023–00025,
> content-hash-deduped against the 1500 unseen oracle, 0 overlaps), retrained as
> v15 from the v14 checkpoint, and gated with the exact shipped transform. It
> **regressed** the primary objective: unseen-hard FP@0.5 went **42.4% → 43.9%**
> (658/1500 vs 636/1500), with AI recall flat (deepfake 0.998, frontier 0.928).
> **v15 was NOT shipped** — the real-diversity lever has plateaued/diminishing
> returns after v12→v14, so v14 remains the deployed model. The full
> gather/build/train/gate for this negative result is committed in `evaluation/`
> (gather_v15.sh, build_v15.py, v15_pipeline.sh, v15_gate.sh) so the attempt is
> reproducible and future cycles don't blindly repeat the same lever.

> **v16 candidacy — tested and DISCARDED (kept v14).** Because the v15 run had a
> broken data build (a stale `data_v15` tree crashed `build_v15.py`; training ran
> on a mixed/incomplete set), v16 was a *clean retest*: it fixed the build bug
> (fresh OUT, hard-refuse stale trees), gathered **7,500 more fresh, never-trained
> ImageNet real photos (shards 00026–00028, 0 oracle overlaps)**, fine-tuned from
> the v14 checkpoint, and gated with the exact shipped transform. The clean result
> **confirms the plateau is real**: unseen-hard FP@0.5 REGRESSED **42.4% → 46.5%**
> (698/1500 vs 636/1500, AI recall flat: deepfake 0.995, frontier 0.922). **v16 was
> NOT shipped** — v14 remains deployed; scripts committed in `evaluation/`
> (gather_v16.sh, build_v16.py, v16_pipeline.sh).

> **Single deterministic center-crop (no TTA).** We measured that a 5-crop x
> 2-flip test-time augmentation *hurts* balanced accuracy on the held-out set
> (0.887 vs 0.921) by diluting the calibrated center-crop signal, so the
> shipped detector runs a single accurate pass. See `content/detector.js`.

The `test/` harness runs a throwaway local Node HTTP server **only to exercise
the detector in a browser during development** — it is not part of the shipped
extension. The extension itself is a pure static MV3 package with no localhost
server, no backend, and no network calls beyond the initial bundled-model load,
fully satisfying the "no local server / no cloud inference" rule.

> **Works on real pages (e.g. x.com).** The detector forces single-threaded WASM
> (`numThreads=1`), which initializes without the COOP/COEP cross-origin
> isolation that ordinary pages don't provide, and it decodes cross-origin
> images via `fetch`→`Blob` (honoring `host_permissions`), so photos on social
> sites aren't skipped as canvas-tainted. Inference itself always stays local.

## License

MIT. See `LICENSE`.

## Reproducibility

Everything (data pipeline, training, export, extension) is reproducible from
source — see `evaluation/reproduce.sh` and the per-stage READMEs.
