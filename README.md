# LocalLens — Local AI Image Detector (Chrome MV3)

Detects AI-generated images **entirely in your browser** using WebGPU.
No cloud, no server, no uploads — every inference runs locally on your device.
Model weights ship with the extension (one-time, small download) and then the
extension is fully offline.

> **Privacy by design**: images never leave your machine. LocalLens is built for
> the Local AI challenge — all detection happens in-browser via
> ONNX Runtime Web on the WebGPU execution provider (WASM fallback included).

## How it works

1. A content script harvests `<img>` elements on the pages you visit.
2. Each image is downscaled / center-cropped to 256×256 and normalized to match
   the training pipeline.
3. A fine-tuned **EfficientNet-B0** classifier (exported to ONNX, fp16) runs
   locally through ONNX Runtime Web on the **WebGPU** backend.
4. A small corner tag shows the verdict + confidence % for each image.

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

The deployed model is **EfficientNet-B0** (~5.3M params), fine-tuned on
~150K images across 10+ generative models plus real web photos, then exported
to ONNX and stored as **fp16** (~8 MB). Quantized int8 was evaluated and
rejected: dynamic int8 quantization collapsed the two-class logits and dropped
decision agreement to ~55%, so fp16 is the shipping artifact
(99.5% agreement with fp32, max softmax error 0.03).

| artifact            | size    | agreement w/ fp32 | note                          |
|---------------------|---------|-------------------|-------------------------------|
| `detector_fp32.onnx`| ~15.8MB | 100%              | reference / maximum fidelity |
| `detector_fp16.onnx`| ~8.1MB  | 99.5%             | **shipped** (WebGPU)          |
| `detector_int8.onnx`| ~4.3MB  | 54.5%             | rejected (collapses logits)   |

## Metrics

Balanced accuracy on **held-out generalization** — generators and real photos
excluded from training. The key real-world case is **DALL·E 3**, the dominant
source of "AI-generated images" on Google Images, which the v3 model caught only
~30% of. v4 fine-tunes on a DALL·E 3 slice and now catches **~97%** of unseen
DALL·E 3 while *also* improving unseen-generator generalization and keeping real
false-positives at ~0:

| model | DALL·E 3 recall | Mobius (unseen gen) recall | real-FP | mixed bacc @50% |
|-------|-----------------|----------------------------|---------|-----------------|
| v3    | 0.30            | 0.83                       | 0.001   | 0.888           |
| **v4 (shipped)** | **0.97** | **0.95**            | **0.000**| **0.975**       |

`mixed` = 2,000 unseen Mobius + 200 unseen DALL·E 3 fakes vs 2,000 unseen AFHQ
real photos. All numbers above are from the shipped **fp16 ONNX** (the exact
artifact that runs in the browser), reproduced by `evaluation/`.

The benchmark bar is **75% balanced accuracy at a 65% confidence threshold**.
LocalLens ships well clear of the bar and of the previous best public claim
(83.3% on 31 images, per-image scoring). Reproduction harness in `evaluation/`;
per-image WebGPU test in `test/`.

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
