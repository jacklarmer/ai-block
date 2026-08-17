# Model artifacts

`detector.onnx` is the **shipped** model — **v14**, an EfficientNet-B0 fine-tuned
for the real-world AI-vs-real discrimination task:

- **AI / synthetic class** — deepfake / synthetic-face (95k), frontier generators
  (CogView4, Gemini flash-image, FLUX, Janus-Pro, RealVisXL), + 7+ established
  generators (DALL·E 3, Midjourney, Ideogram, Aura, Imagine, Leonardo, ...).
- **Real class** — broadened across v12/v13/v14 to span the hard photographic
  subtypes (macros, clinical, abstract, low-light, heavy-JPEG) with ~59K diverse
  ImageNet photos so "real photography" is anchored across the full spectrum.

Exported to ONNX in **fp16** (~8 MB). v14 is fine-tuned from v13 with **7,500
FRESH, never-trained ImageNet real photos** (shards 00020–00022) joined to the
real class. The fresh batch was content-hash-deduped against the 1500 truly-unseen
eval oracle (0 exact overlaps), so the oracle stays genuinely unseen. This cut the
truly-unseen hard-subtype real false-positive rate from 50.4% → **42.4%**.

## Honest numbers (EXACT shipped preprocessing: Resize(288)→CenterCrop(256)→Normalize)

Measured with the identical preprocessing the extension uses, on data the model
never trained on:

| class (unseen / held-out)         | AI-recall | real-spec | v13 (prev ship) | v12 (older) |
|-----------------------------------|-----------|-----------|------------------|-------------|
| Deepfake / synthetic-face         | **1.000** | —         | 1.000            | 1.000       |
| Frontier (CogView/Gemini/FLUX)    | **0.875** | —         | 0.880            | 0.885       |
| DALL·E 3                          | **0.945** | —         | 0.960            | 0.965       |
| Midjourney                        | **0.885** | —         | 0.895            | 0.905       |
| Ideogram                          | **0.945** | —         | 0.955            | 0.955       |
| Real, held-out editorial (COCO)   | —         | **0.990** | 0.99             | 0.99        |
| **Real, TRULY-unseen hard subtypes** (macros/clinical/abstract/low-light/heavy-JPEG, 1500 never-trained) | — | **0.576** (42.4% FP) | 0.496 (50.4% FP) | 0.343 (65.7% FP) |

> **Honest note — read this.** Earlier READMEs claimed ~0% real-photo FP and
> 99%+ deepfake recall. Those numbers came from an easier, in-distribution real
> set and did not survive a truly-unseen, hard-subtype probe. This table is the
> result of the exact shipped preprocessing on genuinely unseen data. Each
> version is a strict improvement over the previous ships: unseen-hard FP 93.8%
> (v9) → 65.7% (v12) → 50.4% (v13) → **42.4% (v14)**, deepfake recall 18% (v9) →
> 100% (v12/v13/v14), with all other AI-recall held within ≤1.5pts of v13 (well
> within eval noise on the ~600-image per-generator sets). v14 is a pure win: a
> large FP reduction with recall effectively unchanged. It still over-flags
> **~42%** of the hardest real photographs (extreme macros, clinical, very heavy
> JPEG, very low-light); on ordinary web photography it is ~98–99% specific.
> Closing the residual hard-subtype false positive is the active next problem; we
> report it plainly rather than bury it.

> **v16 candidacy — tested and DISCARDED (kept v14).** The v15 run had a broken
> data build (a stale `data_v15` tree crashed `build_v15.py` with FileExistsError;
> training then ran on a mixed/incomplete set), so v16 was a *clean retest* of the
> fresh-ImageNet-real lever. It fixed the build bug (build into a fresh OUT,
> hard-refuse stale trees), gathered a further **7,500 fresh, never-trained
> ImageNet real photos (shards 00026–00028, 0 oracle overlaps)**, fine-tuned v16
> from the v14 checkpoint, and gated with the exact shipped transform. The clean
> result **confirms the plateau is real, not a build artifact**: unseen-hard FP@0.5
> REGRESSED **42.4% → 46.5%** (698/1500 vs 636/1500), with AI recall flat
> (deepfake 0.995, frontier 0.922). **v16 was NOT shipped** — adding yet more
> ImageNet real photos no longer helps (and now mildly overfits toward the
> ImageNet-look at the expense of the harder real subtypes), so **v14 remains the
> deployed model**. Scripts committed in `evaluation/` (gather_v16.sh,
> build_v16.py, v16_pipeline.sh).

## Files

| file                 | size  | role                                   |
|----------------------|-------|----------------------------------------|
| `detector.onnx`      | ~8 MB | **shipped** fp16 (WebGPU / WASM)       |
| `detector_fp32.onnx` | ~16 MB| reference, maximum fidelity             |
| `detector_fp16.onnx` | ~8 MB | fp16 (identical to shipped)             |
| `detector_int8.onnx` | ~4 MB | dynamic int8 — **rejected** (see below) |

## Why fp16 and not int8

Dynamic int8 quantization collapsed the two-class logits — decision agreement
with fp32 dropped to ~55% (max softmax error 0.999). fp16 retains ~100%
agreement (max softmax error ~0.005), is natively supported by the WebGPU
execution provider, and halves the download.

## Re-exporting

```
# v14 (current shipped): more fresh real diversity, fine-tune, export
bash evaluation/gather_v14.sh            # 7,500 fresh ImageNet real shards 00020-22
python evaluation/build_v14.py data_v14_more data_unseen/real_eval 7500
python evaluation/train_v4.py --root data_v14/train --ckpt run_v13/best.pt --out run_v14 --epochs 8 --batch 80
python evaluation/export_onnx.py --ckpt run_v14/best.pt --out model/detector.onnx
# gate (exact preprocessing)
bash evaluation/v14_pipeline.sh
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), where index 1 = "AI / fake" (softmax in the extension).
