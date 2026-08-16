# Model artifacts

`detector.onnx` is the **shipped** model — **v13**, an EfficientNet-B0 fine-tuned
for the real-world AI-vs-real discrimination task:

- **AI / synthetic class** — deepfake / synthetic-face (95k), frontier generators
  (CogView4, Gemini flash-image, FLUX, Janus-Pro, RealVisXL), + 7+ established
  generators (DALL·E 3, Midjourney, Ideogram, Aura, Imagine, Leonardo, ...).
- **Real class** — broadened across v12/v13 to span the hard photographic
  subtypes (macros, clinical, abstract, low-light, heavy-JPEG) with ~51K diverse
  ImageNet photos so "real photography" is anchored across the full spectrum.

Exported to ONNX in **fp16** (~8 MB). v13 is fine-tuned from v12 with **17,500
FRESH, never-trained ImageNet real photos** (shards 00000–00002, 00048–00051)
joined to the real class — the lever that cut the truly-unseen hard-subtype
real false-positive rate a further 15 points.

## Honest numbers (EXACT shipped preprocessing: Resize(288)→CenterCrop(256)→Normalize)

Measured with the identical preprocessing the extension uses, on data the model
never trained on:

| class (unseen / held-out)         | AI-recall | real-spec | v12 (prev ship) | v9 (older) |
|-----------------------------------|-----------|-----------|------------------|------------|
| Deepfake / synthetic-face         | **1.000** | —         | 1.000            | 0.18       |
| Frontier (CogView/Gemini/FLUX)    | **0.880** | —         | 0.885            | 0.90       |
| DALL·E 3                          | **0.960** | —         | 0.965            | 0.835      |
| Midjourney                        | **0.895** | —         | 0.905            | 0.735      |
| Ideogram                          | **0.955** | —         | 0.955            | 0.915      |
| Real, held-out editorial (COCO)   | —         | **0.985** | 0.99             | 0.34       |
| **Real, TRULY-unseen hard subtypes** (macros/clinical/abstract/low-light/heavy-JPEG, 1500 never-trained) | — | **0.496** (50.4% FP) | 0.343 (65.7% FP) | 0.062 (93.8% FP) |

> **Honest note — read this.** Earlier READMEs claimed ~0% real-photo FP and
> 99%+ deepfake recall. Those numbers came from an easier, in-distribution real
> set and did not survive a truly-unseen, hard-subtype probe. This table is the
> result of the exact shipped preprocessing on genuinely unseen data. v13 is a
> strict, large improvement over the previous ships (unseen-hard FP 93.8% (v9)
> → 65.7% (v12) → **50.4% (v13)**, deepfake recall 18% (v9) → 100% (v12/v13)),
> with AI recall held within noise. It still over-flags **~50%** of the hardest
> real photographs (extreme macros, clinical, very heavy JPEG, very low-light);
> on ordinary web photography it is ~98–99% specific. Closing the residual
> hard-subtype false positive is the active next problem; we report it plainly
> rather than bury it.

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
# v13 (current shipped): broadened real class, fine-tune, export
python evaluation/gather_v13.sh        # 17,500 fresh ImageNet real shards
python evaluation/build_v13.py data_v13_more data_unseen/real_eval 2
python evaluation/train_v4.py --root data_v13/train --ckpt run_v12/best.pt --out run_v13 --epochs 8 --batch 80
python evaluation/export_onnx.py --ckpt run_v13/best.pt --out model/detector.onnx
# gate (exact preprocessing)
python evaluation/v13_pipeline.sh
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), where index 1 = "AI / fake" (softmax in the extension).
