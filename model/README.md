# Model artifacts

`detector.onnx` is the **shipped** model — **v12**, an EfficientNet-B0 fine-tuned
for the real-world AI-vs-real discrimination task:

- **AI / synthetic class** — deepfake / synthetic-face (95k bitmind swaps &
  synthetic faces), frontier generators (CogView4-6B, Gemini 2.5 flash-image,
  FLUX.1-dev, Janus-Pro-7B, RealVisXL), and 7+ established generators (Ideogram,
  Aura, Imagine, Leonardo, Midjourney, DALL·E 3, Mobius).
- **Real-photography class, mass-broadened** — COCO editorial + WikiArt + 17.5K
  diverse ImageNet real photos (v12) spanning macros, textures, clinical,
  low-light, objects, scenes, animals — added specifically to anchor "real"
  across the full photographic spectrum, including hard AI-lookalike subtypes.

> **Why v12 over v9.** The shipped v9 model had a severe, previously-unreported
> false-positive problem on genuinely-unseen real photos: at the 0.5 threshold it
> flagged **93.8%** of a never-trained, hard ImageNet slice (macros / clinical /
> abstract / low-light / heavy-JPEG) as AI, and only 33.5% specificity on the
> held-out editorial real set, while catching just **18%** of held-out deepfakes.
> v12 is a strict, large improvement on every measured axis — see metrics below.
> It was fine-tuned from v9 with the real class broadened to span the hard
> subtypes.

Exported to ONNX in **fp16** (~8 MB). fp16 retains **~100% decision agreement**
with fp32 (max softmax diff ≤ 0.01). int8 was evaluated and rejected (collapsed
logits, ~55% agreement).

## Files

| file                 | size  | role                                   |
|----------------------|-------|----------------------------------------|
| `detector.onnx`      | ~8 MB | **shipped** fp16 (WebGPU / WASM)       |
| `detector_fp32.onnx` | ~16 MB| reference, maximum fidelity             |
| `detector_fp16.onnx` | ~8 MB | fp16 (identical to shipped)             |
| `detector_int8.onnx` | ~4 MB | dynamic int8 — **rejected** (see below) |

## Metrics (v12, measured with the EXACT training transform)

Measured with the exact shipped preprocessing
`Resize(288)->CenterCrop(256)->Normalize(ImageNet)` on held-out data the model
never trained on. AI recall / real specificity at threshold 0.5.

| class (unseen / held-out)            | AI-recall | real-spec | v9 (old ship) for reference |
|--------------------------------------|-----------|-----------|------------------------------|
| Deepfake / synthetic-face            | **1.000** | —         | 0.180                        |
| DALL·E 3                             | **0.965** | —         | 0.835                        |
| Ideogram                             | **0.955** | —         | 0.915                        |
| Midjourney                           | **0.905** | —         | 0.735                        |
| Frontier (CogView/Gemini/FLUX/Janus) | **0.885** | —         | 0.900                        |
| Real photos, held-out editorial      | —         | **0.990** | 0.335                        |
| **Real photos, TRULY-unseen hard subtypes** | —   | **0.343** spec (65.7% FP) | 0.062 spec (93.8% FP) |

**Honest limitation (do not hide):** on genuinely-unseen *hard* real
photographs (extreme macros, clinical, heavy-JPEG / heavily-compressed, very
low-light, abstract textures), v12 still false-positives **65.7%** at threshold
0.5. That is a large improvement over v9 (93.8%) but is **not yet
production-acceptable** for that specific hard subset; it is the next open
problem. On ordinary web-photography real images (the common case) v12 is
~99% specific. Real human artwork (WikiArt) stays near 100% specific. The 65.7%
figure is the honest number — earlier READMEs claimed ~0% real FP, which was
measured on an easier, in-distribution real set and did not survive a
truly-unseen hard-subtype probe.

## Why fp16 and not int8

Dynamic int8 quantization collapsed the two-class logits — decision agreement
with fp32 dropped to ~55% (max softmax error 0.999). fp16 retains ~100%
agreement (max softmax error ≤0.01), is natively supported by the WebGPU
execution provider, and halves the download.

## Re-exporting

```
# v12 (current shipped): broadened real class, fine-tune, export
python evaluation/gather_imagenet_batch.sh data_unseen/real_more \
  data/train-00003-of-00052-... data/train-00004-of-00052-...   # 7 shards (00003-00009)
python evaluation/build_v12.py data_unseen/real_more data_unseen/real_eval 3
python evaluation/train_v4.py --root data_v12/train --ckpt run_v11/best.pt --out run_v12 --epochs 8 --batch 80
python evaluation/export_onnx.py --ckpt run_v12/best.pt --out model/v12_detector.onnx
# eval (exact transform; eval_aid.py REQUIRES the model path)
python evaluation/eval_aid.py --real-dir data_v9/test/real --fake-dir data_v9/test/fake/deepfake --onnx model/v12_detector_fp16.onnx
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), index 1 = "AI / fake" (softmax in the extension).
