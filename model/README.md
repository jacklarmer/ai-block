# Model artifacts

`detector.onnx` is the **shipped** model — **v8**, an EfficientNet-B0 fine-tuned
on a maximum-breadth, low-false-positive corpus:
- **frontier AI generators** (CogView4-6B, Gemini 2.5 flash-image, FLUX.1-dev,
  Janus-Pro-7B, RealVisXL) from bitmind benchmark snapshots — the newer,
  most-photographic outputs; frontier recall **55% → 91%**,
- **7+ established AI generators** (Ideogram, Aura, Imagine, Leonardo,
  Midjourney, DALL·E 3) — 96–99% recall,
- **real-photography class** (COCO editorial) — real photo FP **~0%**,
- **real-human-art class** (12k WikiArt) — real art FP **47.5% → ~2%**.

Exported to ONNX in **fp16** (~8 MB).

## Files

| file                 | size  | role                                   |
|----------------------|-------|----------------------------------------|
| `detector.onnx`      | ~8 MB | **shipped** fp16 (WebGPU / WASM)       |
| `detector_fp32.onnx` | ~16 MB| reference, maximum fidelity             |
| `detector_fp16.onnx` | ~8 MB | fp16 (identical to shipped)             |
| `detector_int8.onnx` | ~4 MB | dynamic int8 — **rejected** (see below) |

## Why fp16 and not int8

Dynamic int8 quantization collapsed the two-class logits — decision agreement
with fp32 dropped to ~55% (max softmax error 0.999). fp16 retains 99.5%
agreement (max softmax error 0.03), is natively supported by the WebGPU
execution provider, and halves the download.

## Re-exporting

```
# v8 (current shipped): frontier-AI class, fine-tune, export
python evaluation/gather_frontier.py <frontier_dir> 9000
python evaluation/build_v8.py <frontier_dir> 200 4
python evaluation/train_v4.py --root data_v8/train --ckpt run_v7/best.pt --out run_v8 --epochs 8
python evaluation/export_onnx.py --ckpt run_v8/best.pt --out model/detector.onnx
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), where index 1 = "AI / fake" (softmax in the extension).
