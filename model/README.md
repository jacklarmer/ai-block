# Model artifacts

`detector.onnx` is the **shipped** model — **v6**, an EfficientNet-B0 fine-tuned
on a maximum-breadth corpus spanning **7 diverse generators** (Ideogram, Aura
(Stability), Imagine (Meta), Leonardo/StableCog, Midjourney (JourneyDB broad
set), **DALL·E 3** (dominant Google "AI-generated" source)) **plus a broad
real-photography class** (diverse COCO editorial photos) that cuts real-photo
false-positives ~8×. Exported to ONNX in **fp16** (~8 MB).

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
# v6 (current shipped): real-photo class, fine-tune, export
python evaluation/gather_coco.py <coco_real_dir> 25000
python evaluation/build_v6.py <coco_real_dir> 200 3
python evaluation/train_v4.py --root data_v6/train --ckpt run_v5/best.pt --out run_v6 --epochs 8
python evaluation/export_onnx.py --ckpt run_v6/best.pt --out model/detector.onnx
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), where index 1 = "AI / fake" (softmax in the extension).
