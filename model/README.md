# Model artifacts

`detector.onnx` is the **shipped** model — EfficientNet-B0 fine-tuned on a
multi-generator real-vs-AI corpus and exported to ONNX in **fp16** (~8 MB).

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
python evaluation/export_onnx.py --ckpt run_v3/best.pt --out model/detector.onnx
```

Input contract: a raw RGB image, center-cropped to 256×256, normalized by
ImageNet mean/std, laid out `[1,3,256,256]` fp32. Output is 2-class logits
(`[1,2]`), where index 1 = "AI / fake" (softmax in the extension).
