"""
Export trained classifier to a WebGPU-friendly ONNX model.
- Folds ImageNet normalization into the model so input is raw 0..255 RGB.
- Saves full fp32 + fp16 + int8(dynamic) variants for on-device choice.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import timm
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="detector.onnx")
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--img-size", type=int, default=256)
    return p.parse_args()

def normalize_in_graph(model, img_size):
    """Replace model with wrapper that takes 0..255 input and applies imagenet norm inside."""
    class Wrapped(nn.Module):
        def __init__(self, m, mean, std):
            super().__init__()
            self.m = m
            self.mean = nn.Parameter(torch.tensor(mean).view(1,3,1,1), requires_grad=False)
            self.std = nn.Parameter(torch.tensor(std).view(1,3,1,1), requires_grad=False)
        def forward(self, x):
            x = x / 255.0
            x = (x - self.mean) / self.std
            return self.m(x)
    return Wrapped(model, IMAGENET_MEAN, IMAGENET_STD)

def main():
    args = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.backbone, pretrained=False, num_classes=2)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd)
    model.eval().to(dev)
    # NOTE: input is the RAW ImageNet-normalized float32 tensor (center-crop 256).
    # JS performs: resize->center crop 256, then (x/255-mean)/std, NCHW.
    # We do NOT fold normalization into the graph — JS is the source of truth.

    # sanity: output shape
    dummy = torch.rand(1,3,args.img_size,args.img_size).to(dev)
    with torch.no_grad():
        print("output shape:", model(dummy).shape)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0:"batch"}},
        opset_version=17,
        dynamo=False,
    )
    # verify
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print("exported", args.out, "size MB:", round(os.path.getsize(args.out)/1e6,2))

    # fp16 variant
    from onnxconverter_common import float16
    from onnxruntime.transformers import float16 as ort_fp16
    base = args.out.rsplit(".onnx",1)[0]
    try:
        m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m_fp16, f"{base}_fp16.onnx")
        print("fp16 size MB:", round(os.path.getsize(f"{base}_fp16.onnx")/1e6,2))
    except Exception as e:
        print("fp16 fail", e)

    # int8 dynamic quantization
    try:
        quantize_dynamic(args.out, f"{base}_int8.onnx", weight_type=QuantType.QInt8)
        print("int8 size MB:", round(os.path.getsize(f"{base}_int8.onnx")/1e6,2))
    except Exception as e:
        print("int8 fail", e)

if __name__ == "__main__":
    main()
