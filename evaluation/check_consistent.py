"""
Consistency check: does the exported ONNX model produce the same softmax
probabilities as the PyTorch model on the same images? This validates the
deployment artifact that will run in the browser via WebGPU.
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
val_tf = transforms.Compose([
    transforms.Resize(288), transforms.CenterCrop(256),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image-dir", required=True, help="dir of test images")
    a = ap.parse_args()

    import torch, timm
    dev = "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(a.pt, map_location=dev))
    model.eval()

    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name; out = sess.get_outputs()[0].name

    files = sorted(glob.glob(os.path.join(a.image_dir, "*.*")))[:200]
    maxdiff = 0; tot = 0; agree = 0
    for f in files:
        try: img = Image.open(f).convert("RGB")
        except Exception: continue
        x = val_tf(img).unsqueeze(0)
        with torch.no_grad():
            pt_p = torch.softmax(model(x),1)[0].numpy()
        on_p = sess.run([out], {inp: x.numpy()})[0][0]
        on_p = np.exp(on_p - on_p.max()); on_p /= on_p.sum()
        d = float(np.abs(pt_p - on_p).max())
        maxdiff = max(maxdiff, d); tot += 1
        if int(pt_p[1]>0.5) == int(on_p[1]>0.5): agree += 1
    print(f"images={tot} agree={agree} ({agree/max(tot,1):.2%}) max_softmax_diff={maxdiff:.6f}")

if __name__ == "__main__":
    main()
