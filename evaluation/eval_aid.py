"""
Evaluate a trained / exported ONNX model on a held-out test set.
Reports:
  - balanced accuracy
  - balanced accuracy at 65% confidence threshold (accept >=0.65 or <=0.35)
  - precision/recall per class
Uses onnxruntime (CPU) for cross-checking the exported artifact matches the
PyTorch model, and supports both .pt (torch) and .onnx artifacts.
"""
import os, sys, glob, argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_predictor(kind, path):
    if kind == "pt":
        import torch, timm
        dev = "cpu"
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(path, map_location=dev))
        model.eval()
        def predict(img):
            x = val_tf(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                p = torch.softmax(logits, 1)[0].numpy()
            return p  # [real, fake]
        return predict
    else:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.get_outputs()[0].name
        def predict(img):
            x = val_tf(img).unsqueeze(0).numpy()
            logits = sess.run([out], {inp: x})[0][0]
            e = np.exp(logits - logits.max())
            return e / e.sum()
        return predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir")
    ap.add_argument("--fake-dir")
    ap.add_argument("--pt")
    ap.add_argument("--onnx")
    ap.add_argument("--max-per-class", type=int, default=3000)
    a = ap.parse_args()
    predictor = load_predictor("pt" if a.pt else "onnx", a.pt or a.onnx)

    def collect(d):
        if not d: return []
        return sorted(glob.glob(os.path.join(d, "*.*")))[:a.max_per_class]

    real_files = collect(a.real_dir)
    fake_files = collect(a.fake_dir)
    print(f"real={len(real_files)} fake={len(fake_files)}")

    yt, yp, conf = [], [], []
    for files, label in [(real_files, 0), (fake_files, 1)]:
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            p = predictor(img)
            yt.append(label); yp.append(int(p[1] > 0.5)); conf.append(p[1])
    yt=np.array(yt); yp=np.array(yp); conf=np.array(conf)
    bacc = balanced_accuracy_score(yt, yp)
    acc = accuracy_score(yt, yp)
    # at 65% threshold: only consider confident predictions; a conf>=0.65 calls AI,
    # conf<=0.35 calls real; middle ignored.
    mask = np.abs(conf - 0.5) >= 0.15
    if mask.sum() == 0:
        bacc65 = float("nan"); cov = 0
    else:
        y = yt[mask]; c = conf[mask]
        pred = (c >= 0.65).astype(int)
        bacc65 = balanced_accuracy_score(y, pred)
        cov = mask.mean()
    print(f"balanced acc = {bacc:.4f}  acc = {acc:.4f}")
    print(f"balanced acc @65% conf = {bacc65:.4f}  (coverage {cov:.3f}, n={mask.sum()})")
    cm = confusion_matrix(yt, yp)
    print("confusion (rows=real,fake; cols=pred real,fake):\n", cm)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average=None, labels=[0,1])
    print(f"real: prec {p[0]:.3f} rec {r[0]:.3f} | fake: prec {p[1]:.3f} rec {r[1]:.3f}")

if __name__ == "__main__":
    main()
