"""Mixed held-out evaluation across MULTIPLE unseen generators + real photos.
Gives a realistic 'how much better for a Google-like long tail' reading.

Usage: python eval_mixed.py <ckpt.pt> <real_dir> [fake_dir ...] [--maxreal N] [--maxfake N]
Each fake_dir is scored separately AND combined. Reports balanced-accuracy @
multiple thresholds + recall/fp so we can compare v3 vs v4 fairly.
"""
import sys, os, glob, argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import timm

IMG = 256
tf = T.Compose([T.Resize(int(IMG * 1.125)), T.CenterCrop(IMG), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load_scores(ckpt, dirs, n):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(dev).eval()
    def score_dir(d):
        out = []
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))[:n]
        for f in files:
            try:
                x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(dev)
            except Exception:
                continue
            with torch.no_grad():
                log = model(x); m = log.max(1, keepdim=True).values
                e = (log - m).exp(); p = e / e.sum(1, keepdim=True)
                out.append(p[0, 1].item())
        return np.array(out)
    return {os.path.basename(d) if d else "fake": score_dir(d) for d in dirs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", action="append", dest="fakes", required=True)
    ap.add_argument("--maxreal", type=int, default=4000)
    ap.add_argument("--maxfake", type=int, default=4000)
    a = ap.parse_args()
    alls = load_scores(a.ckpt, [a.real], a.maxreal)
    real = alls[os.path.basename(a.real)]
    fakes = load_scores(a.ckpt, a.fakes, a.maxfake)
    print(f"real n={len(real)}")
    for name, s in fakes.items():
        print(f"  fake[{name}] n={len(s)}")
    # combined
    comb = np.concatenate(list(fakes.values()))
    groups = {"real": real}
    for name, s in fakes.items():
        groups[name] = s
    groups["ALL_FAKE"] = comb
    print("\nthr   " + "  ".join(f"{k:>8}" for k in groups) + "   (AI recall per group; real col = real FP)")
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        row = [f"{t:.2f}"]
        for name, s in groups.items():
            v = (s >= t).mean()
            row.append(f"{v:>8.3f}")
        print("  ".join(row))
    # balanced acc at 0.50 and 0.65 over real vs ALL_FAKE
    for t in [0.50, 0.65]:
        recall = (comb >= t).mean(); fp = (real >= t).mean()
        print(f"bacc@{t} = {0.5*(recall + (1-fp)):.4f} (recall {recall:.3f}, realFP {fp:.3f})")

if __name__ == "__main__":
    main()
