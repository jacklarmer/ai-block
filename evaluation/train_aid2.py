"""
Train real-vs-AI classifier v2 with generalization-focused augmentation.
Key ideas for cross-generator generalization (AIGC detection literature):
  - Random JPEG-compression degradation (web images are compressed; forces model to
    learn artifacts robust to compression rather than relying on noise texture)
  - High-frequency emphasis (mix image with Laplacian/Sobel residual) so the model
    learns the high-frequency signature all generators share
  - Aggressive geometric + color augmentation
  - Label smoothing + strong regularization
Exports best checkpoint, then we convert to ONNX separately.
"""
import os, sys, argparse, time, glob, random, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import timm
from sklearn.metrics import balanced_accuracy_score
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="run2")
    p.add_argument("--val-split", type=float, default=0.08)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--label-smooth", type=float, default=0.05)
    return p.parse_args()

CLASS2IDX = {"real": 0, "fake": 1}

def build_data(root):
    images, labels = [], []
    for clazz in ["real", "fake"]:
        d = os.path.join(root, clazz)
        files = glob.glob(os.path.join(d, "*.*"))
        random.shuffle(files)
        print(f"{clazz}: {len(files)}", flush=True)
        for f in files:
            images.append(f); labels.append(CLASS2IDX[clazz])
    return images, labels

# ---- robust train transform ----
class RandJPEG(object):
    """Random JPEG re-compression; forces robustness to web compression."""
    def __init__(self, qmin=30, qmax=95, p=0.8):
        self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img):
        if random.random() < self.p and img.mode == "RGB":
            q = random.randint(self.qmin, self.qmax)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q)
            return Image.open(buf).convert("RGB")
        return img

class HighFreqMix(object):
    """Mix (with prob) a random fraction of the high-frequency residual into the image."""
    def __init__(self, p=0.5, alpha_range=(0.0, 0.6)):
        self.p, self.alpha_range = p, alpha_range
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        # Laplacian high-pass
        lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=3)
        alpha = random.uniform(*self.alpha_range)
        # reduced color for residual, add back
        out = arr + alpha * np.sign(lap) * np.abs(lap)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.4, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    RandJPEG(),
    HighFreqMix(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ImgDS(Dataset):
    def __init__(self, files, labels, tf): self.files, self.labels, self.tf = files, labels, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img), self.labels[i]

def main():
    args = parse()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    images, labels = build_data(args.root)
    idx = list(range(len(images))); rng = random.Random(0); rng.shuffle(idx)
    n_val = int(len(idx)*args.val_split)
    val_idx = set(idx[:n_val])
    tr_files=[images[i] for i in idx if i not in val_idx]; tr_labels=[labels[i] for i in idx if i not in val_idx]
    va_files=[images[i] for i in idx if i in val_idx];   va_labels=[labels[i] for i in idx if i in val_idx]

    model = timm.create_model(args.backbone, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # label smoothing CE
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    tr_ds = ImgDS(tr_files, tr_labels, train_tf); va_ds = ImgDS(va_files, va_labels, val_tf)
    from collections import Counter
    cnt = Counter(tr_labels)
    w = torch.tensor([1.0/cnt[l] for l in tr_labels])
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr_labels), replacement=True)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"train {len(tr_files)} val {len(va_files)}", flush=True)

    best = -1
    for ep in range(args.epochs):
        model.train(); t0=time.time(); run=0; correct=0; tot=0
        for x,y in tr_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item()*x.size(0); tot += x.size(0)
            correct += (model(x).argmax(1)==y).sum().item() if False else 0
        sched.step()
        print(f"ep{ep} loss={run/max(tot,1):.4f} {time.time()-t0:.0f}s", flush=True)
        yt, conf = [], []
        model.eval()
        with torch.no_grad():
            for x,y in va_dl:
                x = x.to(device)
                p = F.softmax(model(x),1)[:,1].cpu().numpy()
                yt += y.tolist(); conf += p.tolist()
        yt_a=np.array(yt); conf_a=np.array(conf)
        bacc = balanced_accuracy_score(yt_a, (conf_a>0.5).astype(int))
        mask = np.abs(conf_a-0.5) >= 0.15
        bacc65 = balanced_accuracy_score(yt_a[mask], (conf_a[mask]>0.65).astype(int)) if mask.sum()>0 else 0
        print(f"  VAL bacc={bacc:.4f} acc65={bacc65:.4f} n={mask.sum()}", flush=True)
        if bacc65 > best:
            best = bacc65
            torch.save(model.state_dict(), os.path.join(args.out,"best.pt"))
            print("  saved best", flush=True)
    print("DONE best bacc65", best, flush=True)

if __name__ == "__main__":
    main()
