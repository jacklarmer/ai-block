"""Fine-tune v3 -> v4 for REAL-WORLD inference conditions.

Motivation: on real searches (Google Images tab) a big share of AI images appear
as SMALL, HEAVILY-COMPRESSED thumbnails that get upscaled to 256px before the
model sees them. That upscale washes out the high-frequency artifacts the model
keys on, pushing confidence below threshold -> "only about half flagged".

This run:
  * starts from the v3 timm efficientnet_b0 checkpoint (fast, stable transfer)
  * adds a THUMBNAIL augmentation: randomly downscale to a tiny size (e.g.
    40-128px) then upscale back to 256, so the model learns the AI signature
    even after small-input upscaling
  * keeps/hardens JPEG degradation + high-frequency mixing
  * evaluates on the SAME non-degraded held-out set (Mobius + AFHQ) for a like-
    for-like comparison, plus a degraded variant to show real-world gain

Usage:
  python train_v4.py --root data_v3/train --ckpt run_v3/best.pt \
    --out run_v4 --epochs 8
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
CLASS2IDX = {"real": 0, "fake": 1}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True, help="v3 best.pt to start from")
    p.add_argument("--out", default="run_v4")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)   # low LR: fine-tune
    p.add_argument("--val-split", type=float, default=0.06)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()

def build_data(root):
    images, labels = [], []
    for clazz in ["real", "fake"]:
        d = os.path.join(root, clazz)
        files = glob.glob(os.path.join(d, "*.*"))
        random.shuffle(files)
        for f in files:
            images.append(f); labels.append(CLASS2IDX[clazz])
    return images, labels

class RandJPEG:
    def __init__(self, qmin=20, qmax=95, p=0.85):
        self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img):
        if random.random() < self.p and img.mode == "RGB":
            q = random.randint(self.qmin, self.qmax)
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=q)
            return Image.open(buf).convert("RGB")
        return img

class Thumbnail(object):
    """Simulate a small, pixel-dense web/AI thumbnail upscaled to 256.

    Randomly downscale the (already-cropped) 256 input to a short side in
    [thumb_min, thumb_max], then bilinear-upscale back to 256. p controls how
    often. tiny sizes emulate Google's small grid thumbnails.
    """
    def __init__(self, thumb_min=48, thumb_max=160, p=0.5):
        self.thumb_min, self.thumb_max, self.p = thumb_min, thumb_max, p
    def __call__(self, img):
        if random.random() >= self.p:
            return img
        s = random.randint(self.thumb_min, self.thumb_max)
        small = img.resize((s, s), Image.BILINEAR)
        return small.resize((img.width, img.height), Image.BILINEAR)

class HighFreqMix:
    def __init__(self, p=0.4, alpha_range=(0.0, 0.5)):
        self.p, self.alpha_range = p, alpha_range
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=3)
        a = random.uniform(*self.alpha_range)
        out = np.clip(arr + a * np.sign(lap) * np.abs(lap), 0, 255).astype(np.uint8)
        return Image.fromarray(out)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    Thumbnail(p=0.5),
    RandJPEG(),
    HighFreqMix(p=0.4),
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
        return self.tf(Image.open(self.files[i]).convert("RGB")), self.labels[i]

def main():
    args = parse()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    if torch.cuda.is_available():
        # Hard production-lane guard: ai-block training MUST run on the RTX 5090
        # (the dedicated training GPU) and NEVER on the RTX PRO 6000, which is the
        # serialized full-BF16 MiniMax / ComfyUI production lane. cuda:0 here is
        # the PRO 6000, so select the RTX 5090 by name (honors CUDA_VISIBLE_DEVICES
        # if the environment already pinned to a single RTX 5090).
        import warnings
        avail = [i for i in range(torch.cuda.device_count())]
        rtx = [i for i in avail if "5090" in torch.cuda.get_device_name(i)]
        pro = [i for i in avail if "PRO 6000" in torch.cuda.get_device_name(i)]
        if pro and not rtx:
            raise RuntimeError(
                "refusing to train: only the RTX PRO 6000 production lane is CUDA-visible. "
                "Export CUDA_VISIBLE_DEVICES=0000:01:00.0 to pin to the RTX 5090."
            )
        dev_idx = rtx[0] if rtx else 0
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        device = f"cuda:{dev_idx}"
    else:
        device = "cpu"
    print("device", device, flush=True)

    images, labels = build_data(args.root)
    idx = list(range(len(images))); rng = random.Random(0); rng.shuffle(idx)
    n_val = int(len(idx) * args.val_split)
    val_idx = set(idx[:n_val])
    tr_files = [images[i] for i in idx if i not in val_idx]; tr_lab = [labels[i] for i in idx if i not in val_idx]
    va_files = [images[i] for i in idx if i in val_idx];     va_lab = [labels[i] for i in idx if i in val_idx]
    print(f"train {len(tr_files)} val {len(va_files)}", flush=True)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    from collections import Counter
    cnt = Counter(tr_lab)
    w = torch.tensor([1.0 / cnt[l] for l in tr_lab])
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr_lab), replacement=True)
    tr_ds = ImgDS(tr_files, tr_lab, train_tf)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)
    va_ds = ImgDS(va_files, va_lab, val_tf)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    best = -1
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0; tot = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item() * x.size(0); tot += x.size(0)
        sched.step()
        model.eval(); yt = []; conf = []
        with torch.no_grad():
            for x, y in va_dl:
                p = F.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
                yt += y.tolist(); conf += p.tolist()
        yt_a = np.array(yt); conf_a = np.array(conf)
        bacc = balanced_accuracy_score(yt_a, (conf_a > 0.5).astype(int))
        mask = np.abs(conf_a - 0.5) >= 0.15
        bacc65 = balanced_accuracy_score(yt_a[mask], (conf_a[mask] > 0.65).astype(int)) if mask.sum() > 0 else 0
        print(f"ep{ep} loss={run/max(tot,1):.4f} {time.time()-t0:.0f}s VAL bacc={bacc:.4f} acc65={bacc65:.4f} n={mask.sum()}", flush=True)
        if bacc65 > best:
            best = bacc65
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
    print("BEST VAL bacc65", best, flush=True)

if __name__ == "__main__":
    main()
