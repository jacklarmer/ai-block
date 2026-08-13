"""
Stream-gather a diverse real + multi-generator AI corpus from BitMind HF repos.
Saves images to:
  data/bitmind_real/<n>.jpg
  data/bitmind_ai/<gen>_<n>.jpg
"""
import os, sys, io, random
from datasets import load_dataset
from PIL import Image

OUT = "/home/jack/aidet/data"
os.makedirs(f"{OUT}/bitmind_real", exist_ok=True)
os.makedirs(f"{OUT}/bitmind_ai", exist_ok=True)

PLAN = [
    # (repo, split, per-generator real/ai, real_or_ai, tag)
    ("bitmind/bm-subnet-real",              "train", 25000, "real", "bmreal"),
    ("bitmind/bm-subnet-sdxl-256",          "train", 20000, "ai",   "sdxl"),
    ("bitmind/GenImage_ADM",                "train",  8000, "ai",   "adm"),
    ("bitmind/GenImage_BigGAN",             "train",  8000, "ai",   "biggan"),
    ("bitmind/GenImage_glide",              "train",  8000, "ai",   "glide"),
    ("bitmind/GenImage_wukong",             "train",  8000, "ai",   "wukong"),
    ("bitmind/GenImage_VQDM",               "train",  8000, "ai",   "vqdm"),
    ("bitmind/GenImage_MidJourney",         "train",  8000, "ai",   "midjourney"),
    ("bitmind/bm-subnet-FLUX.1-dev",        "train",  6000, "ai",   "flux"),
]

def save(img, path):
    img = img.convert("RGB")
    # center-crop square to reduce aspect-ratio variance, save as jpg
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    img.save(path, quality=92)

for repo, split, n, kind, tag in PLAN:
    print(f"=== {tag} {kind} from {repo} (target {n}) ===", flush=True)
    try:
        # auto-detect split: use first available if given split missing
        from datasets import get_dataset_split_names
        avail = get_dataset_split_names(repo)
        if split not in avail:
            split = avail[0]
            print(f"  {tag}: using split {split}", flush=True)
        ds = load_dataset(repo, split=split, streaming=True)
        it = iter(ds)
        count = 0
        while count < n:
            try:
                row = next(it)
            except StopIteration:
                print(f"  {tag}: exhausted at {count}", flush=True)
                break
            except Exception as e:
                print(f"  {tag}: row err, skip ({e})", flush=True)
                continue
            try:
                img = row["image"]
                sub = f"{OUT}/bitmind_real" if kind == "real" else f"{OUT}/bitmind_ai"
                p = os.path.join(sub, f"{tag}_{count}.jpg")
                save(img, p)
                count += 1
                if count % 2000 == 0:
                    print(f"  {tag}: {count}", flush=True)
            except Exception as e:
                print("  img err", e, flush=True)
                continue
        print(f"  {tag}: DONE {count}", flush=True)
    except Exception as e:
        print(f"  {tag}: FAILED {e}", flush=True)
print("ALL BITMIND GATHER DONE", flush=True)
