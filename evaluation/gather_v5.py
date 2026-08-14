"""Gather a DIVERSE multi-generator AI corpus for the v5 fine-tune.

Adds generator types the v4 model hasn't seen (or barely): Ideogram, Aura,
Imagine (Meta), Leonardo/StableCog, JourneyDB (Midjourney), DeepfakeDataset,
synthetic humans, plus MORE DALL-E 3. Each lands in its own dir under out/ so we
can (a) hold out per-generator slices and (b) rebalance later.

Usage: python gather_v5.py <out> <per_gen>
"""
import sys, os, io, time, traceback
from PIL import Image
from datasets import load_dataset

OUT = sys.argv[1]
PER = int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)

PLAN = [
    ("bitmind/ideogram-27k",              "ideogram"),
    ("bitmind/bm-aura-imagegen",          "aura"),
    ("bitmind/bm-imagine",                "imagine"),
    ("bitmind/Deepfake-leonardo-stablecog","leonardo"),
    ("bitmind/JourneyDB",                 "midjourney"),
    ("bitmind/DeepfakeDataset",           "deepfake"),
    ("bitmind/syntheticHuman_v10",        "synthfaces"),
    ("bitmind/Dalle-3-1M",                "dalle3"),
]

def get_img(row):
    for k in ("image", "png", "jpg"):
        v = row.get(k)
        if v is not None:
            return v
    return None

for repo, tag in PLAN:
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    start = len(os.listdir(outdir))
    want = PER if start == 0 else 0  # only fill if empty (resume-friendly)
    print(f"=== {tag} from {repo} (have {start}) ===", flush=True)
    if want <= 0:
        print(f"   skipped (already has {start})", flush=True)
        continue
    # column name may vary per repo; respect the auto-detected split
    from datasets import get_dataset_split_names
    try:
        avail = get_dataset_split_names(repo)
        split = "train" if "train" in avail else avail[0]
        ds = load_dataset(repo, split=split, streaming=True)
    except Exception as e:
        print(f"   FAILED load {e}", flush=True); continue
    n = 0
    last_save = time.time()
    it = iter(ds)
    while n < want:
        try:
            row = next(it)
        except StopIteration:
            print(f"   exhausted at {n}", flush=True); break
        except Exception as e:
            print(f"   row err {e}", flush=True); continue
        img = get_img(row)
        if img is None:
            continue
        try:
            im = img.convert("RGB")
            im.thumbnail((256, 256), Image.LANCZOS)
            im.save(os.path.join(outdir, f"{tag}_{start+n:06d}.jpg"), "JPEG", quality=90)
            n += 1
            last_save = time.time()
            if n % 500 == 0:
                print(f"   {n}", flush=True)
        except Exception as e:
            print(f"   img err {e}", flush=True); continue
        if time.time() - last_save > 240:
            print("   STALL 240s, breaking", flush=True); break
    print(f"   DONE {tag}: +{n}", flush=True)

print("ALL GATHER V5 DONE", flush=True)
