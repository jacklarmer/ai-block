"""Gather COCO real photographs for the v6 fine-tune.

COCO = diverse real everyday scenes (people, animals, vehicles, outdoor,
objects, action) — the kind of genuine editorial photography that Wikipedia and
similar sites use. This teaches the model what REAL PHOTOGRAPHS look like so it
stops false-flagging real editorial images as AI (the model's current "real"
class is art/faces-heavy, which is why Wikipedia photos get misflagged).

Usage: python gather_coco.py <outdir> <max_n>
"""
import sys, os, time
from PIL import Image
from datasets import load_dataset

OUT, MAXN = sys.argv[1], int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
start = len(os.listdir(OUT))  # resume-friendly

ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
n = 0
last_save = time.time()
it = iter(ds)
try:
    while n < MAXN:
        try:
            row = next(it)
        except StopIteration:
            print("exhausted", flush=True); break
        except Exception as e:
            print("row err", str(e)[:60], flush=True); continue
        img = row.get("image")
        if img is None:
            continue
        try:
            im = img.convert("RGB")
            im.thumbnail((256, 256), Image.LANCZOS)
            im.save(os.path.join(OUT, f"coco_{start+n:06d}.jpg"), "JPEG", quality=92)
            n += 1
            last_save = time.time()
            if n % 2000 == 0:
                print(f"  {n}", flush=True)
        except Exception as e:
            print("img err", str(e)[:60], flush=True); continue
        if time.time() - last_save > 240:
            print("STALL 240s break", flush=True); break
finally:
    print(f"GATHER COCO DONE: +{n}", flush=True)
