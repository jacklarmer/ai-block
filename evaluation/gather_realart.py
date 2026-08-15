"""Gather real human-made ART/illustration (NOT AI) for the v7 fine-tune.

v6 false-flags real artwork (paintings, historical illustrations, diagrams)
at ~34-42% because the model learned "clean/artistic/high-detail image" ≈ AI.
WikiArt is real human art (paintings, sketches, prints) — the exact class that
should NOT be flagged. Adding it to the "real" class teaches the model that a
polished artistic image is not necessarily AI-generated.

Usage: python gather_realart.py <outdir> <max_n>
"""
import sys, os, time
from PIL import Image
from datasets import load_dataset

OUT, MAXN = sys.argv[1], int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
start = len(os.listdir(OUT))

ds = load_dataset("Artificio/WikiArt", split="train", streaming=True)
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
            im.save(os.path.join(OUT, f"art_{start+n:06d}.jpg"), "JPEG", quality=92)
            n += 1
            last_save = time.time()
            if n % 2000 == 0:
                print(f"  {n}", flush=True)
        except Exception as e:
            print("img err", str(e)[:60], flush=True); continue
        if time.time() - last_save > 200:
            print("STALL 200s break", flush=True); break
finally:
    print(f"GATHER REALART DONE: +{n}", flush=True)
