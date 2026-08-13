"""
Assemble unified train/val dataset:
  data/all/real/  (real photos: hemg real + bitmind_real)
  data/all/fake/  (AI images: hemg fake + bitmind_ai across generators)
For de-duplication across sources, content-hash based. Then stratified 90/10 split
into data/train/{real,fake} and data/val/{real,fake}.
"""
import os, sys, hashlib, random, shutil, glob

SRC = "/home/jack/aidet/data"
ALL = "/home/jack/aidet/data_all2"
TRAIN = "/home/jack/aidet/data_all2/train"
VAL = "/home/jack/aidet/data_all2/val"

random.seed(42)

def hash_file(p, chunk=65536):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def collect(src_dirs, out_class):
    """Copy images from src_dirs into ALL/out_class, dedup by content hash."""
    out = os.path.join(ALL, out_class)
    os.makedirs(out, exist_ok=True)
    seen = set()
    n = 0
    for d in src_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            try:
                h = hash_file(p)
            except Exception:
                continue
            if h in seen:
                continue
            seen.add(h)
            dst = os.path.join(out, f"{out_class}_{len(seen)}.{p.split('.')[-1]}")
            shutil.copy(p, dst)
            n += 1
    return n

# Real sources
real = collect([f"{SRC}/hemg_img/real", f"{SRC}/bitmind_real"], "real")
# Fake sources: all AI dirs
fake_dirs = [f"{SRC}/hemg_img/fake", f"{SRC}/bitmind_ai"]
fake = collect(fake_dirs, "fake")
print(f"REAL total: {real}", flush=True)
print(f"FAKE total: {fake}", flush=True)

# Stratified 90/10 split
for cls in ["real", "fake"]:
    files = glob.glob(os.path.join(ALL, cls, "*"))
    random.shuffle(files)
    n_val = int(len(files) * 0.1)
    val_files, train_files = files[:n_val], files[n_val:]
    for out_root, fl in [(TRAIN, train_files), (VAL, val_files)]:
        d = os.path.join(out_root, cls)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(fl):
            shutil.copy(f, os.path.join(d, f"{cls}_{i}.jpg"))
    print(f"{cls}: train={len(train_files)} val={len(val_files)}", flush=True)

print("DONE", flush=True)
