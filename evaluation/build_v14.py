"""Build v14 layout: join FRESH, content-hash-deduped ImageNet real photos to
the real class. The dedup removes any exact-duplicate of the truly-unseen
oracle (data_unseen/real_eval, 1500 never-trained) so the oracle stays clean.

Layout:
  data_v14/train/fake        <- reuse v13 AI
  data_v14/train/real        <- v13 real + fresh deduped ImageNet photos (REP)
  data_v14/test/fake/*       <- per-generator held-out (reuse)
  data_v14/test/real         <- training-family held-out (reuse)
  data_v14/test/real_unseen  <- the SAME 1500 oracle, NEVER trained
"""
import os, sys, glob, hashlib

FRESH = sys.argv[1]       # data_v14_more (sh* subdirs of fresh ImageNet)
REAL_UNSEEN = sys.argv[2] # data_unseen/real_eval  (1500, held out, untouched)
REP = int(sys.argv[3])
OUT = "data_v14"

def link_tree(src, dst, prefix=""):
    os.makedirs(dst, exist_ok=True); n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, prefix + f)); n += 1
    return n

# carry over v13 train + test
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v13/{sub}"
    if os.path.isdir(sd):
        n = link_tree(sd, f"{OUT}/{sub}")
        print(f"carried {sub}: {n}", flush=True)

# Build a content-hash set of the oracle (truly-unseen eval, NEVER trained).
def hash_set(dirpath):
    s = set()
    for f in glob.glob(os.path.join(dirpath, "*.jpg")):
        with open(f, "rb") as fh:
            s.add(hashlib.sha256(fh.read()).hexdigest())
    return s

print("hashing oracle...", flush=True)
oracle_hashes = hash_set(REAL_UNSEEN)
print(f"oracle hashes: {len(oracle_hashes)}", flush=True)

# copy fresh real, deduped against oracle
os.makedirs(f"{OUT}/train/real", exist_ok=True)
n_add = 0; n_dup = 0
fres = glob.glob(os.path.join(FRESH, "**", "*.jpg"), recursive=True)
for f in sorted(fres):
    with open(f, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    if h in oracle_hashes:
        n_dup += 1
        continue
    shard_key = os.path.basename(os.path.dirname(f))  # e.g. sh00020
    os.symlink(os.path.abspath(f), os.path.join(f"{OUT}/train/real", f"v14_{shard_key}_{os.path.basename(f)}"))
    n_add += 1
    if n_add >= REP:
        break
print(f"+{n_add} fresh real added; {n_dup} dup-oracle excluded", flush=True)

# fresh unseen eval slice (oracle, untouched) — identical to v13
os.makedirs(f"{OUT}/test/real_unseen", exist_ok=True)
n_eval = 0
for f in sorted(glob.glob(os.path.join(REAL_UNSEEN, "*.jpg"))):
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_unseen/unseen_{os.path.basename(f)}"); n_eval += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real", "test/real_unseen"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}", flush=True)
print("DONE", flush=True)
