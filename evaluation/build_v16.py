"""Build v16 layout: join FRESH, content-hash-deduped ImageNet real photos
(shards 00026-00028, never used) to the real class, on top of the v14 corpus.
The dedup removes any exact-duplicate of the truly-unseen oracle
(data_unseen/real_eval, 1500 never-trained) so the oracle stays clean.

v15 fix: build into a freshly created OUT dir so a stale/partial prior build
never collides (the v15 pipeline crashed on FileExistsError and then trained on
the broken tree — that regression is why v15 was discarded). We also hard-fail
if the OUT dir already contains symlinks from a prior build instead of silently
training on leftovers.

Layout:
  data_v16/train/fake        <- reuse v14 AI
  data_v16/train/real        <- v14 real + fresh deduped ImageNet photos (REP)
  data_v16/test/fake/*       <- per-generator held-out (reuse)
  data_v16/test/real         <- training-family held-out (reuse)
  data_v16/test/real_unseen  <- the SAME 1500 oracle, NEVER trained
"""
import os, sys, glob, hashlib, shutil

FRESH = sys.argv[1]       # data_v16_more (sh* subdirs of fresh ImageNet)
REAL_UNSEEN = sys.argv[2] # data_unseen/real_eval  (1500, held out, untouched)
REP = int(sys.argv[3])
OUT = "data_v16"

def link_tree(src, dst, prefix=""):
    os.makedirs(dst, exist_ok=True); n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, prefix + f)); n += 1
    return n

# --- v15 fix: start from a clean OUT; refuse a pre-existing non-empty one ---
if os.path.exists(OUT) and sum(len(fs) for _,_,fs in os.walk(OUT)) > 0:
    raise SystemExit(f"REFUSE: {OUT} already has content (stale build). Remove it and re-run.")
os.makedirs(OUT, exist_ok=True)

# carry over v14 train + test
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v14/{sub}"
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
    shard_key = os.path.basename(os.path.dirname(f))
    os.symlink(os.path.abspath(f), os.path.join(f"{OUT}/train/real", f"v16_{shard_key}_{os.path.basename(f)}"))
    n_add += 1
    if n_add >= REP:
        break
print(f"+{n_add} fresh real added; {n_dup} dup-oracle excluded", flush=True)

# fresh unseen eval slice (oracle, untouched)
os.makedirs(f"{OUT}/test/real_unseen", exist_ok=True)
n_eval = 0
for f in sorted(glob.glob(os.path.join(REAL_UNSEEN, "*.jpg"))):
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_unseen/unseen_{os.path.basename(f)}"); n_eval += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real", "test/real_unseen"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}", flush=True)
print("DONE", flush=True)
