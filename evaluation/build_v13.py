"""Build v13 layout: FRESH diverse real photos (ImageNet shards 00000-00002,
00048-00051 — never used in v10/v11/v12) joined to the real class, to drive the
truly-unseen hard-subtype real false-positive down further from v12's 65.7%.

Layout:
  data_v13/train/fake        <- all prior AI (reuse v12)
  data_v13/train/real        <- v12 real + fresh diverse ImageNet photos (REP)
  data_v13/test/fake/*       <- per-generator held-out (reuse v9/v12)
  data_v13/test/real         <- training-family real held-out (reuse)
  data_v13/test/real_unseen  <- the SAME 1500 ImageNet-val oracle (never trained)
"""
import os, sys, glob

REAL_MORE = sys.argv[1]    # data_v13_more (sh* subdirs of fresh ImageNet)
REAL_UNSEEN = sys.argv[2]  # data_unseen/real_eval (1500, held out, untouched)
REP = int(sys.argv[3])
OUT = "data_v13"

def link_tree(src, dst, prefix=""):
    os.makedirs(dst, exist_ok=True); n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, prefix + f)); n += 1
    return n

# carry over v12 train + test
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v12/{sub}"
    if os.path.isdir(sd):
        link_tree(sd, f"{OUT}/{sub}")

# add fresh diverse real photos into train/real, repeated
os.makedirs(f"{OUT}/train/real", exist_ok=True)
rm = sorted(glob.glob(f"{REAL_MORE}/*/*.jpg"))
n_add = 0
for rep in range(REP):
    for i, f in enumerate(rm):
        os.symlink(os.path.abspath(f), f"{OUT}/train/real/fresh{rep}_{i:06d}.jpg"); n_add += 1

# fresh unseen eval slice (ImageNet val, NOT trained) — kept identical
os.makedirs(f"{OUT}/test/real_unseen", exist_ok=True)
n_eval = 0
for f in sorted(glob.glob(f"{REAL_UNSEEN}/*.jpg")):
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_unseen/unseen_{os.path.basename(f)}"); n_eval += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real", "test/real_unseen"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}")
print(f"+{n_add} fresh real added to train/real across {len(rm)} unique")
print(f"unseen val eval slice: {n_eval}")
print("DONE")
