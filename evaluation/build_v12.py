"""Build v12 layout: MASSLY broaden the real class with diverse real-world
photography (fixes the severe unseen-real false-positive rate).

v11 reduced unseen-val FP from 87% -> 79% but still far too high. Root cause:
the real class is too narrow (COCO editorial + WikiArt + 3k ImageNet). v12
adds ~17.5K diverse ImageNet real photographs (7 train shards, distinct
classes: macros, textures, clinical, low-light, objects, animals) directly
into the real class so "real photography" is anchored across the full
spectrum, including AI-lookalike subtypes.

The 1,500 ImageNet-val set stays as rigorously-unseen eval (never trained).

Layout:
  data_v12/train/fake        <- all prior AI (reuse v11)
  data_v12/train/real        <- v11 real + ~17.5K more ImageNet real photos
  data_v12/test/fake/*       <- per-generator held-out (reuse v11)
  data_v12/test/real         <- training-family real held-out (reuse)
  data_v12/test/real_unseen  <- 1,500 ImageNet-val (rigorously unseen)
"""
import os, sys, glob

REAL_MORE = sys.argv[1]    # data_unseen/real_more (sh* subdirs)
REAL_UNSEEN = sys.argv[2]  # data_unseen/real_eval (1,500, held out)
REP = int(sys.argv[3])
OUT = "data_v12"

def link_tree(src, dst, prefix=""):
    os.makedirs(dst, exist_ok=True); n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, prefix + f)); n += 1
    return n

# carry over v11 train + test
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v11/{sub}"
    if os.path.isdir(sd):
        link_tree(sd, f"{OUT}/{sub}")

# add diverse real photos (per-shard subdirs) into train/real, repeated
os.makedirs(f"{OUT}/train/real", exist_ok=True)
rm = sorted(glob.glob(f"{REAL_MORE}/*/*.jpg"))
n_add = 0
for rep in range(REP):
    for i, f in enumerate(rm):
        os.symlink(os.path.abspath(f), f"{OUT}/train/real/more{rep}_{i:06d}.jpg"); n_add += 1

# fresh unseen eval slice (ImageNet val, NOT trained)
os.makedirs(f"{OUT}/test/real_unseen", exist_ok=True)
n_eval = 0
for f in sorted(glob.glob(f"{REAL_UNSEEN}/*.jpg")):
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_unseen/unseen_{os.path.basename(f)}"); n_eval += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real", "test/real_unseen"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}")
print(f"+{n_add} diverse real added to train/real across {len(rm)} unique")
print(f"unseen val eval slice: {n_eval}")
print("DONE")
