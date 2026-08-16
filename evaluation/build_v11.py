"""Build v11 layout: REAL "looks-like-AI" photography subtype class.

v10 failed to fix the huge unseen-FP problem (87% on fresh unseen photos).
Root cause: the training real class is narrow (high-res COCO editorial +
WikiArt art); broad real-world photography subtypes (macros, clinical,
abstract, out-of-focus, heavy JPEG, low-light) legitimately resemble AI output
and are all false-flagged.

v11 adds 3000 real ImageNet (train-shard) photographs — spanning these hard
"looks-like-AI" subtypes — directly into the real training class, so the model
learns that smooth/low-detail real photos are still real. The FRESH 1500
ImageNet-val set is reserved as rigorously-unseen eval (never trained on).

Reuses v10 train/test, drops v10's ineffective low-res class (it didn't help),
adds ImageNet real photos to train/real.
"""
import os, sys, glob

IMGNET_TRAIN = sys.argv[1]   # data_unseen/real (3000)
IMGNET_EVAL  = sys.argv[2]   # data_unseen/real_eval (1500, held OUT of training)
REP = int(sys.argv[3])
OUT = "data_v11"

def link_tree(src, dst):
    os.makedirs(dst, exist_ok=True); n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, f)); n += 1
    return n

# carry over v10 train + test (reuse symlinks)
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v10/{sub}"
    if os.path.isdir(sd):
        link_tree(sd, f"{OUT}/{sub}")

# add ImageNet real-lookalike photos into train/real (repeated)
os.makedirs(f"{OUT}/train/real", exist_ok=True)
rn = sorted(glob.glob(f"{IMGNET_TRAIN}/*.jpg"))
n_add = 0
for rep in range(REP):
    for f in rn:
        os.symlink(os.path.abspath(f), f"{OUT}/train/real/imnet_{rep}_{os.path.basename(f)}"); n_add += 1

# fresh unseen eval slice (ImageNet val, NOT trained)
os.makedirs(f"{OUT}/test/real_unseen", exist_ok=True)
n_eval = 0
for f in sorted(glob.glob(f"{IMGNET_EVAL}/*.jpg")):
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_unseen/unseen_{os.path.basename(f)}"); n_eval += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real", "test/real_unseen"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}")
print(f"ImageNet real-lookalike added to train: {n_add}")
print(f"FRESH unseen val eval slice: {n_eval}")
print("DONE")
