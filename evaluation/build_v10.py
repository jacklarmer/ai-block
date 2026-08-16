"""Build v10 layout: add a LOW-RESOLUTION real class.

v9 exposed a resolution shortcut: real photos downscaled to 256px were
flagged ~97-99% as AI because the training real class was high-res while AI
images are low-res. v10 adds genuine low-res real photos to the real class so
the model stops treating low resolution as an AI tell.

Reuses v8 layout + adds data_v10_add/real_low into train/real.
Also keep a held-out low-res real slice (from unseen ImageNet, NOT trained on)
in test/real_low so we can PROVE the backend fix generalizes.
"""
import os, sys, glob

LOWRES_TRAIN = sys.argv[1]   # data_v10_add/real_low (COCO downscaled)
REAL_UNSEEN  = sys.argv[2]   # data_unseen/real (ImageNet, held OUT of training)
REP = int(sys.argv[3])       # repeats for low-res real in train
OUT = "data_v10"

def link_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            os.symlink(os.path.abspath(os.path.join(root, f)), os.path.join(dst, f)); n += 1
    return n

# 1) carry over ALL of data_v9 train + test (reuse symlinks)
for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
    sd = f"data_v9/{sub}"
    if os.path.isdir(sd):
        link_tree(sd, f"{OUT}/{sub}")

# 2) add low-res REAL photos into train/real (repeated)
os.makedirs(f"{OUT}/train/real", exist_ok=True)
lr = glob.glob(f"{LOWRES_TRAIN}/*.jpg")
n_lr = 0
for rep in range(REP):
    for f in lr:
        os.symlink(os.path.abspath(f), f"{OUT}/train/real/lowres_{rep}_{os.path.basename(f)}"); n_lr += 1

# 3) held-out low-res real slice from UNSEEN ImageNet (for backend proof)
os.makedirs(f"{OUT}/test/real_low", exist_ok=True)
n_hold = 0
for f in glob.glob(f"{REAL_UNSEEN}/*.jpg")[:300]:
    os.symlink(os.path.abspath(f), f"{OUT}/test/real_low/unseen_{os.path.basename(f)}"); n_hold += 1

for sub in ["train/fake", "train/real", "test/fake", "test/real_low", "test/real"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}")
print(f"low-res real added to train: {n_lr}")
print(f"unseen low-res held-out (ImageNet, not trained): {n_hold}")
print("DONE")
