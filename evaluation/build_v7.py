"""Build v7 layout: real ART (WikiArt) joins the 'real' class.

v6 fixed real-PHOTO false-positives (COCO), but still false-flags real
ARTWORK/illustrations ~34-42% (paintings, historical plates, diagrams look
"clean"/AI-ish). v7 adds 12k real human-made WikiArt images so the model learns
polished/artistic does NOT imply AI.

Layout (flat, symlink-friendly for the trainer's root/clazz/*.* glob):
  train/real  = v6 train/real  + real_art (repeated xR)
  train/fake  = v6 train/fake  (unchanged)
  test/real   = held-out COCO (photo FP) + held-out WikiArt (art FP)
  test/fake   = v6 per-generator slices (regression)

Usage: python build_v7.py <realart_dir> <art_holdout> <art_repeat> <coco_dir> <coco_holdout>
"""
import os, sys, glob, shutil

V6 = "/home/jack/aidet/data_v6"
ART = sys.argv[1]; ART_HO = int(sys.argv[2]); ART_REP = int(sys.argv[3])
COCO_DIR = sys.argv[4]; COCO_HO = int(sys.argv[5])
OUT = "/home/jack/aidet/data_v7"

def rm(p):
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
    elif os.path.isdir(p): shutil.rmtree(p)

rm(OUT); os.makedirs(f"{OUT}/train/real"); os.makedirs(f"{OUT}/train/fake")
os.makedirs(f"{OUT}/test/real"); os.makedirs(f"{OUT}/test/fake")

# ---- fakes unchanged from v6 ----
n_fake = 0
for f in sorted(glob.glob(f"{V6}/train/fake/*")):
    os.symlink(f, f"{OUT}/train/fake/{(os.path.basename(f))}"); n_fake += 1

# ---- real: v6 reals ----
n_v6real = 0
for f in sorted(glob.glob(f"{V6}/train/real/*")):
    os.symlink(f, f"{OUT}/train/real/{(os.path.basename(f))}"); n_v6real += 1

# ---- real-art (new class): hold out ART_HO for test, repeat rest ART_REP ----
arts = sorted(glob.glob(f"{ART}/*"))
art_test, art_train = arts[:ART_HO], arts[ART_HO:]
n_art_train = 0
for rep in range(ART_REP):
    for f in art_train:
        os.symlink(f, f"{OUT}/train/real/art_{rep}_{os.path.basename(f)}"); n_art_train += 1
n_art_test = len(art_test)
for f in art_test:
    os.symlink(f, f"{OUT}/test/real/art_{os.path.basename(f)}")

# ---- COCO held-out real (photo FP) ----
cocos = sorted(glob.glob(f"{COCO_DIR}/*"))
coco_test = cocos[:COCO_HO]
n_coco_test = len(coco_test)
for f in coco_test:
    os.symlink(f, f"{OUT}/test/real/coco_{os.path.basename(f)}")

# ---- fake test slices from v6 ----
for gdir in glob.glob(f"{V6}/test/fake/*"):
    rm(f"{OUT}/test/fake/{os.path.basename(gdir)}")
    os.symlink(gdir, f"{OUT}/test/fake/{os.path.basename(gdir)}")

print(f"train/fake: {n_fake}", flush=True)
print(f"train/real: v6_real={n_v6real} art={n_art_train} (rep {ART_REP})", flush=True)
print(f"test/real: art={n_art_test} coco={n_coco_test}", flush=True)
print("DONE", flush=True)
