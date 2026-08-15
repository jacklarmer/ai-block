"""Build v6 training/held-out layout: real photography joins the 'real' class.

v5's 'real' class was art/faces-heavy, so real editorial photographs
(Wikipedia etc.) occasionally false-flag. v6 adds diverse COCO real
photography and rebalances so genuine photos are well-represented.

Layout:
  data_v6/train/real/    -> FLAT: v5 real symlinks + COCO real (x N repeat)
  data_v6/train/fake/    -> symlink to v5 train/fake (all AI generators)
  data_v6/test/real/     -> COCO held-out slice (NEW, measures the fix)
  data_v6/test/fake/     -> existing per-gen + mobius + dalle (must not regress)

Usage: python build_v6.py <coco_dir> <n_test> <real_repeat>
"""
import sys, os, glob

coco_dir = sys.argv[1]
n_test = int(sys.argv[2])
REAL_REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 3
V5 = "/home/jack/aidet/data_v5"
OUT = "/home/jack/aidet/data_v6"

def rm(p):
    if os.path.islink(p) or os.path.exists(p):
        os.system(f"rm -rf {p}")

def ln(src, dst):
    rm(dst); os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(src, dst)

# fakes unchanged from v5 (all generators)
ln(f"{V5}/train/fake", f"{OUT}/train/fake")

# real: v5 reality (faces/art) + COCO photography, COCO repeated for weight
rm(f"{OUT}/train/real"); os.makedirs(f"{OUT}/train/real", exist_ok=True)
n_v5real = 0
for f in glob.glob(f"{V5}/train/real/*.jpg"):
    try:
        os.symlink(f, f"{OUT}/train/real/r5_{os.path.basename(f)}"); n_v5real += 1
    except Exception: pass
coco = sorted(glob.glob(os.path.join(coco_dir, "*.jpg")))
coco = [os.path.abspath(f) for f in coco]
train_coco = coco[:-n_test] if len(coco) > n_test else coco
test_coco = coco[-n_test:] if len(coco) > n_test else []
n_coco = 0
for rep in range(REAL_REPEAT):
    for f in train_coco:
        try:
            os.symlink(f, f"{OUT}/train/real/cc_{rep}_{os.path.basename(f)}"); n_coco += 1
        except Exception: pass
# held-out real test (measured directly for false-positives)
rm(f"{OUT}/test"); os.makedirs(f"{OUT}/test/real", exist_ok=True); os.makedirs(f"{OUT}/test/fake", exist_ok=True)
for f in test_coco:
    try:
        os.symlink(f, f"{OUT}/test/real/{os.path.basename(f)}")
    except Exception: pass
# fakes test slices from v5
for gdir in glob.glob(f"{V5}/test/fake/*"):
    g = os.path.basename(gdir)
    rm(f"{OUT}/test/fake/{g}")
    os.symlink(gdir, f"{OUT}/test/fake/{g}")
nreal_test = len(test_coco)
print(f"train/real: v5_real={n_v5real} coco_real={n_coco}", flush=True)
print(f"test/real (coco held-out): {nreal_test}", flush=True)
print("DONE", flush=True)
