"""Build v5 training/held-out layout from v4 data + newly gathered generators.

Layout (flat, matches train script root/{real,fake} glob):
  data_v5/train/real/    -> symlink to data_v3/train/real (unchanged real pool)
  data_v5/train/fake/    -> FLAT symlinks: v4 train fakes + per-gen new fakes
  data_v5/test/real/     -> symlink to data_v3/test/real (unseen AFHQ)
  data_v5/test/fake/<gen> -> held-out slice per new generator + mobius/dalle

For each newly gathered generator we hold out the LAST ~200 as an unseen test
slice and put the rest in training (x4 repeat so novel styles get gradient).
Usage: python build_v5.py <add_dir> <n_test_per_gen> [train_repeat]
"""
import sys, os, glob

add_dir = sys.argv[1]
n_test = int(sys.argv[2])
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
V3 = "/home/jack/aidet/data_v3"
V4 = "/home/jack/aidet/data_v4"
OUT = "/home/jack/aidet/data_v5"

def rm(p):
    if os.path.islink(p) or os.path.exists(p):
        os.system(f"rm -rf {p}")

def ln(src, dst):
    rm(dst); os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(src, dst)

# train real + test real + mobius held-out (from v4)
ln(f"{V3}/train/real", f"{OUT}/train/real")
ln(f"{V3}/test/real", f"{OUT}/test/real")
ln(f"{V3}/test/fake", f"{OUT}/test/fake/mobius")

# train/fake: reuse v4 train fakes (v3 + dalle) + add new generators
rm(f"{OUT}/train/fake"); os.makedirs(f"{OUT}/train/fake", exist_ok=True)
n_v4 = 0
for f in glob.glob(f"{V4}/train/fake/*.jpg"):
    try:
        os.symlink(f, f"{OUT}/train/fake/v4_{os.path.basename(f)}"); n_v4 += 1
    except Exception: pass
os.makedirs(f"{OUT}/test/fake", exist_ok=True)
total_new = 0
for gen in sorted(os.listdir(add_dir)):
    gd = os.path.join(add_dir, gen)
    if not os.path.isdir(gd):
        continue
    files = sorted(glob.glob(os.path.join(gd, "*.jpg")))
    files = [os.path.abspath(f) for f in files]
    train = files[:-n_test] if len(files) > n_test else files
    test = files[-n_test:] if len(files) > n_test else []
    # training (REP repeats), namespaced
    for rep in range(REP):
        for f in train:
            try:
                os.symlink(f, f"{OUT}/train/fake/{gen}_{rep}_{os.path.basename(f)}")
            except Exception: pass
    total_new += len(train) * REP
    # held-out test slice
    tg = f"{OUT}/test/fake/{gen}"
    os.makedirs(tg, exist_ok=True)
    for f in test:
        try:
            os.symlink(f, f"{tg}/{os.path.basename(f)}")
        except Exception: pass
    print(f"  gen {gen}: train={len(train)*REP} test={len(test)}", flush=True)

print(f"train/fake: v4={n_v4} new_fake={total_new}", flush=True)
print("DONE", flush=True)
