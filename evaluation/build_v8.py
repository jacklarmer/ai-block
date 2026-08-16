"""Build v8 layout: FRONTIER AI class joins the fake side.

Adds newer, harder-to-detect AI generators (CogView4-6B, Gemini-2.5-flash,
FLUX.1-dev, Janus-Pro, RealVisXL from bitmind benchmark snapshots) as a new
repeated fake class, so the model learns current frontier AI output — not just
the older DALL-E/SDXL-era stuff. Real photo + real art classes are kept intact
(no regression).

Layout (flat, symlink-friendly root/clazz/*.*):
  train/fake = v7 train/fake + frontier (repeated xR)
  train/real = v7 train/real  (unchanged)
  test/fake  = v7 per-gen slices + frontier held-out
  test/real  = v7 real (photo + art)

Usage: python build_v8.py <frontier_dir> <holdout> <repeat>
"""
import os, sys, glob, shutil

V7 = "/home/jack/aidet/data_v7"
FR = sys.argv[1]; HO = int(sys.argv[2]); REP = int(sys.argv[3])
OUT = "/home/jack/aidet/data_v8"

def rm(p):
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
    elif os.path.isdir(p): shutil.rmtree(p)

rm(OUT); os.makedirs(f"{OUT}/train/real"); os.makedirs(f"{OUT}/train/fake")
os.makedirs(f"{OUT}/test/real"); os.makedirs(f"{OUT}/test/fake")

# ---- real: unchanged from v7 ----
n_real = 0
for f in sorted(glob.glob(f"{V7}/train/real/*")):
    os.symlink(f, f"{OUT}/train/real/{os.path.basename(f)}"); n_real += 1

# ---- fake: v7 fakes + frontier (hold out HO, repeat rest REP) ----
n_fake_v7 = 0
for f in sorted(glob.glob(f"{V7}/train/fake/*")):
    os.symlink(f, f"{OUT}/train/fake/{os.path.basename(f)}"); n_fake_v7 += 1

frs = sorted(glob.glob(f"{FR}/*"))
fr_test, fr_train = frs[:HO], frs[HO:]
n_fr = 0
for rep in range(REP):
    for f in fr_train:
        os.symlink(f, f"{OUT}/train/fake/fr_{rep}_{os.path.basename(f)}"); n_fr += 1
os.makedirs(f"{OUT}/test/fake/frontier", exist_ok=True)
for f in fr_test:
    os.symlink(f, f"{OUT}/test/fake/frontier/{os.path.basename(f)}")

# ---- real test: from v7 ----
for f in sorted(glob.glob(f"{V7}/test/real/*")):
    os.symlink(f, f"{OUT}/test/real/{os.path.basename(f)}")

# ---- fake test slices from v7 ----
for gdir in glob.glob(f"{V7}/test/fake/*"):
    rm(f"{OUT}/test/fake/{os.path.basename(gdir)}")
    os.symlink(gdir, f"{OUT}/test/fake/{os.path.basename(gdir)}")

print(f"train/fake: v7={n_fake_v7} frontier={n_fr} (rep {REP})", flush=True)
print(f"train/real: {n_real}", flush=True)
print(f"test/fake/frontier held-out: {len(fr_test)}", flush=True)
print("DONE", flush=True)
