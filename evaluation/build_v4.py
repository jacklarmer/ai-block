"""Build v4 training/held-out layout from existing v3 data + gathered DALL-E 3.

Flat layout (matches train script's root/{real,fake} glob):
  data_v4/train/real/   -> symlink to data_v3/train/real
  data_v4/train/fake/   -> FLAT: symlinks to all v3 fakes + first n_train DALL-E
  data_v4/test/real/    -> symlink to data_v3/test/real (unseen AFHQ)
  data_v4/test/fake/mobius -> symlink to data_v3/test/fake (unseen Mobius)
  data_v4/test/fake/dalle  -> LAST n_test gathered DALL-E images (held-out)

Holding out the LAST-gathered DALL-E is important: a DALL-E sample used in
training must not also be measured as 'unseen', or the eval is inflated.
Usage: python build_v4.py <dalle_dir> <n_train> <n_test>
"""
import sys, os, glob

dalle_dir = sys.argv[1]
n_train = int(sys.argv[2])
n_test = int(sys.argv[3])
V3 = "/home/jack/aidet/data_v3"
OUT = "/home/jack/aidet/data_v4"

def rm(p):
    if os.path.islink(p) or os.path.exists(p):
        os.system(f"rm -rf {p}")

def ln(src, dst):
    rm(dst); os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(src, dst)

# train real
ln(f"{V3}/train/real", f"{OUT}/train/real")
# test real + mobius
ln(f"{V3}/test/real", f"{OUT}/test/real")
ln(f"{V3}/test/fake", f"{OUT}/test/fake/mobius")

# flat train/fake: symlink each v3 fake (namespaced to avoid collision) + DALL-E
rm(f"{OUT}/train/fake")
os.makedirs(f"{OUT}/train/fake", exist_ok=True)
n_v3 = 0
for f in glob.glob(f"{V3}/train/fake/*.jpg"):
    try:
        os.symlink(f, f"{OUT}/train/fake/v3_{os.path.basename(f)}"); n_v3 += 1
    except Exception:
        pass

dalle = sorted(glob.glob(os.path.join(dalle_dir, "*.jpg")))
# Resolve to ABSOLUTE paths so the symlink targets are valid regardless of CWD.
dalle = [os.path.abspath(f) for f in dalle]
print(f"gathered DALL-E total {len(dalle)}", flush=True)
train_slice = dalle[:n_train]
test_slice = dalle[-n_test:]
os.makedirs(f"{OUT}/test/fake/dalle", exist_ok=True)
n_dtr = 0
# Repeat DALL-E train slice 4x so the novel generator gets enough gradient
# signal (it is ~3% of the fake pool otherwise and would be under-learned).
for rep in range(4):
    for f in train_slice:
        try:
            os.symlink(f, f"{OUT}/train/fake/dalle_{rep}_{os.path.basename(f)}"); n_dtr += 1
        except Exception:
            pass
for f in test_slice:
    try:
        os.symlink(f, f"{OUT}/test/fake/dalle/" + os.path.basename(f))
    except Exception:
        pass
print(f"train/fake: v3={n_v3}, dalle_train={n_dtr}", flush=True)
print(f"test/fake/dalle held-out={len(os.listdir(f'{OUT}/test/fake/dalle'))}", flush=True)
print("DONE", flush=True)
