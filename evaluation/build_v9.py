"""Build v9 layout: DEEPFAKE / synthetic-face class joins the fake side.

Adds 95k real-world deepfake / face-swap / synthetic-face images (from the
bitmind DeepfakeDataset AI.zip) as a new AI class — the #1 real-world AI
fraud / misinfo vector. The model learns a much harder, more general notion
of "this face/head is synthetic" that generalizes to swaps, not just
whole-image diffusion.

Layout mirrors prior builds (flat symlink farm the trainer can walk):
  data_v9/train/fake/  <- all existing AI (v7 fakes + v8 frontier) + deepfake
  data_v9/train/real/  <- 100% of v8 real classes (unchanged)
  data_v9/test/fake/deepfake/  <- held-out deepfake (200)
  data_v9/test/real/           <- held-out real (unchanged labels)
"""
import os, sys, random

SRC = sys.argv[1]            # data_v9_dl/deepfake_ai
HELDOUT = int(sys.argv[2])   # 200
REP = int(sys.argv[3])       # 4
OUT = "data_v9"
existing_dirs = ["data_v8"]

os.makedirs(f"{OUT}/test/fake/deepfake", exist_ok=True)
os.makedirs(f"{OUT}/test/real", exist_ok=True)

# deepfake files
df = []
for root, _, files in os.walk(SRC):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            df.append(os.path.join(root, f))

# existing classes (v8) : reuse symlinks directly (train AND test held-out)
for base in existing_dirs:
    for sub in ["train/fake", "train/real", "test/fake", "test/real"]:
        sd = os.path.join(base, sub)
        if not os.path.isdir(sd): continue
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
        for f in os.listdir(sd):
            os.symlink(os.path.abspath(os.path.join(sd, f)), f"{OUT}/{sub}/{f}")

os.makedirs(f"{OUT}/train/fake", exist_ok=True)

random.seed(0); random.shuffle(df)
held = df[:HELDOUT]; tr = df[HELDOUT:]

nf = 0
for f in held:
    os.symlink(os.path.abspath(f), f"{OUT}/test/fake/deepfake/dp_{os.path.basename(f)}")
nf = 0
for rep in range(REP):
    for f in tr:
        os.symlink(os.path.abspath(f), f"{OUT}/train/fake/dp_{rep}_{os.path.basename(f)}"); nf += 1

# report counts
for sub in ["train/fake", "train/real", "test/fake/deepfake", "test/real"]:
    p = f"{OUT}/{sub}"
    c = sum(len(files) for _,_,files in os.walk(p)) if os.path.isdir(p) else 0
    print(f"{sub}: {c}")
print("DONE")
