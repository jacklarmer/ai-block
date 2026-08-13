"""
Build v3 training data:
  Training   real: hemg real + bmreal + MS-COCO real + AFHQ training slice
  Training   fake: hemg fake + all bitmind_ai generators (sdxl,biggan,glide,wukong,
                   vqdm,adm,midjourney,flux)  [NOT mobius]
  Held-out   real: AFHQ held-out slice (unseen real photos)
  Held-out   fake: bitmind mobius (unseen generator)

Layout:
  data_v3/train/{real,fake}
  data_v3/test/{real,fake}
"""
import os, glob, random, shutil, hashlib

random.seed(7)

def hash_file(p, chunk=65536):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def copy_dedup(src_files, outdir):
    os.makedirs(outdir, exist_ok=True)
    seen=set(); n=0
    for p in src_files:
        try: h=hash_file(p)
        except Exception: continue
        if h in seen: continue
        seen.add(h)
        ext=p.split(".")[-1]
        shutil.copy(p, os.path.join(outdir, f"i_{n}.{ext}"))
        n+=1
    return n

base="/home/jack/aidet/data"
v3="/home/jack/aidet/data_v3"
os.makedirs(os.path.join(v3,"train"), exist_ok=True)
os.makedirs(os.path.join(v3,"test"), exist_ok=True)

# ---- REAL training: hemg + bmreal + coco + afhq(training slice) ----
real_train = (
    glob.glob(base+"/hemg_img/real/*") +
    glob.glob(base+"/bitmind_real/*") +
    glob.glob(base+"/test_heldout/real/coco_*")   # COCO -> training
)
afhq = glob.glob(base+"/test_heldout/real/afhq_*")
random.shuffle(afhq)
split = int(len(afhq)*0.5)
real_train += afhq[split:]          # 50% afhq -> training
afhq_hold = afhq[:split]            # 50% afhq -> held-out real
print("real_train_files", len(real_train), flush=True)

# ---- FAKE training: all bitmind_ai generators except mobius (mobius is held-out) ----
fake_train_files = glob.glob(base+"/hemg_img/fake/*")
for tag in ["sdxl","biggan","glide","wukong","vqdm","adm","midjourney","flux"]:
    fake_train_files += glob.glob(base+"/bitmind_ai/"+tag+"_*")
print("fake_train_files", len(fake_train_files), flush=True)

n_real = copy_dedup(real_train, os.path.join(v3,"train","real"))
n_fake = copy_dedup(fake_train_files, os.path.join(v3,"train","fake"))
print("train real", n_real, "fake", n_fake, flush=True)

# ---- held-out test ----
mobius = glob.glob(base+"/test_heldout/fake/mobius_*")
n_hold_fake = copy_dedup(mobius, os.path.join(v3,"test","fake"))
n_hold_real = copy_dedup(afhq_hold, os.path.join(v3,"test","real"))
print("test real", n_hold_real, "fake", n_hold_fake, flush=True)
print("DONE", flush=True)
