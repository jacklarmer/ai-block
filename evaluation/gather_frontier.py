"""Gather FRONTIER AI-image outputs for the v8 fine-tune.

Sweeps the bitmind image-benchmarks daily snapshots (Aug-Sep 2025) and pulls
rows from the NEWER, harder-to-detect generators — the frontier models that
post-date our DALL-E/SDXL-heavy training: CogView4-6B, Gemini-2.5-flash-image,
FLUX.1-dev, Janus-Pro-7B, RealVisXL. These outputs are the closest thing to
"what people are posting now" with trustworthy AI/real labels.

Only rows labelled AI (bitmind: 1 or 2, i.e. not -1) are kept. Dedup by media
hash within a session via a `seen` set and resume-friendly listing.

Usage: python gather_frontier.py <outdir> <max_n>
"""
import sys, os, time
from PIL import Image
from datasets import load_dataset

OUT, MAXN = sys.argv[1], int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
seen = set(os.listdir(OUT))   # resume: skip files already saved

# frontier generators to prefer (newer / more realistic than SDXL-era)
PREFER = ["cogview4", "gemini-2.5-flash", "janus", "flux.1", "realvisxl", "cogview"]

# full bitmind daily-snapshot time series (Aug 21 - Sep 16 2025), newest first;
# each holds frontier-generator AI output mixed with real. Dedup via media hash.
SNAPS = [
    "split_20250916_181648","split_20250915_141646","split_20250914_181635",
    "split_20250913_201617","split_20250912_171724","split_20250911_191641",
    "split_20250910_211645","split_20250909_231638","split_20250908_231634",
    "split_20250908_031640","split_20250907_061726","split_20250906_101618",
    "split_20250905_112051","split_20250904_141819","split_20250903_171626",
    "split_20250902_181639","split_20250901_221746","split_20250901_051630",
    "split_20250831_101744","split_20250830_121729","split_20250829_152007",
    "split_20250828_171741","split_20250827_171634","split_20250826_201630",
    "split_20250826_011735","split_20250825_011818","split_20250824_091637",
    "split_20250823_212018","split_20250823_101844","split_20250822_081634",
    "split_20250821_203929","split_20250821_201430","split_20250821_200935",
    "split_20250821_195938","split_20250821_194436","split_20250821_192503",
    "split_20250821_191036","split_20250821_185537","split_20250821_184448",
    "split_20250821_182427","split_20250821_180425","split_20250821_173926",
    "split_20250821_170937","split_20250821_163927","split_20250821_161924",
    "split_20250821_155424","split_20250821_152925","split_20250821_150928",
    "split_20250821_145425","split_20250821_143429","split_20250821_142925",
    "split_20250821_141427","split_20250821_140445","split_20250821_134425",
    "split_20250821_133425","split_20250821_131426","split_20250821_130425",
    "split_20250821_124926","split_20250821_123925","split_20250821_122924",
    "split_20250821_120925",
]
print("sweeping", len(SNAPS), "snapshots", flush=True)

n = 0
last_save = time.time()
tried = set()
for cfg in SNAPS:
    if n >= MAXN:
        break
    try:
        ds = load_dataset("bitmind/bm-image-benchmarks", cfg, split="train", streaming=True)
    except Exception as e:
        print("cfg err", cfg, str(e)[:50], flush=True); continue
    it = iter(ds)
    while n < MAXN:
        try:
            row = next(it)
        except StopIteration:
            break
        except Exception:
            continue
        name = (row.get("model_name") or "")
        low = name.lower()
        # only frontier-relevant, and only AI rows (label != -1). Prefer the
        # newer gens; skip pure SDXL-shadows lanes we already have.
        if row.get("label") == -1:
            continue
        if not any(p in low for p in PREFER):
            continue
        key = str(row.get("media_hash") or "")
        if not key or f"{key}.jpg" in seen:
            continue
        img = row.get("media_image")
        if img is None:
            continue
        try:
            im = img.convert("RGB"); im.thumbnail((256, 256), Image.LANCZOS)
            im.save(os.path.join(OUT, f"{key}.jpg"), "JPEG", quality=92)
            seen.add(f"{key}.jpg"); n += 1; last_save = time.time()
            if n % 1000 == 0:
                print("  n=", n, "cfg=", cfg, flush=True)
        except Exception:
            continue
        if time.time() - last_save > 200:
            print("STALL break in cfg", cfg, flush=True); break
print(f"GATHER FRONTIER DONE: +{n}", flush=True)
