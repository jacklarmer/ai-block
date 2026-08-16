"""Generate FRESH frontier AI-image samples on the RTX 5090 ONLY.

Uses zai-org/CogView4-6B — a current (2025), MIT-licensed, ungated frontier
text-to-image model — to synthesize brand-new, photographic-style outputs that
our training has never seen. These are genuinely-"fresh frontier" AI samples
(as opposed to scraped benchmarks).

SAFETY: this script MUST run on the RTX 5090 ONLY (Jack's explicit constraint).
It pins CUDA_VISIBLE_DEVICES and hard-asserts at runtime that the active device
is the RTX 5090; if the environment maps GPU 0 to the PRO 6000 it refuses to run
rather than touch the production lane.

Usage: CUDA_VISIBLE_DEVICES=1 python generate_frontier.py <outdir> <n>
"""
import os, sys, time, random
import torch
from PIL import Image

OUT = sys.argv[1]
N = int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)

# ---------- hard device guard: must be the RTX 5090 ----------
devname = torch.cuda.get_device_name(0)
assert "RTX 5090".lower() in devname.lower(), f"REFUSING: active device is {devname!r}, not the RTX 5090"
print("device OK (RTX 5090):", devname, flush=True)

from diffusers import CogView4Pipeline

t0 = time.time()
print("loading CogView4-6B in fp16...", flush=True)
pipe = CogView4Pipeline.from_pretrained(
    "zai-org/CogView4-6B",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="/home/jack/aidet/model_cache",
).to("cuda")
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

# diverse photographic/frontier prompts — the realistic, hard-to-detect cases
prompts = [
    "A candid 50mm photograph of a barista at a small coffee shop, natural window light, shallow depth of field",
    "A travel photo of a narrow Venice alley with laundry hanging between old brick buildings, golden hour",
    "A macro photograph of a monarch butterfly on a purple wildflower, dew droplets, sharp detail",
    "A documentary-style photo of an elderly farmer in a wheat field, late afternoon sun, film grain",
    "A portrait of a young woman with freckles in a rain-soaked city street, neon bokeh reflections",
    "A photograph of a vintage 1967 Mustang in a desert at dusk, warm tones, cinematic",
    "A close-up photograph of a chef plating a gourmet dish, steam rising, restaurant kitchen",
    "An outdoor photo of a family hiking a misty mountain trail, spring wildflowers, candid",
    "A photo of a tattooed skateboarder mid-trick in a graffiti-covered concrete park, motion blur",
    "A quiet photo of a reading nook by a window with a sleeping cat, soft afternoon light, books",
    "A wildlife photo of a red fox in snow at twilight, alert ears up, shallow focus",
    "A street photograph of a crowded night market in Tokyo, lanterns and steam, warm glow",
    "A product photo of a matte black espresso machine on a wood counter, minimal, clean studio light",
    "A candid photo of children playing in a fountain on a hot summer day, water droplets sparkling",
    "A photo of a lighthouse on a stormy coast, dramatic clouds, waves crashing, moody",
    "A portrait of a musician with a vintage acoustic guitar on a rooftop at sunset, backlit",
    "An aerial photograph of turquoise coastal waters meeting white sand, geometric patterns",
    "A photo of a cozy cabin interior with a wood-burning stove, warm firelight, rustic",
    "A candid photo of a nurse smiling in a bright modern hospital corridor, natural light",
    "A photo of a speedboat wake crossing a calm fjord at sunrise, mirror reflections",
]

rng = random.Random(2026)
seed = 2026
os.makedirs(OUT, exist_ok=True)
existing = set(os.listdir(OUT))
n = 0
try:
    while n < N:
        # vary prompts (cycle) + inference steps + seed for diversity
        p = prompts[n % len(prompts)]
        steps = rng.choice([24, 28, 32])
        try:
            g = torch.Generator(device="cuda").manual_seed(seed + n)
            img = pipe(p, num_inference_steps=steps, guidance_scale=6.0, generator=g).images[0]
            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            fn = f"cogview4_{seed+n:05d}.jpg"
            if fn not in existing:
                img.save(os.path.join(OUT, fn), "JPEG", quality=92)
                n += 1
                if n % 100 == 0:
                    print("  generated", n, flush=True)
        except Exception as e:
            print("gen err:", str(e)[:80], flush=True)
            continue
finally:
    print(f"GENERATE FRONTIER DONE: +{n} (CogView4-6B)", flush=True)
