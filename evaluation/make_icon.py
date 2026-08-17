#!/usr/bin/env python3
"""Generate the AI Block logo: a red NO/prohibition symbol over a ROBOT glyph
(per Jack — robot emoji, not computer). Renders crisp geometric shapes.
"""
from PIL import Image, ImageDraw
import math, os

def render_icon(size):
    S = size * 8  # supersample for crisp edges
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    I = lambda v: int(round(v))

    # ---- robot head (light slate shell) ----
    # main head: rounded rectangle
    head_l, head_t = S * 0.18, S * 0.16
    head_r, head_b = S * 0.82, S * 0.68
    d.rounded_rectangle(
        [I(head_l), I(head_t), I(head_r), I(head_b)],
        radius=I(S * 0.10), fill=(196, 204, 216, 255)
    )
    # face panel (darker, like a screen/face plate)
    pl = head_l + S * 0.055; pt = head_t + S * 0.09
    pr = head_r - S * 0.055; pb = head_b - S * 0.09
    d.rounded_rectangle(
        [I(pl), I(pt), I(pr), I(pb)],
        radius=I(S * 0.05), fill=(52, 62, 80, 255)
    )
    # two eyes (light dots)
    ey = pt + S * 0.10
    er = S * 0.075
    for e in (S * 0.38, S * 0.62):
        d.ellipse([I(e - er), I(ey - er), I(e + er), I(ey + er)], fill=(226, 234, 244, 255))
    # antenna
    ax = S * 0.5
    d.line([I(ax), I(head_t - S*0.02), I(ax), I(head_t - S*0.10)], fill=(160,170,184,255), width=I(S*0.02))
    d.ellipse([I(ax - S*0.035), I(head_t - S*0.16), I(ax + S*0.035), I(head_t - S*0.09)], fill=(232,60,60,255))

    # ---- NO / prohibition symbol over the robot ----
    cx = S * 0.5
    cy = S * 0.44
    r = S * 0.30
    d.ellipse(
        [I(cx - r), I(cy - r), I(cx + r), I(cy + r)],
        outline=(232, 60, 60, 255), width=I(S * 0.08)
    )
    sl_w = I(S * 0.08)
    x1, y1 = I(cx - r * 0.60), I(cy - r * 0.60)
    x2, y2 = I(cx + r * 0.60), I(cy + r * 0.60)
    d.line([x1, y1, x2, y2], fill=(232, 60, 60, 255), width=sl_w)

    img = img.resize((size, size), Image.LANCZOS)
    return img

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icons")
os.makedirs(outdir, exist_ok=True)
for s in (16, 48, 128):
    im = render_icon(s)
    p = os.path.join(outdir, f"icon{s}.png")
    im.save(p)
    print("wrote", os.path.normpath(p), im.size)
