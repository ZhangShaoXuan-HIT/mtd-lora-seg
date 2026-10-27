import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1f2933"
MUTED = "#5b6b7f"
LINE = "#cbd5e1"
FROZEN_FILL = "#eef2f6"
FROZEN_EDGE = "#94a3b8"
LORA = "#d97706"
LORA_FILL = "#fff4e5"
GREEN = "#2f7d5c"
GREEN_FILL = "#eaf6ef"
RED = "#b94a44"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.linewidth": 0.0,
    "svg.fonttype": "none",
})

fig = plt.figure(figsize=(183/25.4, 108/25.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 183)
ax.set_ylim(0, 108)
ax.set_aspect("equal")
ax.axis("off")

def box(x, y, w, h, fc, ec, lw=0.9, rad=2.0, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, color=INK, lw=1.4, ls="-", rad=0.0, z=4, ms=7):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, linestyle=ls, zorder=z,
                        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(p)

FONT_SCALE = 1.34

def label(x, y, s, size=7.2, color=INK, weight="normal", ha="center", va="center", z=6, rotation=0):
    ax.text(x, y, s, fontsize=size * FONT_SCALE, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, linespacing=1.05, rotation=rotation)

# panel letters
label(2.0, 104.0, "a", size=9, weight="bold", ha="left")
label(2.0, 40.0, "b", size=9, weight="bold", ha="left")

# ---------------- (a) pipeline ----------------
# input card
box(4, 76, 22, 18, WHITE, LINE, lw=0.9)
img_path = ROOT / "data" / "Magnetic-tile-defect-datasets.-master" / "MT_Crack" / "Imgs" / "exp6_num_116623.jpg"
mask_path = ROOT / "data" / "Magnetic-tile-defect-datasets.-master" / "MT_Crack" / "Imgs" / "exp6_num_116623.png"
im = np.array(Image.open(img_path).convert("L"))
mask = np.array(Image.open(mask_path).convert("L"))
ax.imshow(im, cmap="gray", extent=[5.2, 24.8, 79.0, 90.2], aspect="auto", zorder=2)
ax.add_patch(Rectangle((5.2, 79.0), 19.6, 11.2, fill=False, ec=LINE, lw=0.6, zorder=3))
label(15, 95.0, "Input", size=8.0, weight="bold")
label(15, 92.4, "384 × 384", size=6.4, color=MUTED)
arrow(26.5, 85, 34.0, 88, color=INK, lw=1.7)

# encoder container
box(32, 45, 72, 55, WHITE, MUTED, lw=1.0)
label(34.0, 97.2, "SegFormer-B0 encoder", size=8.2, weight="bold", ha="left")
box(91.5, 94.4, 11.0, 4.2, FROZEN_FILL, FROZEN_EDGE, lw=0.7, rad=1.2)
label(97.0, 96.5, "frozen", size=6.2, color=MUTED)

stages = [
    ("S1", "96 × 96 · 32", 82.0, 4),
    ("S2", "48 × 48 · 64", 70.0, 3),
    ("S3", "24 × 24 · 160", 58.0, 2),
    ("S4", "12 × 12 · 256", 46.0, 2),
]
for i, (name, shape, y, g) in enumerate(stages):
    box(35.0, y, 66.0, 10.0, FROZEN_FILL, FROZEN_EDGE, lw=0.7, rad=1.4)
    label(37.0, y + 6.8, name, size=7.6, weight="bold", ha="left")
    label(37.0, y + 3.6, shape, size=6.3, color=MUTED, ha="left")
    # feature-map glyph
    gx, gy = 66.0, y + 2.2
    cell = 1.3
    for r in range(g):
        for c in range(g):
            ax.add_patch(Rectangle((gx + c*cell, gy + r*cell), cell*0.82, cell*0.82,
                                   facecolor="#8fa3b8", edgecolor=WHITE, lw=0.3, zorder=3))
    label(86.0, y + 5.0, "×2", size=6.4, color=MUTED)
    if i < 3:
        arrow(68.0, y - 0.2, 68.0, y - 1.6, color=MUTED, lw=1.2, ms=6)

# input into S1
arrow(34.2, 86.5, 35.0, 86.5, color=INK, lw=1.7, ms=6)

# multi-scale bus
bus_x = 106.5
for _, _, y, _ in stages:
    ax.plot([101.2, bus_x], [y + 5.0, y + 5.0], color=MUTED, lw=1.0, zorder=3)
    ax.add_patch(Circle((bus_x, y + 5.0), 0.55, facecolor=MUTED, edgecolor=WHITE, lw=0.4, zorder=4))
ax.plot([bus_x, bus_x], [50.0, 87.0], color=MUTED, lw=1.2, zorder=3)
arrow(bus_x, 69.0, 112.0, 69.0, color=INK, lw=1.7)

# decoder
box(112, 57, 30, 25, GREEN_FILL, GREEN, lw=1.1)
label(127, 77.0, "Decoder head", size=7.2, weight="bold")
label(127, 73.2, "trainable", size=6.4, color=GREEN)
label(127, 69.8, "0.395M params", size=6.2, color=MUTED)
arrow(142.5, 69.0, 146.0, 69.0, color=INK, lw=1.7)

# mask output
box(146, 57, 30, 25, WHITE, LINE, lw=0.9)
# overlay image with red prediction
rgb = np.stack([im, im, im], axis=-1).astype(float) / 255.0
out = rgb.copy()
m = mask > 127
out[m] = 0.55 * out[m] + 0.45 * np.array([0.91, 0.29, 0.27])
ax.imshow(out, extent=[147.2, 174.8, 60.2, 80.4], aspect="auto", zorder=2)
ax.add_patch(Rectangle((147.2, 60.2), 27.6, 20.2, fill=False, ec=LINE, lw=0.6, zorder=3))
label(161, 54.8, "Defect mask", size=7.6, weight="bold")

# budget bar
label(112.0, 49.5, "trainable 0.543M (14.6%)", size=6.2, color=INK, ha="left")
ax.add_patch(Rectangle((112.0, 46.8), 64.0, 1.6, facecolor="#e2e8f0", edgecolor="none", zorder=2))
ax.add_patch(Rectangle((112.0, 46.8), 64.0*0.146, 1.6, facecolor=LORA, edgecolor="none", zorder=3))
ax.add_patch(Rectangle((112.0, 46.8), 64.0, 1.6, fill=False, edgecolor=LINE, lw=0.6, zorder=4))

# zoom connector
ax.plot([68.0, 68.0], [45.0, 42.0], color=MUTED, lw=0.9, linestyle=(0, (2.2, 2.2)), zorder=4)
label(70.0, 43.4, "block detail", size=6.0, color=MUTED, ha="left")

# ---------------- (b) block detail ----------------
box(4, 4, 172, 36, WHITE, LINE, lw=0.9)
label(6.0, 37.6, "Transformer block (frozen encoder)", size=7.4, weight="bold", ha="left")

# flow boxes
flow_y, flow_h = 22.0, 11.0
box(7, flow_y, 15, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(14.5, flow_y + 7.3, "LN", size=6.8, weight="bold")
label(14.5, flow_y + 3.6, "norm", size=5.8, color=MUTED)

box(26, flow_y, 42, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(47.0, flow_y + 7.5, "Efficient Self-Attention", size=6.6, weight="bold")
label(47.0, flow_y + 3.8, "q  k  v  o", size=6.2, color=MUTED)

box(72, flow_y, 11, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(77.5, flow_y + 5.5, "Add", size=6.4)

box(87, flow_y, 15, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(94.5, flow_y + 7.3, "LN", size=6.8, weight="bold")
label(94.5, flow_y + 3.6, "norm", size=5.8, color=MUTED)

box(106, flow_y, 42, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(127.0, flow_y + 7.5, "Mix-FFN", size=6.8, weight="bold")
label(127.0, flow_y + 3.8, "fc1  GELU  fc2", size=6.0, color=MUTED)

box(152, flow_y, 15, flow_h, FROZEN_FILL, FROZEN_EDGE, lw=0.7)
label(159.5, flow_y + 5.5, "Add", size=6.4)

# flow arrows
for x1, x2 in [(22.0, 25.6), (68.0, 71.6), (83.0, 86.6), (102.0, 105.6), (148.0, 151.6)]:
    arrow(x1, flow_y + 5.5, x2, flow_y + 5.5, color=INK, lw=1.3, ms=6)

# LoRA boxes
box(26, 8.0, 42, 8.2, LORA_FILL, LORA, lw=1.1)
label(47.0, 13.8, "LoRA q/k/v/o  ·  r = 8", size=6.2, color=LORA, weight="bold")
label(47.0, 10.5, "B·A + W  (W frozen)", size=5.9, color=MUTED)
for x in [32.0, 41.0, 50.0, 59.0]:
    arrow(x, 16.4, x, flow_y + 0.2, color=LORA, lw=1.0, ms=5)

box(106, 8.0, 42, 8.2, LORA_FILL, LORA, lw=1.1)
label(127.0, 13.8, "LoRA fc1/fc2  ·  r = 8", size=6.2, color=LORA, weight="bold")
label(127.0, 10.5, "low-rank channel mixing", size=5.9, color=MUTED)
for x in [116.0, 138.0]:
    arrow(x, 16.4, x, flow_y + 0.2, color=LORA, lw=1.0, ms=5)



fig.savefig(OUT / "fig_seg_arch_v2.png", dpi=300, facecolor=WHITE,
            bbox_inches=None, pad_inches=0)
fig.savefig(OUT / "fig_seg_arch_v2.svg", facecolor=WHITE,
            bbox_inches=None, pad_inches=0)
fig.savefig(OUT / "fig_seg_arch_v2.pdf", facecolor=WHITE,
            bbox_inches=None, pad_inches=0)
print("saved v2")
