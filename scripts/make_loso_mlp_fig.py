import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("tileup/figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
folds = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
frozen = [0.4768, 0.5226, 0.4688, 0.5172, 0.4438, 0.4653]
lora = [0.7158, 0.7297, 0.7055, 0.6568, 0.7517, 0.7781]
mlp = [0.7938, 0.7422, 0.7733, 0.7028, 0.8277, 0.8366]
full = [0.7902, 0.7190, 0.7387, 0.6816, 0.8655, 0.8393]
x = np.arange(len(folds))
w = 0.2
fig, ax = plt.subplots(figsize=(8.2, 3.4))
ax.bar(x - 1.5*w, frozen, w, label="Frozen", color="#9aa8b8")
ax.bar(x - 0.5*w, lora, w, label="LoRA qkvo", color="#2457c5")
ax.bar(x + 0.5*w, mlp, w, label="qkvo + MLP", color="#6d3fc0")
ax.bar(x + 1.5*w, full, w, label="Full", color="#e0a33a")
ax.set_xticks(x)
ax.set_xticklabels([f"test={f}" for f in folds], fontsize=8.5)
ax.set_ylabel("Defect IoU")
ax.set_ylim(0.35, 0.90)
ax.grid(axis="y", alpha=0.2, linestyle="--")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=8.2)
plt.tight_layout()
plt.savefig(OUT / "fig_loso_mlp.png", dpi=240, bbox_inches="tight", facecolor="white")
print("saved", OUT / "fig_loso_mlp.png")
