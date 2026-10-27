import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

OUT = Path("tileup/figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})

fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.75), gridspec_kw={"wspace": 0.18})
rng = np.random.default_rng(3)

for ax, title, session_mode in zip(axes, ["Random image split", "Group-held-out split"], [False, True]):
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
    for s in range(6):
        # session label
        ax.text(s + 0.5, 2.72, f"exp{s+1}", ha="center", va="center", fontsize=7.5)
        # 8 small crops per session
        for r in range(8):
            row, col = r // 4, r % 4
            x = s + 0.08 + col * 0.22
            y = 1.78 - row * 0.48
            if session_mode:
                test = (s == 5)
            else:
                test = rng.random() < 0.35
            color = "#e0a33a" if test else "#2457c5"
            ax.add_patch(Rectangle((x, y), 0.18, 0.36, facecolor=color, edgecolor="white", linewidth=0.6))
    # bracket / label
    if session_mode:
        ax.annotate("", xy=(5.05, 0.55), xytext=(5.95, 0.55),
                    arrowprops=dict(arrowstyle="-", color="#b94a44", lw=2))
        ax.text(5.5, 0.32, "unseen", ha="center", va="top", fontsize=7.5, color="#b94a44")
    else:
        ax.text(3.0, 0.30, "same group appears on both sides", ha="center", va="top",
                fontsize=7.2, color="#b94a44")

# one shared legend
handles = [Rectangle((0,0),1,1, facecolor="#2457c5", edgecolor="white"),
           Rectangle((0,0),1,1, facecolor="#e0a33a", edgecolor="white")]
fig.legend(handles, ["train", "test"], loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_split_protocol.png", dpi=240, bbox_inches="tight", facecolor="white")
print("saved", OUT / "fig_split_protocol.png")
