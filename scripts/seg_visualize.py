import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtd_data import load_items, make_splits  # noqa: E402
from mtd_seg_lora import build_model, make_group_splits, SIZE, MEAN, STD  # noqa: E402

CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]


def boundary(mask):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    er = -F.max_pool2d(-m, 3, 1, 1)
    return ((m - er).squeeze() > 0).numpy()


def overlay(img, pred, gt):
    out = img.copy()
    red = np.zeros_like(out)
    red[..., 0] = 1.0
    out = np.where(pred[..., None], 0.55 * out + 0.45 * red, out)
    b = boundary(torch.from_numpy(gt.astype(np.float32)))
    out[b] = np.array([0.1, 0.9, 0.2])
    return np.clip(out, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--group-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(ROOT / "figures" / "fig_seg_visual.png"))
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    items = load_items()
    splits = make_group_splits(items, args.split_seed) if args.group_split else make_splits(items)
    test = [it for it in splits["test"] if it["label"] != 4]  # exclude MT_Free
    # choose the largest-defect example from up to 4 defect classes
    picked = []
    for c in range(5):
        cand = []
        for it in test:
            if it["label"] != c:
                continue
            m = np.array(Image.open(it["path"].with_suffix(".png")).convert("L").resize((SIZE, SIZE), Image.NEAREST))
            area = int((m > 127).sum())
            if area > 20:
                cand.append((area, it))
        cand.sort(key=lambda x: -x[0])
        if cand:
            picked.append(cand[0][1])
    picked = picked[: args.n]
    print("picked", [(it["filename"], CLASSES[it["label"]]) for it in picked])

    frozen = build_model("frozen")[0].cuda().eval()
    frozen.load_state_dict(torch.load(Path(args.ckpt_dir) / "frozen_s0.pt", map_location="cuda"))
    lora = build_model("lora", rank=8, target_set="qkvo")[0].cuda().eval()
    lora.load_state_dict(torch.load(Path(args.ckpt_dir) / "lora_s0.pt", map_location="cuda"))

    fig, axes = plt.subplots(len(picked), 4, figsize=(13, 3.2 * len(picked)))
    if len(picked) == 1:
        axes = axes[None, :]
    for row, it in enumerate(picked):
        img = Image.open(it["path"]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        gt_img = Image.open(it["path"].with_suffix(".png")).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        gt = (np.array(gt_img) > 127).astype(np.float32)
        x = T.functional.normalize(T.functional.to_tensor(img), MEAN, STD).unsqueeze(0).cuda()
        with torch.no_grad():
            lf = F.interpolate(frozen(pixel_values=x).logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            ll = F.interpolate(lora(pixel_values=x).logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            pf = torch.sigmoid(lf[:, 1]).squeeze().cpu().numpy()
            pl = torch.sigmoid(ll[:, 1]).squeeze().cpu().numpy()
        pf = (pf > 0.5).astype(np.float32)
        pl = (pl > 0.5).astype(np.float32)
        arr = np.array(img).astype(np.float32) / 255.0

        def iou(p):
            inter = (p * gt).sum()
            return inter / max((p + gt - p * gt).sum(), 1)

        axes[row, 0].imshow(arr)
        axes[row, 0].set_title(f"{CLASSES[it['label']]}  input")
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("GT mask")
        axes[row, 2].imshow(overlay(arr, pf, gt))
        axes[row, 2].set_title(f"Frozen  IoU={iou(pf):.3f}")
        axes[row, 3].imshow(overlay(arr, pl, gt))
        axes[row, 3].set_title(f"LoRA qkvo  IoU={iou(pl):.3f}")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
