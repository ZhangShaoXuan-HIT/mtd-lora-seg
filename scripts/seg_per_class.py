import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtd_data import load_items  # noqa: E402
from mtd_seg_lora import build_model, make_group_splits, SIZE, MEAN, STD  # noqa: E402

CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]


def load_model(kind, ckpt):
    if kind == "frozen":
        model = build_model("frozen")[0]
    else:
        model = build_model("lora", rank=8, target_set="qkvo")[0]
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    return model.cuda().eval()


@torch.no_grad()
def run(model, items):
    out = {}
    for c in range(6):
        inter = union = 0.0
        for it in items:
            if it["label"] != c:
                continue
            img = Image.open(it["path"]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
            mask = Image.open(it["path"].with_suffix(".png")).convert("L").resize((SIZE, SIZE), Image.NEAREST)
            gt = (np.array(mask) > 127).astype(np.float32)
            x = T.functional.normalize(T.functional.to_tensor(img), MEAN, STD).unsqueeze(0).cuda()
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits[:, 1]) > 0.5).float().squeeze().cpu().numpy()
            inter += float((pred * gt).sum())
            union += float(pred.sum() + gt.sum() - (pred * gt).sum())
        out[CLASSES[c]] = round(inter / max(union, 1e-6), 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default=str(ROOT / "results" / "reproduced" / "ckpt_group"))
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "reproduced" / "mtd_per_class_iou.json"))
    args = parser.parse_args()

    items = load_items()
    splits = make_group_splits(items, 0)
    test = splits["test"]
    ckpt = Path(args.ckpt_dir)
    frozen = load_model("frozen", ckpt / "frozen_s0.pt")
    lora = load_model("lora", ckpt / "lora_s0.pt")
    res = {"frozen": run(frozen, test), "lora": run(lora, test)}
    print(json.dumps(res, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
