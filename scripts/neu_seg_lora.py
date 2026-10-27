import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtd_seg_lora as seg  # noqa: E402

DATA = Path(os.environ.get("NEU_SEG_ROOT", ROOT / "data" / "NEU_Seg-main"))
OUT = ROOT / "out" / "neu_seg"
OUT.mkdir(parents=True, exist_ok=True)


def load_items():
    items = []
    for split in ["training", "test"]:
        for img in sorted((DATA / "images" / split).glob("*.jpg")):
            mask = DATA / "annotations" / split / (img.stem + ".png")
            if mask.exists():
                items.append({"path": img, "mask_path": mask, "label": 0, "filename": img.name, "split": split})
    return items


def split_train(items, val_ratio=0.1, seed=0):
    train = [it for it in items if it["split"] == "training"]
    test = [it for it in items if it["split"] == "test"]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train))
    n_val = int(round(len(train) * val_ratio))
    val = [train[i] for i in idx[:n_val]]
    tr = [train[i] for i in idx[n_val:]]
    print("train", len(tr), "val", len(val), "test", len(test))
    return tr, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["frozen", "lora", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--targets", type=str, default="qkvo")
    parser.add_argument("--patience", type=int, default=99)
    parser.add_argument("--out", type=str, default=str(OUT / "neu_seg.json"))
    args = parser.parse_args()

    seg.SIZE = args.size
    items = load_items()
    tr, va, te = split_train(items, 0.1, 0)
    results = {}
    for seed in args.seeds:
        for mode in args.modes:
            key = f"{mode}_s{seed}"
            print("RUN", key, flush=True)
            t0 = time.time()
            r = seg.run(mode, tr, va, te, seed=seed, epochs=args.epochs, batch=args.batch,
                        rank=args.rank, target_set=args.targets, patience_limit=args.patience)
            r.update({"dataset": "NEU-Seg", "size": args.size, "epochs": args.epochs})
            results[key] = r
            print(f"RESULT {key} iou={r['iou']:.4f} dice={r['dice']:.4f} acc={r['acc']:.4f} "
                  f"best_ep={r['best_epoch']} sec={time.time()-t0:.0f}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
