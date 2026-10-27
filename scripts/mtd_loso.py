import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtd_seg_lora as seg  # noqa: E402
from mtd_data import load_items  # noqa: E402

OUT = ROOT / "out" / "mtd_loso"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]


def group_of(filename):
    return filename.split("_")[0]


def make_fold(items, test_group):
    val_group = GROUPS[(GROUPS.index(test_group) + 1) % len(GROUPS)]
    splits = {"train": [], "val": [], "test": []}
    for it in items:
        g = group_of(it["filename"])
        if g == test_group:
            splits["test"].append(it)
        elif g == val_group:
            splits["val"].append(it)
        else:
            splits["train"].append(it)
    return splits, val_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["frozen", "lora"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--targets", type=str, default="qkvo")
    parser.add_argument("--out", type=str, default=str(OUT / "loso.json"))
    args = parser.parse_args()

    seg.SIZE = 384
    items = load_items()
    results = {}
    for test_group in GROUPS:
        splits, val_group = make_fold(items, test_group)
        print(f"FOLD test={test_group} val={val_group} train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}", flush=True)
        for seed in args.seeds:
            for mode in args.modes:
                key = f"{test_group}_{mode}_s{seed}"
                print("RUN", key, flush=True)
                t0 = time.time()
                r = seg.run(mode, splits["train"], splits["val"], splits["test"],
                            seed=seed, epochs=args.epochs, batch=args.batch,
                            rank=8, target_set=args.targets, patience_limit=99)
                r.update({"fold": test_group, "val_group": val_group, "mode": mode, "seed": seed})
                results[key] = r
                print(f"RESULT {key} iou={r['iou']:.4f} dice={r['dice']:.4f} sec={time.time()-t0:.0f}", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
                torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
