import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtd_data import load_items  # noqa: E402
from mtd_seg_lora import run, make_group_splits, OUT  # noqa: E402

CONFIGS = [
    # name, rank, targets
    ("r4_qv", 4, "qv"),
    ("r8_qv", 8, "qv"),
    ("r16_qv", 16, "qv"),
    ("r8_qkvo", 8, "qkvo"),
    ("r16_qkvo", 16, "qkvo"),
    ("r16_qkvo_mlp", 16, "qkvo_mlp"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out", type=str, default=str(OUT / "seg_ablation_group_3seed.json"))
    args = parser.parse_args()

    items = load_items()
    splits = make_group_splits(items, 0)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    results = {}
    t0 = time.time()
    for name, rank, targets in CONFIGS:
        for seed in args.seeds:
            key = f"{name}_s{seed}"
            print(f"RUN {key}", flush=True)
            r = run("lora", tr, va, te, seed=seed, epochs=args.epochs, batch=2, rank=rank,
                    target_set=targets, patience_limit=99)
            r.update({"name": name, "rank": rank, "targets": targets, "seed": seed})
            results[key] = r
            print(f"RESULT {key} iou={r['iou']:.4f} dice={r['dice']:.4f} "
                  f"trainable={r['trainable']}/{r['total']} sec={r['train_sec']:.0f}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            torch.cuda.empty_cache()
    print(f"ALL_DONE {time.time()-t0:.0f}s", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
