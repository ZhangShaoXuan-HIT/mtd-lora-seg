"""Generate the static MTD and NEU-Seg split manifests used in the paper."""

import csv
import json
import os
from pathlib import Path

import numpy as np

from mtd_data import DATA, group_of, load_items

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
GROUPS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
NEU = Path(os.environ.get("NEU_SEG_ROOT", ROOT / "data" / "NEU_Seg-main"))


def rel(p: Path, base: Path) -> str:
    return p.relative_to(base).as_posix()


def mtd_manifests():
    items = load_items()
    rows = []
    for it in items:
        rows.append({
            "image": rel(it["path"], DATA.parent),
            "mask": rel(it["path"].with_suffix(".png"), DATA.parent),
            "class": it["class"],
            "group": group_of(it["filename"]),
            "filename": it["filename"],
        })
    rows.sort(key=lambda r: (r["group"], r["class"], r["filename"]))

    with (SPLITS / "mtd_images.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "class", "group", "filename"])
        w.writeheader()
        w.writerows(rows)

    folds = {}
    for test_group in GROUPS:
        val_group = GROUPS[(GROUPS.index(test_group) + 1) % len(GROUPS)]
        train_groups = ["exp0"] + [g for g in GROUPS if g not in (test_group, val_group)]
        fold = {"test_group": test_group, "val_group": val_group,
                "train_groups": sorted(train_groups), "train": [], "val": [], "test": []}
        for it in items:
            g = group_of(it["filename"])
            rec = {"image": rel(it["path"], DATA.parent),
                   "mask": rel(it["path"].with_suffix(".png"), DATA.parent),
                   "class": it["class"]}
            if g == test_group:
                fold["test"].append(rec)
            elif g == val_group:
                fold["val"].append(rec)
            else:
                fold["train"].append(rec)
        folds[test_group] = fold

    payload = {
        "protocol": "leave-one-filename-group-out",
        "groups": GROUPS,
        "note": "Group is the expN filename prefix. It is not documented as an acquisition session.",
        "folds": folds,
    }
    (SPLITS / "mtd_logo_folds.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    design = {
        "protocol": "design-selection group split (exp6 test, exp3 validation)",
        "test_group": "exp6",
        "val_group": "exp3",
        "train_groups": ["exp0", "exp1", "exp2", "exp4", "exp5"],
        "note": "Used to choose the target set and rank; not an unseen-test estimate.",
        "train": [], "val": [], "test": [],
    }
    for it in items:
        g = group_of(it["filename"])
        rec = {"image": rel(it["path"], DATA.parent),
               "mask": rel(it["path"].with_suffix(".png"), DATA.parent),
               "class": it["class"]}
        if g == design["test_group"]:
            design["test"].append(rec)
        elif g == design["val_group"]:
            design["val"].append(rec)
        else:
            design["train"].append(rec)
    (SPLITS / "mtd_design_exp6.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8")

    print("MTD images", len(rows), "folds", len(folds),
          "design train/val/test", len(design["train"]), len(design["val"]), len(design["test"]))


def neu_manifests():
    rows = []
    train = []
    for split in ["training", "test"]:
        for img in sorted((NEU / "images" / split).glob("*.jpg")):
            mask = NEU / "annotations" / split / (img.stem + ".png")
            if not mask.exists():
                continue
            rec = {"image": rel(img, NEU.parent), "mask": rel(mask, NEU.parent),
                   "split": split, "filename": img.name}
            rows.append(rec)
            if split == "training":
                train.append(rec)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(train))
    n_val = int(round(len(train) * 0.1))
    val_ids = {train[i]["image"] for i in idx[:n_val]}
    for rec in rows:
        if rec["split"] == "test":
            rec["split"] = "test"
        elif rec["image"] in val_ids:
            rec["split"] = "val"
        else:
            rec["split"] = "train"

    rows.sort(key=lambda r: (r["split"], r["filename"]))
    with (SPLITS / "neu_seg_split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask", "split", "filename"])
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ["train", "val", "test"]}
    print("NEU-Seg", counts)


if __name__ == "__main__":
    SPLITS.mkdir(exist_ok=True)
    mtd_manifests()
    neu_manifests()
