"""Dataset paths and split helpers for the Magnetic Tile Defect (MTD) experiments."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get(
    "MTD_ROOT",
    ROOT / "data" / "Magnetic-tile-defect-datasets.-master",
))
CLASSES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free", "MT_Uneven"]


def load_items():
    """Return one record per MTD image; masks share the image stem with .png."""
    items = []
    for ci, cname in enumerate(CLASSES):
        img_dir = DATA / cname / "Imgs"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Missing {img_dir}. Follow data/README.md to place the MTD dataset."
            )
        for p in sorted(img_dir.glob("*.jpg")):
            items.append({
                "path": p,
                "label": ci,
                "filename": p.name,
                "class": cname,
            })
    return items


def make_splits(items):
    """Class-stratified random split used only as the random-split baseline."""
    by_class = {c: [] for c in range(6)}
    for it in items:
        by_class[it["label"]].append(it)
    splits = {"train": [], "val": [], "test": []}
    for c, arr in by_class.items():
        arr = sorted(arr, key=lambda x: x["filename"])
        n = len(arr)
        n_test = max(8, int(round(n * 0.2)))
        n_val = max(4, int(round(n * 0.1)))
        splits["test"] += arr[:n_test]
        splits["val"] += arr[n_test:n_test + n_val]
        splits["train"] += arr[n_test + n_val:]
    return splits


def group_of(filename):
    """Return the expN filename prefix, e.g. exp3 from exp3_num_123.jpg."""
    return filename.split("_")[0]
