# Split manifests

These manifests define the exact folds used in the paper. Paths are relative to the `data/` directory and contain no image bytes.

## Files

### `mtd_images.csv`

One row per MTD image.

| Column | Meaning |
|---|---|
| `image` | relative path to the `.jpg` image |
| `mask` | relative path to the `.png` mask |
| `class` | `MT_Blowhole`, `MT_Break`, `MT_Crack`, `MT_Fray`, `MT_Free`, `MT_Uneven` |
| `group` | `expN` filename prefix, or `exp0` for the three free images |
| `filename` | original filename |

Counts: 1,344 images; `exp0` has 3 images and is used only for training.

### `mtd_logo_folds.json`

Six leave-one-filename-group-out folds. For each fold:

- `test_group`: held-out `expN`;
- `val_group`: next `expN` in cyclic order;
- `train_groups`: all remaining groups plus `exp0`;
- `train`, `val`, `test`: lists of `{image, mask, class}` records.

The test and validation groups never overlap. A group never appears in two splits.

### `mtd_design_exp6.json`

The design-selection split used for Table 3 and Table 6:

- test: `exp6`;
- validation: `exp3`;
- training: `exp0`, `exp1`, `exp2`, `exp4`, `exp5`.

The target set and rank were chosen on this split. It is not an unseen-test estimate.

### `neu_seg_split.csv`

One row per NEU-Seg image.

| Column | Meaning |
|---|---|
| `image` | relative path to the `.jpg` image |
| `mask` | relative path to the `.png` mask |
| `split` | `train`, `val`, or `test` |
| `filename` | original filename |

Training/validation come from the official training folder; `test` is the official test folder. The 10% validation split uses `numpy.random.RandomState(0)` and preserves the code's permutation rule, giving 3,267 train / 363 validation / 840 test.

## Regenerate

From the repository root, with the datasets in place:

```bash
python scripts/make_manifests.py
```

This overwrites the four files above. It does not modify the datasets.

## Provenance caveat

`expN` is an undocumented filename prefix. The public MTD README does not state that it is an acquisition session, a production batch, or a strip identifier. The manifests therefore define a **filename-group split**, not a verified cross-session split.
