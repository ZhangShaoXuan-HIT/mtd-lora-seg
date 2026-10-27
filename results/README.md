# Results

Seed-level JSON files used in the paper. Each file contains one entry per run, keyed by fold/mode/seed. The `iou` field is foreground defect IoU; `dice` is foreground Dice.

## MTD LOGO

| File | Contents | Paper table |
|---|---|---|
| `mtd_logo_attention_3seed.json` | Frozen and attention-only LoRA, six folds × three seeds | Table 1 |
| `mtd_logo_mlp_3seed.json` | qkvo+MLP, six folds × three seeds; keys use `lora` but trainable count is 542,722 | Table 1 |
| `mtd_logo_full_3seed.json` | Full fine-tuning, six folds × three seeds | Table 1 |
| `mtd_logo_adapter_3seed.json` | Bottleneck adapter, six folds × three seeds | Table 1 |

The fold rule is in `splits/mtd_logo_folds.json`: test `expN`, validation next `expN`, remaining groups plus `exp0` for training.

## MTD design group

| File | Contents | Paper table |
|---|---|---|
| `mtd_design_frozen_attention_3seed.json` | Frozen and attention-only LoRA on the exp6 design split | Table 3 |
| `mtd_design_full_3seed.json` | Full fine-tuning on the design split | Table 3 |
| `mtd_design_adapter_3seed.json` | Bottleneck adapter on the design split | Table 3 |
| `mtd_design_bitfit_3seed.json` | BitFit on the design split | Table 3 |
| `mtd_design_rank_target_3seed.json` | Rank and target probes (r4/r8/r16 qv, r8/r16 qkvo, r16 qkvo+MLP) | Table 6 |
| `mtd_design_mlp_3seed.json` | qv+MLP and qkvo+MLP probes; `r8_qkvo_mlp` is the Table 3 qkvo+MLP row | Table 3, Table 6 |
| `mtd_per_class_iou.json` | Per-class IoU for Frozen and attention-only LoRA on exp6, seed 0 | Table 4 |

The design split is exp6 test, exp3 validation, and exp0/1/2/4/5 training. It is a design-selection split, not an unseen test.

## NEU-Seg

| File | Contents | Paper table |
|---|---|---|
| `neu_seg_frozen_attention_full_3seed.json` | Frozen, attention-only LoRA, Full, three seeds | Table 2 |
| `neu_seg_mlp_3seed.json` | qkvo+MLP, three seeds; keys use `lora` | Table 2 |
| `neu_seg_adapter_3seed.json` | Bottleneck adapter, three seeds | Table 2 |

NEU-Seg uses the official train/test folders and a 10% validation split; exact rows are in `splits/neu_seg_split.csv`.

## Failed baselines

`failed_baselines/` contains the DeepLabV3 and scratch U-Net runs that did not converge under the shared budget. They are kept for transparency and are not part of the main comparison.

## Aggregation

The paper reports the mean over folds and seeds. For a quick check:

```python
import json, statistics as st

d = json.load(open("results/mtd_logo_mlp_3seed.json"))
vals = [v["iou"] for v in d.values()]
print(len(vals), st.mean(vals))
```

For the paper's group-level mean, average the three seeds within each fold first, then average the six fold means. With balanced seed counts the two orders give the same result.
