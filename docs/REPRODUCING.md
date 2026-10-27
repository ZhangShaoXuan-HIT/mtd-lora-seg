# Reproducing the paper results

This document gives the exact commands behind the main tables. Run everything from the repository root.

## 1. Data

Place the two datasets under `data/` or set `MTD_ROOT` and `NEU_SEG_ROOT`. See `data/README.md` for the expected layout and original download links.

Check the expected MTD count:

```bash
python - <<'PY'
from scripts.mtd_data import load_items
items = load_items()
print(len(items))  # 1344
print(sorted({i["filename"].split("_")[0] for i in items}))
PY
```

Check the NEU-Seg count:

```bash
python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ.get("NEU_SEG_ROOT", "data/NEU_Seg-main"))
for split in ["training", "test"]:
    print(split, len(list((root / "images" / split).glob("*.jpg"))))
PY
```

## 2. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your driver. The paper used an RTX 4060 Laptop with 8 GB VRAM; smaller GPUs may need a smaller batch size.

The model `nvidia/segformer-b0-finetuned-ade-512-512` is fetched from Hugging Face on first use. If you need a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 3. MTD LOGO

The fold definition is `splits/mtd_logo_folds.json`: test `expN`, validation next `expN`, training the remaining groups plus `exp0`.

```bash
mkdir -p results/reproduced

python scripts/mtd_loso.py \
  --modes frozen lora \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_attention_3seed.json

python scripts/mtd_loso.py \
  --modes lora \
  --targets qkvo_mlp \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_mlp_3seed.json

python scripts/mtd_loso.py \
  --modes full \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 --lr-full 6e-5 \
  --out results/reproduced/mtd_logo_full_3seed.json

python scripts/mtd_loso.py \
  --modes adapter \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_adapter_3seed.json
```

Expected approximate wall-clock time on the 8 GB laptop GPU: several hours for all four commands. The paper's saved runs took roughly 3–8 minutes per fold/seed depending on the mode.

## 4. NEU-Seg

```bash
python scripts/neu_seg_lora.py \
  --modes frozen lora full \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_frozen_attention_full_3seed.json

python scripts/neu_seg_lora.py \
  --modes lora \
  --targets qkvo_mlp \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_mlp_3seed.json

python scripts/neu_seg_lora.py \
  --modes adapter \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_adapter_3seed.json
```

The 10% validation split is `RandomState(0)` over the official training set. It is recorded in `splits/neu_seg_split.csv`.

## 5. Design group (Table 3 and Table 6)

```bash
python scripts/mtd_seg_lora.py \
  --group-split --split-seed 0 \
  --modes frozen bitfit lora adapter full \
  --targets qkvo \
  --seeds 0 1 2 --epochs 12 --batch 2 \
  --out results/reproduced/mtd_design_methods_3seed.json

python scripts/mtd_seg_ablation_group.py \
  --seeds 0 1 2 --epochs 12 \
  --out results/reproduced/mtd_design_rank_target_3seed.json

python scripts/mtd_seg_ablation_group2.py \
  --seeds 0 1 2 --epochs 12 \
  --out results/reproduced/mtd_design_mlp_3seed.json
```

`--split-seed 0` gives test `exp6` and validation `exp3`, matching `splits/mtd_design_exp6.json`.

## 6. Table aggregation

Within each fold, average the three seeds; then average the six folds. The result files are keyed by `fold_mode_sN`.

Example for qkvo+MLP:

```python
import json
import statistics as st

d = json.load(open("results/mtd_logo_mlp_3seed.json"))
folds = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
means = []
for f in folds:
    vals = [v["iou"] for k, v in d.items() if k.startswith(f + "_")]
    means.append(st.mean(vals))
print(means)
print(st.mean(means))
```

The paper's MTD qkvo+MLP mean is 0.7794; Full is 0.7724.

## 7. Figures

The five paper figures are included in `figures/`. The qualitative and architecture figures depend on the saved checkpoints and the original figure scripts; the result JSON files are sufficient to regenerate the quantitative bar and parameter plots.

## 8. Optional checkpoints and per-class results

The main tables do not require saved checkpoints. To reproduce the per-class table or qualitative panels, train with `--save-dir`:

```bash
python scripts/mtd_seg_lora.py \
  --group-split --split-seed 0 \
  --modes frozen lora \
  --targets qkvo --seeds 0 \
  --epochs 12 --batch 2 \
  --save-dir results/reproduced/ckpt_group \
  --out results/reproduced/design_ckpt.json
```

Then point `seg_per_class.py` at the checkpoint directory. The version in this repository expects `results/reproduced/ckpt_group/frozen_s0.pt` and `lora_s0.pt`; edit the paths at the bottom of the script if your directory differs.

## 9. Common problems

- **Missing dataset**: set `MTD_ROOT` / `NEU_SEG_ROOT`; the default is `data/`.
- **CUDA out of memory**: lower `--batch` to 1, or close other GPU processes. The paper used batch 2 for MTD and NEU-Seg.
- **Hugging Face download blocked**: set `HF_ENDPOINT` to a reachable mirror.
- **Different last decimal**: GPU nondeterminism and library versions can shift the fourth decimal; the paper's claims are at the point level, not the fourth decimal.
- **No `results/reproduced` directory**: create it with `mkdir -p results/reproduced`.
