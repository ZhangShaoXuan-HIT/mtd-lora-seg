# Low-Rank Adaptation for Industrial Defect Segmentation

Code and result artifacts for the paper:

> Where to Put Low-Rank Capacity? A Grouped-Split Study of LoRA for Industrial Defect Segmentation

This repository contains the training, evaluation, split, and figure code for a controlled comparison of where low-rank adaptation is placed in a frozen SegFormer-B0 encoder. It is an empirical study, not a new adapter architecture.

## What is included

- SegFormer-B0 + LoRA training and evaluation on Magnetic Tile Defect (MTD) and NEU-Seg.
- Frozen, BitFit, attention-only LoRA, LoRA + MLP, bottleneck adapter, and full fine-tuning.
- Leave-one-filename-group-out (LOGO) evaluation on MTD.
- Three seeds for the main comparisons.
- Static split manifests so the exact folds can be checked without downloading the data.
- Seed-level result JSON files used in the paper.
- The five paper figures.

## What is not included

- Raw images and masks. Both datasets are public; see [`data/README.md`](data/README.md).
- Model checkpoints. They are large and are not needed to inspect the results.
- The manuscript source. This repository is for reproducibility, not for redistributing the paper.

## Main results

Foreground defect IoU, mean over seeds. MTD uses six filename groups; NEU-Seg uses the official train/test folders with a 10% validation split.

| Dataset | Frozen | Attention-only LoRA | qkvo + MLP | Full |
|---|---:|---:|---:|---:|
| MTD LOGO | 0.482 | 0.723 | **0.779** | 0.772 |
| NEU-Seg | 0.483 | 0.684 | **0.699** | 0.715 |

MTD: qkvo+MLP minus Full = **+0.70 points**, 95% CI **[-2.0, +3.4]**, p=0.54. This is not an equivalence result.

NEU-Seg: qkvo+MLP minus Full = **-1.56 points**, p=0.013.

A five-group sensitivity analysis excluding the design-selected exp6 group gives MTD 0.768 vs 0.759 for Full (+0.89, p=0.52).

## Repository layout

```text
.
├── data/                      # download instructions only; no images
├── splits/                    # static MTD/NEU split manifests
├── scripts/
│   ├── mtd_data.py            # MTD paths and split helpers
│   ├── mtd_seg_lora.py        # training/evaluation for all adaptation modes
│   ├── mtd_loso.py            # leave-one-filename-group-out runner
│   ├── neu_seg_lora.py        # NEU-Seg runner
│   ├── mtd_seg_ablation_group.py
│   ├── mtd_seg_ablation_group2.py
│   ├── make_manifests.py      # regenerate splits/
│   ├── seg_per_class.py       # per-class IoU (needs checkpoints)
│   ├── seg_visualize.py       # qualitative panels (needs checkpoints)
│   └── ...
├── results/                   # seed-level JSON used in the paper
├── figures/                   # paper figures
└── docs/
    └── REPRODUCING.md         # detailed commands and notes
```

## Data setup

Download the two public datasets:

1. **Magnetic Tile Defect**  
   https://github.com/abin24/Magnetic-tile-defect-datasets.

2. **NEU-Seg**  
   https://github.com/DHW-Master/NEU_Seg

Place them as:

```text
data/
├── Magnetic-tile-defect-datasets.-master/
│   ├── MT_Blowhole/Imgs/*.jpg, *.png
│   ├── MT_Break/Imgs/
│   ├── MT_Crack/Imgs/
│   ├── MT_Fray/Imgs/
│   ├── MT_Free/Imgs/
│   └── MT_Uneven/Imgs/
└── NEU_Seg-main/
    ├── images/training/*.jpg
    ├── images/test/*.jpg
    ├── annotations/training/*.png
    └── annotations/test/*.png
```

Alternatively, set:

```bash
export MTD_ROOT=/path/to/Magnetic-tile-defect-datasets.-master
export NEU_SEG_ROOT=/path/to/NEU_Seg-main
```

The code never downloads the datasets automatically. Follow the original dataset licences.

## Environment

Tested with:

- Windows, Python 3.14
- PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126
- transformers 5.20.2, peft 0.21.2
- NVIDIA RTX 4060 Laptop, 8 GB VRAM, 16 GB RAM

A clean environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The Hugging Face model `nvidia/segformer-b0-finetuned-ade-512-512` is downloaded on first use. If you need a mirror, set `HF_ENDPOINT` yourself; the code does not hard-code one.

## Quick start: MTD LOGO

Run from the repository root.

```bash
# 1. Attention-only LoRA and Frozen (Table 1)
python scripts/mtd_loso.py \
  --modes frozen lora \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_attention_3seed.json

# 2. qkvo + MLP (Table 1)
python scripts/mtd_loso.py \
  --modes lora \
  --targets qkvo_mlp \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_mlp_3seed.json

# 3. Full fine-tuning (Table 1)
python scripts/mtd_loso.py \
  --modes full \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 --lr-full 6e-5 \
  --out results/reproduced/mtd_logo_full_3seed.json

# 4. Bottleneck adapter (Table 1)
python scripts/mtd_loso.py \
  --modes adapter \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 12 --batch 2 \
  --out results/reproduced/mtd_logo_adapter_3seed.json
```

`mtd_loso.py` uses the fold rule in `splits/mtd_logo_folds.json`: one `expN` group is test, the next is validation, and the rest plus `exp0` are training.

## Quick start: NEU-Seg

```bash
# Frozen, attention-only LoRA, Full (Table 2)
python scripts/neu_seg_lora.py \
  --modes frozen lora full \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_frozen_attention_full_3seed.json

# qkvo + MLP (Table 2)
python scripts/neu_seg_lora.py \
  --modes lora \
  --targets qkvo_mlp \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_mlp_3seed.json

# Adapter (Table 2)
python scripts/neu_seg_lora.py \
  --modes adapter \
  --targets qkvo \
  --seeds 0 1 2 \
  --epochs 8 --batch 2 --size 256 \
  --out results/reproduced/neu_seg_adapter_3seed.json
```

NEU-Seg uses the official train/test folders; 10% of the official training set is held out for validation with `RandomState(0)`. The resulting split is in `splits/neu_seg_split.csv`.

## Design-group runs (Table 3 and Table 6)

The design group is exp6 test, exp3 validation, and exp0/1/2/4/5 training. It was used to choose the target set and rank and is **not** an unseen-test estimate.

```bash
# Frozen, BitFit, LoRA, Adapter, Full on the design group
python scripts/mtd_seg_lora.py \
  --group-split --split-seed 0 \
  --modes frozen bitfit lora adapter full \
  --targets qkvo \
  --seeds 0 1 2 --epochs 12 --batch 2 \
  --out results/reproduced/mtd_design_methods_3seed.json

# qkvo + MLP and the rank/target probes
python scripts/mtd_seg_ablation_group.py \
  --seeds 0 1 2 --epochs 12 \
  --out results/reproduced/mtd_design_rank_target_3seed.json

python scripts/mtd_seg_ablation_group2.py \
  --seeds 0 1 2 --epochs 12 \
  --out results/reproduced/mtd_design_mlp_3seed.json
```

The qkvo+MLP design-group result is the `r8_qkvo_mlp` entry in `mtd_design_mlp_3seed.json`.

## Regenerating the split manifests

```bash
python scripts/make_manifests.py
```

This writes:

- `splits/mtd_images.csv` — one row per MTD image/mask pair.
- `splits/mtd_logo_folds.json` — six LOGO folds.
- `splits/mtd_design_exp6.json` — exp6 design split.
- `splits/neu_seg_split.csv` — official NEU train/test plus the 10% validation split.

The manifests contain relative paths only. No images are copied.

## Metric

The reported IoU is **foreground defect IoU**: class 1 only, with intersections and unions pooled over the test set. It is not a two-class mean IoU. Dice is computed on the same foreground definition.

## What the code does not claim

- It does not claim that `expN` is an acquisition session. The public MTD README does not define `expN`; the code and paper treat it as a filename group.
- It does not claim that qkvo+MLP matches Full. The MTD interval includes zero and a two-point deficit; the result is a non-significant difference, not equivalence.
- It does not claim a new LoRA variant. The contribution is the comparison and the grouped-split evaluation.
- It does not include a tuned Full baseline sweep. Full uses a fixed budget; see the paper's limitations.

## Reproducibility notes

- The main numbers are seed-level means over three seeds. GPU nondeterminism can move the last decimal.
- Full fine-tuning is sensitive to learning rate; the paper uses `6e-5`, 12 epochs on MTD and 8 on NEU-Seg.
- Checkpoints are not included. `seg_per_class.py` and `seg_visualize.py` expect checkpoints produced with `--save-dir`; they are optional and not needed for the main tables.
- The failed DeepLabV3 and scratch U-Net runs are kept under `results/failed_baselines/` for transparency, not as main comparisons.

## Citation

If you use this code, please cite the paper and the two datasets. See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT. Datasets: follow the licences of the original MTD and NEU-Seg repositories. No dataset images are redistributed here.
