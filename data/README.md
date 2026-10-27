# Data

No images or masks are included in this repository. Download the two public datasets from their original sources.

## Magnetic Tile Defect (MTD)

- Repository: https://github.com/abin24/Magnetic-tile-defect-datasets.
- Paper: Huang Y, Qiu C, Yuan K (2020) Surface defect saliency of magnetic tile. The Visual Computer 36:85–96. https://doi.org/10.1007/s00371-018-1588-5
- Size: 1,344 grayscale images across six folders.
- Masks: one `.png` per `.jpg`, same stem.
- Task here: binary foreground defect segmentation.

Expected layout:

```text
data/Magnetic-tile-defect-datasets.-master/
├── MT_Blowhole/Imgs/*.jpg, *.png
├── MT_Break/Imgs/
├── MT_Crack/Imgs/
├── MT_Fray/Imgs/
├── MT_Free/Imgs/
└── MT_Uneven/Imgs/
```

`expN` is a filename prefix. The public README does not define it as an acquisition session. This repository treats it as a filename group and does not claim strip-level or session-level generalization.

## NEU-Seg

- Repository: https://github.com/DHW-Master/NEU_Seg
- Size: 4,470 images with binary masks.
- Official split: 3,630 training and 840 test images.
- This repository holds out 10% of the official training set for validation with `RandomState(0)`.

Expected layout:

```text
data/NEU_Seg-main/
├── images/training/*.jpg
├── images/test/*.jpg
├── annotations/training/*.png
└── annotations/test/*.png
```

## Environment variables

The code looks for the datasets under `data/` by default. To use another location:

```bash
export MTD_ROOT=/path/to/Magnetic-tile-defect-datasets.-master
export NEU_SEG_ROOT=/path/to/NEU_Seg-main
```

On Windows PowerShell:

```powershell
$env:MTD_ROOT = "C:\path\to\Magnetic-tile-defect-datasets.-master"
$env:NEU_SEG_ROOT = "C:\path\to\NEU_Seg-main"
```

## Licences

The datasets are not covered by this repository's MIT licence. Check the original repositories and the dataset paper before redistributing images, masks, or derived manifests.
