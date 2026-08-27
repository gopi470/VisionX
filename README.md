# Solar Filament Segmentation Challenge 2026 - VisionX v2 Solution

Automated, high-precision segmentation of solar filaments from GONG H-Alpha observations ($2048 \times 2048$ grayscale images) evaluated under the **Panoptic Quality (PQ)** metric.

---

## 🌟 Overview & Architecture (v2 Upgrade)

This repository contains the complete end-to-end v2 pipeline designed for the **Solar Filament Segmentation Challenge 2026** (IEEE BigData Cup / Kaggle).

The v2 architecture features a **Config-Driven 4-Stage Cascade Pipeline**:
1. **Stage 1 (Dual-Path Candidate Proposal)**: Merges Path A (YOLO11s at $1280 \times 1280$) and Path B (Always-on CLAHE connected-components) via IoU-NMS to ensure high recall for both large structures and subtle filament barbs.
2. **Stage 2 (Fixed-GSD 4-Channel Crop Extraction)**: Computes adaptive crop sizes ($768\text{--}1280\text{px}$) keeping pixels-per-filament-width constant. Stacks a 4-channel input tensor: `[Raw Grayscale, CLAHE (clipLimit=2.0), Binary Seed, Normalized Distance Transform]`.
3. **Stage 3 (Deep Refinement & Composite Loss)**: Refines candidates using `smp.Unet` (default backbone `convnext_small` / `resnet34`) trained with `CompositeLoss` ($0.30 \text{ BCE} + 0.40 \text{ Dice} + 0.30 \text{ BoundaryLoss}$), EMA, SWA, and Hard-Negative Mining.
4. **Stage 4 (Budget-Aware TTA, Overlap Suppression & Skeleton Gap Repair)**: Runs 6-pass budget-aware TTA (H/V flips $\times$ multi-scale), Otsu adaptive thresholding, greedy pixel collision resolution, and skeleton-based connectivity gap closing before COCO RLE encoding.

---

## 📁 Repository Structure

```
VisionX/
├── README.md               # Main project overview & documentation (v2 updated)
├── READMEext.md            # Ultimate beginner-to-expert scientific explanatory guide
├── SETUP.md                # Comprehensive Kaggle & local execution guide (v2 cell updated)
├── WORKFLOW.md             # Complete technical pipeline flow diagram & mathematical spec
├── requirements.txt        # Package dependencies (scipy, scikit-image, albumentations, etc.)
├── antigravity_logs.md     # Progress tracking & prompt execution log
├── test_suite.py           # 8-test validation suite covering all v2 functionality
├── train.py                # v2 Training loop (EMA, SWA, Composite Loss, Hard-Negative Mining)
├── infer.py                # v2 Inference runner & Kaggle submission generator with Budget Guard
└── src/
    ├── __init__.py         # Package exports
    ├── config.py           # Central V2Config dataclass (all features config-gated)
    ├── losses.py           # BoundaryLoss, TverskyLoss, and CompositeLoss factory
    ├── utils.py            # COCO RLE utilities & timestamp GroupKFold splitting
    ├── metrics.py          # PQ evaluation & Stratified PQ by filament width bucket
    ├── dataset.py          # Fixed-GSD 4-channel loader + Albumentations pipeline
    ├── models.py           # Refinement neural networks (Unet + mean-init 4th channel)
    ├── postprocess.py      # Greedily non-overlapping mask suppression & skeleton gap repair
    └── pipeline.py         # End-to-end cascade runner & EnsemblePipeline
```

---

## 📊 Evaluation Metric & Stratified PQ

Predictions are evaluated using **Panoptic Quality ($PQ$)**:

$$\text{PQ} = \text{SQ} \times \text{RQ} = \frac{\sum_{(y, \hat{y}) \in \text{TP}} \text{IoU}(y, \hat{y})}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

- **$\text{SQ}$ (Segmentation Quality)**: Average IoU of matched ground-truth and predicted masks ($\text{IoU} > 0.50$).
- **$\text{RQ}$ (Recognition Quality)**: $F_1$-score of instance detection.

**v2 Addition**: `src/metrics.py` provides `evaluate_stratified_pq()` which computes PQ broken out by GT filament width buckets:
- **Thin**: $\le 2\text{px}$ (most challenging solar barb tendrils)
- **Medium**: $3\text{--}6\text{px}$
- **Thick**: $> 6\text{px}$

---

## 🚀 Quick Kaggle Run Command

In your Kaggle Notebook (with GPU T4 enabled):

```bash
!pip install -q scipy scikit-image segmentation-models-pytorch albumentations ultralytics
!if [ ! -d "/kaggle/working/VisionX" ]; then git clone https://github.com/gopi470/VisionX.git /kaggle/working/VisionX; fi
%cd /kaggle/working/VisionX
!git pull origin main
!python infer.py --output_dir /kaggle/working
```

---

## 📄 License & Attribution

- **License**: CC BY-NC 4.0 / MIT
- **Data Source**: GONG H-Alpha observations, National Solar Observatory (NSO / AURA / NSF).
