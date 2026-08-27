# Solar Filament Segmentation Challenge 2026 - VisionX Solution

Automated, high-precision segmentation of solar filaments from GONG H-Alpha observations ($2048 \times 2048$ grayscale images) evaluated under the **Panoptic Quality (PQ)** metric.

---

## 🌟 Overview & Architecture

This repository contains the complete end-to-end pipeline designed for the **Solar Filament Segmentation Challenge 2026** (IEEE BigData Cup / Kaggle).

The architecture uses a modular **Two-Stage Detection & Refinement Cascade**:
1. **Stage 1 (Region Candidate Proposal)**: Detect potential solar filament regions from full-resolution H-Alpha observations.
2. **Stage 2 (Cropped Fine-Structure Refinement)**: Extract candidate crops, preprocess with Adaptive Contrast Enhancement (CLAHE + multi-channel input), and refine fine-scale filament morphology (barbs & spines) using deep convolutional backbones.
3. **Non-Overlapping Mask Suppression & COCO RLE**: Greedily resolve pixel collisions across overlapping candidates and convert binary masks into standard COCO RLE format (`pycocotools`).

---

## 📁 Repository Structure

```
VisionX/
├── README.md               # Main project overview & documentation
├── SETUP.md                # Comprehensive Kaggle & local environment setup guide
├── requirements.txt        # Python package dependencies & versions
├── antigravity_logs.md     # Progress tracking & prompt execution log
├── .gitignore              # Ignored dataset, checkpoint, and build files
├── src/
│   ├── __init__.py
│   ├── utils.py            # COCO RLE utilities & stem splitting
│   ├── metrics.py          # Panoptic Quality (PQ) & Dice evaluation
│   ├── dataset.py          # Dataset loader & Albumentations pipeline
│   ├── models.py           # Refinement neural network models
│   ├── postprocess.py      # Non-overlapping mask suppression & thresholding
│   └── pipeline.py         # End-to-end cascade runner
├── train.py                # Model training entrypoint
└── infer.py                # Test set inference & Kaggle submission generator
```

---

## 📊 Evaluation Metric

Predictions are scored using **Panoptic Quality ($PQ$)**:

$$\text{PQ} = \text{SQ} \times \text{RQ} = \frac{\sum_{(y, \hat{y}) \in \text{TP}} \text{IoU}(y, \hat{y})}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

- **$\text{SQ}$ (Segmentation Quality)**: Average IoU of matched ground-truth and predicted masks ($IoU > 0.5$).
- **$\text{RQ}$ (Recognition Quality)**: $F_1$-score of instance detection.

---

## 🚀 Getting Started

Refer to [SETUP.md](SETUP.md) for detailed instructions on running locally or on Kaggle.


---

## 📄 License & Attribution

- **License**: CC BY-NC 4.0 / MIT
- **Data Source**: GONG H-Alpha observations, National Solar Observatory (NSO / AURA / NSF).
