# VisionX - Ultimate Beginner to Expert Guide ☀️🛰️

Welcome to the comprehensive, step-by-step explanatory guide for **VisionX v2**! This document is written so that **anyone**—from high school students and curious space enthusiasts to machine learning engineers—can understand **what solar filaments are**, **why segmenting them is critical for protecting Earth**, **how computer vision models solve this challenge**, and **how to run our code**.

---

## 📖 Table of Contents
1. [☀️ What Are Solar Filaments & Why Do They Matter?](#1-what-are-solar-filaments--why-do-they-matter)
2. [🛰️ Space Weather Threats: Protect Our Tech Grid](#2-space-weather-threats-protect-our-tech-grid)
3. [🧩 The Challenge: Why Is This Hard For AI?](#3-the-challenge-why-is-this-hard-for-ai)
4. [💡 The VisionX v2 Solution: Step-by-Step Explanation](#4-the-visionx-v2-solution-step-by-step-explanation)
5. [📊 How Are Predictions Scored? (Panoptic Quality & Width Buckets)](#5-how-are-predictions-scored-panoptic-quality--width-buckets)
6. [📂 Repository Tour & Code Guide](#6-repository-tour--code-guide)
7. [🚀 How You Can Run This Project](#7-how-you-can-run-this-project)

---

## ☀️ 1. What Are Solar Filaments & Why Do They Matter?

### The Sun's Magnetic Atmosphere
 space is filled with super-heated plasma ($1,000,000\text{ °C}$) driven by twisting magnetic fields from inside the Sun.

### What Is a Filament?
- A **solar filament** is a massive ribbon of dense, cooler plasma ($10,000\text{ °C}$) suspended high in the Sun's atmosphere by magnetic field lines.
- Because filaments are cooler than the hot surface behind them, when we view the Sun using special filters (specifically **H-Alpha light at 656.28 nanometers**), filaments appear as **dark, snake-like threads** across the Sun.
- When a filament rotates to the edge (limb) of the Sun, it sticks out into dark space and looks like a bright, glowing loop—scientists call this a **prominence**.

---

## 🛰️ 2. Space Weather Threats: Protect Our Tech Grid

Why do scientists care about finding every single filament?

> **Key Insight**: Filaments are the physical anchors of **Coronal Mass Ejections (CMEs)**. 

When magnetic field lines holding a filament snap and erupt, they launch **billions of tons of magnetized plasma into space at millions of miles per hour**.

If directed toward Earth:
1. **Power Grid Failures**: Geomagnetic storms induce excess electrical current in power lines, blowing out high-voltage transformers (e.g. the 1989 Quebec blackouts).
2. **Satellite & GPS Disruption**: Solar radiation expands Earth's upper atmosphere, dragging down low-Earth orbit satellites and scrambling GPS signals.
3. **Astronaut Radiation Hazard**: Eruptions release lethal space radiation for space station crews and Moon/Mars missions.

Automated AI segmentation allows space weather observatories to **track filaments 24/7/365** and issue early warnings!

---

## 🧩 3. The Challenge: Why Is This Hard For AI?

1. **Tiny Thread-Like Features ("Barbs")**:
   - Filaments aren't simple smooth ovals. They have a main central **spine** with microscopic thread-like tendrils extending outward called **barbs** ($1\text{--}2\text{ pixels}$ wide at native resolution). Standard AI models downsample images and wipe these barbs out completely.
2. **Sun Noise & Atmospheric Blurring**:
   - Ground telescopes look through Earth's atmosphere, causing shimmering, clouds, and optical noise.
3. **Multi-Annotator Variations**:
   - Human solar physics experts annotated the dataset (MAGFiLO) with slight individual variations.

---

## 💡 4. The VisionX v2 Solution: Step-by-Step Explanation

**VisionX v2** uses an advanced **4-Stage Cascade Pipeline** with dual proposals and 4-channel input:

```
  ┌────────────────────────────────────────────────────────┐
  │           Full H-Alpha Image (2048 x 2048)             │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 1: Dual-Path Proposal                           │
  │  - Path A: YOLO11s detector at conf=0.10, 1280px       │
  │  - Path B: Always-on CLAHE connected components        │
  │  - Merge Path A ∪ Path B via IoU-NMS                    │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 2: Fixed-GSD 4-Channel Crop Extraction          │
  │  - Adaptive crop sizes (768-1280px) per filament       │
  │  - Ch0: Grayscale | Ch1: CLAHE (clipLimit=2.0)         │
  │  - Ch2: Binary Seed | Ch3: Seed Distance Transform     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 3: Deep Refinement & Composite Loss             │
  │  - smp.Unet (convnext_small / resnet34 backbone)       │
  │  - Trained with 0.30 BCE + 0.40 Dice + 0.30 BoundaryLoss │
  │  - EMA weights, SWA, and Hard-Negative Mining          │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │  Stage 4: Post-Processing & Skeleton Gap Repair        │
  │  - 6-Pass Budget-Aware TTA (Flips x Scales)            │
  │  - Otsu Adaptive Thresholding                          │
  │  - Greedy Mask Collision Resolution                    │
  │  - Skeleton-based Connectivity Gap Closing → COCO RLE   │
  └────────────────────────────────────────────────────────┘
```

### Stage 1: Dual-Path Candidate Proposal
- **Path A (YOLO11s)**: Scans the solar disk at $1280 \times 1280$ with a low threshold ($\text{conf}=0.10$).
- **Path B (CLAHE Connected Components)**: Runs unconditionally to catch faint dark filaments YOLO might miss. Both paths merge using IoU-NMS.

### Stage 2: Fixed-GSD 4-Channel Crop Extraction
- Computes crop sizes dynamically ($768\text{--}1280\text{px}$) so pixels-per-filament-width stays constant.
- Feeds 4 channels: `[Grayscale, CLAHE, Binary Seed (INTER_NEAREST), Distance Transform]`.

### Stage 3: Deep Neural Refinement
- Uses `smp.Unet` trained with a specialized `CompositeLoss` ($0.30 \text{ BCE} + 0.40 \text{ Dice} + 0.30 \text{ BoundaryLoss}$) that puts $4\times$ more weight on filament boundary edges.

### Stage 4: Skeleton Gap Repair & RLE
- Runs 6-pass budget TTA, Otsu self-calibrating threshold, greedy overlap resolution, and **skeleton-based gap repair** to reconnect broken filament segments before exporting lossless COCO RLE strings.

---

## 📊 5. How Are Predictions Scored? (Panoptic Quality & Width Buckets)

$$\text{Panoptic Quality (PQ)} = \text{Segmentation Quality (SQ)} \times \text{Recognition Quality (RQ)}$$

1. **Recognition Quality ($\text{RQ}$)**: Measures detection accuracy ($F_1$-score balancing False Positives and False Negatives).
2. **Segmentation Quality ($\text{SQ}$)**: Measures average boundary IoU of correctly matched filaments.

**v2 Stratified Evaluation**: Our codebase breaks PQ down into width buckets (`thin` $\le 2\text{px}$, `medium` $3\text{--}6\text{px}$, `thick` $>6\text{px}$) so we can specifically track how well the model segments razor-thin solar barb tendrils.

---

## 📂 6. Repository Tour & Code Guide

- [`src/config.py`](src/config.py): Central `V2Config` dataclass — controls all pipeline flags and toggles.
- [`src/losses.py`](src/losses.py): `BoundaryLoss`, `TverskyLoss`, and `CompositeLoss` factory.
- [`src/utils.py`](src/utils.py): RLE encoding/decoding and GroupKFold timestamp splitting.
- [`src/metrics.py`](src/metrics.py): Panoptic Quality ($PQ$) math and stratified width evaluation.
- [`src/dataset.py`](src/dataset.py): Fixed-GSD 4-channel loader + Albumentations pipeline.
- [`src/models.py`](src/models.py): Neural network builder (`smp.Unet` + 4th channel mean init).
- [`src/postprocess.py`](src/postprocess.py): Greedily non-overlapping mask resolution and skeleton gap repair.
- [`src/pipeline.py`](src/pipeline.py): Dual-path proposal, budget TTA, and `EnsemblePipeline`.
- [`train.py`](train.py): Training entrypoint (EMA, SWA, Hard-Negative Mining).
- [`infer.py`](infer.py): Inference entrypoint with Budget Guard.
- [`test_suite.py`](test_suite.py): 8-test local verification suite.

---

## 🚀 7. How You Can Run This Project

### Quick Local Verification Test (No Dataset Needed)
```bash
python test_suite.py
```

### Full Run on Kaggle (Recommended)
Paste this in a Kaggle GPU T4 Notebook cell:

```bash
!pip install -q scipy scikit-image segmentation-models-pytorch albumentations ultralytics
!if [ ! -d "/kaggle/working/VisionX" ]; then git clone https://github.com/gopi470/VisionX.git /kaggle/working/VisionX; fi
%cd /kaggle/working/VisionX
!git pull origin main
!python infer.py --output_dir /kaggle/working
```
