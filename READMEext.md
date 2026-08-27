# VisionX - Ultimate Beginner to Expert Guide ☀️🛰️

Welcome to the comprehensive, step-by-step explanatory guide for **VisionX**! This document is written so that **anyone**—from high school students and curious space enthusiasts to machine learning engineers—can understand **what solar filaments are**, **why segmenting them is critical for protecting Earth**, **how computer vision models solve this challenge**, and **how to run our code**.

---

## 📖 Table of Contents
1. [☀️ What Are Solar Filaments & Why Do They Matter?](#1-what-are-solar-filaments--why-do-they-matter)
2. [🛰️ Space Weather Threats: Protect Our Tech Grid](#2-space-weather-threats-protect-our-tech-grid)
3. [🧩 The Challenge: Why Is This Hard For AI?](#3-the-challenge-why-is-this-hard-for-ai)
4. [💡 The VisionX Solution: Step-by-Step Explanation](#4-the-visionx-solution-step-by-step-explanation)
5. [📊 How Are Predictions Scored? (Panoptic Quality Explained)](#5-how-are-predictions-scored-panoptic-quality-explained)
6. [📂 Repository Tour & Code Guide](#6-repository-tour--code-guide)
7. [🚀 How You Can Run This Project](#7-how-you-can-run-this-project)

---

## ☀️ 1. What Are Solar Filaments & Why Do They Matter?

### The Sun's Magnetic Atmosphere
The Sun is a giant, glowing ball of super-heated gas (plasma). Powerful magnetic fields bubble up from deep inside the Sun and twist through its outer atmosphere (the chromosphere).

### What Is a Filament?
- A **solar filament** is a massive ribbon of dense, cooler plasma ($10,000\text{ °C}$) suspended high in the Sun's atmosphere ($1,000,000\text{ °C}$) by magnetic field lines.
- Because filaments are cooler than the hot surface behind them, when we view the Sun using special filters (specifically **H-Alpha light at 656.28 nanometers**), filaments appear as **dark, snake-like threads** across the Sun.
- When a filament rotates to the edge (limb) of the Sun, it sticks out into dark space and looks like a bright, glowing loop—scientists call this a **prominence**.

![Actual GONG H-Alpha Solar Observation Image (Credit: NSO/AURA/NSF)](docs/solar_filament_sample.jpeg)
*Figure 1: Real high-resolution GONG H-Alpha solar observation image from the MAGFiLO dataset. The dark thread-like features are solar filaments! (Credit: NSO/AURA/NSF).*

---

## 🛰️ 2. Space Weather Threats: Protect Our Tech Grid

Why do scientists care about finding every single filament?

> **Key Insight**: Filaments are the physical anchors of **Coronal Mass Ejections (CMEs)**. 

When the magnetic field lines holding a filament become unstable, they snap and erupt, launching **billions of tons of magnetized plasma into space at millions of miles per hour**.

If an eruption is directed toward Earth:
1. **Power Grid Failures**: Geomagnetic storms induce excess electrical current in power lines, blowing out high-voltage transformers (e.g. the 1989 Quebec blackouts).
2. **Satellite & GPS Disruption**: Solar radiation expands Earth's upper atmosphere, dragging down low-Earth orbit satellites and scrambling GPS signals used by commercial aviation, shipping, and smartphones.
3. **Astronaut Radiation Hazard**: Space radiation from erupting filaments is lethal to space station crews during spacewalks or future Moon/Mars missions.

Automated AI segmentation allows space weather observatories to **track filaments 24/7/365** and issue early warnings before dangerous solar eruptions occur!

---

## 🧩 3. The Challenge: Why Is This Hard For AI?

Segmenting solar filaments from ground-based observatory telescopes (like the GONG network) is surprisingly difficult:

1. **Tiny Thread-Like Features ("Barbs")**:
   - Filaments aren't simple smooth ovals. They have a main central **spine** with microscopic thread-like tendrils extending outward called **barbs**. Standard AI models that downsample images turn these $1-3\text{ pixel}$ barbs invisible.
2. **Sun Noise & Atmospheric Blurring**:
   - Ground telescopes look through Earth's atmosphere, which causes shimmering, variable brightness, atmospheric clouds, and optical noise.
3. **Multi-Annotator Variations**:
   - Human solar physics experts annotated the dataset (MAGFiLO). Sometimes two different human experts outlined the exact same filament slightly differently! AI must learn robust features despite human variations.

---

## 💡 4. The VisionX Solution: Step-by-Step Explanation

To solve these challenges without losing tiny barb details, **VisionX** uses a **2-Stage Cascade Pipeline**:

```
 ┌────────────────────────────────────────────────────────┐
 │           Full H-Alpha Image (2048 x 2048)             │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Step 1: Quick Scanner (Detector)                      │
 │  - Scans full Sun to locate candidate filament boxes  │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Step 2: Zoomed Refinement Network (U-Net)             │
 │  - Zooms in on each box with 40% margin               │
 │  - Applies CLAHE contrast filter to pop dark threads   │
 │  - Segments precise fine-grain boundaries at 512x512   │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Step 3: Overlap Cleanup & Kaggle RLE Formatting       │
 │  - Greedily resolves pixel disputes between boxes      │
 │  - Encodes final masks into lossless COCO RLE strings  │
 └────────────────────────────────────────────────────────┘
```

### Step 1: Candidate Detector
First, an object detector quickly scans the entire $2048 \times 2048$ solar disk and draws bounding boxes around all suspected filaments.

### Step 2: High-Resolution Crop Refinement
Instead of forcing the AI to look at the whole giant image at once, we **crop out each box with a 40% padding margin**. We feed a 3-layer image stack into a U-Net neural network:
- **Layer 1**: The original grayscale crop.
- **Layer 2**: A **CLAHE** (Contrast Limited Adaptive Histogram Equalization) image, which boosts local contrast so faint filament barbs pop out clearly.
- **Layer 3**: A seed mask showing the candidate region.

### Step 3: Overlap Resolution
If two candidate boxes overlap on the Sun, our post-processor automatically assigns disputed pixels to the prediction with the higher confidence score, ensuring **zero double-counted pixels**.

---

## 📊 5. How Are Predictions Scored? (Panoptic Quality Explained)

The Kaggle competition uses **Panoptic Quality ($PQ$)**, which is calculated as:

$$\text{Panoptic Quality (PQ)} = \text{Segmentation Quality (SQ)} \times \text{Recognition Quality (RQ)}$$

Think of it like a test with two grades:
1. **Recognition Quality ($\text{RQ}$)**: *Did you find the right objects?*
   - Measures how well you detected real filaments without missing any ($\text{False Negatives}$) or hallucinating fake ones ($\text{False Positives}$).
2. **Segmentation Quality ($\text{SQ}$)**: *How clean are your outlines?*
   - For every filament you found correctly, it calculates the **Intersection over Union (IoU)**—how precisely your outline matches the expert human annotation.

Multiplying $\text{SQ} \times \text{RQ}$ gives the final **Panoptic Quality score (between 0.0 and 1.0)**.

---

## 📂 6. Repository Tour & Code Guide

Here is a simple map of our codebase:

- [`src/utils.py`](src/utils.py): Tools for converting masks to Kaggle RLE strings and splitting data cleanly without leakage.
- [`src/metrics.py`](src/metrics.py): Code that calculates Panoptic Quality ($PQ$), $SQ$, and $RQ$.
- [`src/dataset.py`](src/dataset.py): Image loader that crops regions and applies CLAHE contrast enhancement.
- [`src/models.py`](src/models.py): The U-Net neural network definition and loss function.
- [`src/postprocess.py`](src/postprocess.py): Resolves overlapping masks so filaments don't collision.
- [`train.py`](train.py): Script you run to train the AI model on your GPU.
- [`infer.py`](infer.py): Script that generates `submission.csv` for Kaggle.
- [`test_suite.py`](test_suite.py): Self-test script to verify that math and code work properly.

---

## 🚀 7. How You Can Run This Project

### Quick Local Verification Test (No Dataset Needed)
Run our built-in test suite to verify that all mathematical operations, RLE encoders, and neural networks work on your computer:
```bash
python test_suite.py
```

### Full Run on Kaggle (Recommended)
1. Open [Kaggle](https://www.kaggle.com), create a new Notebook, and attach the `MAGFiLO_1.0_Kaggle_2026` dataset.
2. Enable **GPU T4** or **P100** acceleration in notebook settings.
3. Run inference to generate your submission file:
   ```bash
   python infer.py --data_root /kaggle/input/MAGFiLO_1.0_Kaggle_2026 --output_dir /kaggle/working
   ```
4. Download `/kaggle/working/submission.csv` and submit it to the leaderboard!
