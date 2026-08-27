# VisionX - Complete Pipeline Workflow & Architecture Guide 🛠️🔄

This document provides a comprehensive step-by-step breakdown of the **VisionX end-to-end workflow**, detailing every stage from raw GONG H-Alpha data ingestion to final Kaggle RLE submission encoding.

---

## 🏗️ 1. End-to-End Pipeline Workflow Diagram

```
                       ┌────────────────────────────────────────────────┐
                       │   Raw GONG H-Alpha Image (2048 x 2048 x 1)     │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │ Stage 1: Candidate Proposal & Feature Extractor│
                       │ - YOLO11s / Region Proposal Network           │
                       │ - Adaptive CLAHE Contrast Region Thresholding │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │ Stage 2: 1024x1024 Multi-Channel Crop Extractor│
                       │ - 40% Context Margin Padding                   │
                       │ - Stack: [Grayscale, CLAHE Filter, Seed Mask] │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │ Stage 3: SOTA U-Net++ ConvNeXt Large Refiner   │
                       │ - Nested Dense Skip Pathways                   │
                       │ - Combined BCE + Soft Dice Loss Optimization   │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │ Stage 4: 15-Pass Test-Time Augmentation (TTA)  │
                       │ - 5-Directional Seed Box Jittering [0, ±8px]   │
                       │ - Horizontal & Vertical Flip Ensembling        │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │ Stage 5: Non-Overlapping Collision Resolution  │
                       │ - Confidence-Sorted Greedy Pixel Allocation   │
                       │ - PyCOCO Lossless RLE String Encoding          │
                       └───────────────────────┬────────────────────────┘
                                               │
                                               ▼
                       ┌────────────────────────────────────────────────┐
                       │         Final Kaggle submission.csv            │
                       └────────────────────────────────────────────────┘
```

---

## 🔬 2. Step-by-Step Technical Process Breakdown

### Step 1: Data Ingestion & GroupKFold Timestamp Split ([`src/utils.py`](src/utils.py))
- **Observation Parsing**: Reads `MAGFiLO_1.0_Annotations_kaggle2026_train.json` containing ground-truth COCO polygons, spines, bounding boxes, and image metadata.
- **Leakage Prevention**: Groups observations by timestamp stem (`YYYYMMDDHHMMSS`) so multi-annotator versions of the exact same solar observation (`010101-` vs `010102-`) never leak between training and validation folds.

### Step 2: Adaptive Candidate Proposal ([`src/pipeline.py`](src/pipeline.py))
- **Detection Network**: Scans full-resolution $2048 \times 2048$ solar disk at $1280 \times 1280$ inference resolution using **YOLO11s** ($\text{confidence threshold} = 0.15$).
- **Dark Feature Fallback**: If the detector returns 0 candidates, an **Adaptive Solar Dark Region Thresholding** algorithm applies CLAHE contrast filtering and connected components analysis to ensure high recall across fainter plasma structures.

### Step 3: High-Resolution $1024 \times 1024$ Crop Extraction ([`src/dataset.py`](src/dataset.py))
- **Context Padding**: Adds a 40% margin around candidate boxes to capture surrounding magnetic field background context.
- **3-Channel Tensor Assembly**:
  - `Channel 0`: Raw Grayscale Crop.
  - `Channel 1`: CLAHE (Contrast Limited Adaptive Histogram Equalization) Contrast-Enhanced Crop.
  - `Channel 2`: Candidate Seed Mask.
- **Scale Resolution**: Resizes candidate crops to **$1024 \times 1024$ pixels** to preserve thin barb microstructures ($1-2\text{ pixels wide}$).

### Step 4: U-Net++ Nested Feature Refinement ([`src/models.py`](src/models.py))
- **Architecture**: **U-Net++ (Nested U-Net)** with a heavy **ConvNeXt Large** / **EfficientNet-B7** encoder.
- **Loss Function**: **Combined BCE + Soft Dice Loss**:
  $$\mathcal{L}_{\text{total}} = 0.45 \cdot \mathcal{L}_{\text{BCE}} + 0.55 \cdot \mathcal{L}_{\text{SoftDice}}$$
- **Optimization**: 40 Training Epochs with Cosine Annealing Learning Rate scheduling ($3\times 10^{-4} \rightarrow 1\times 10^{-6}$) and Automatic Mixed Precision (AMP).

### Step 5: 15-Pass Test-Time Augmentation (TTA) ([`src/pipeline.py`](src/pipeline.py))
- **Box Jittering**: Shifts seed masks in 5 directions $[(0, 0), (+8\text{px}, 0), (-8\text{px}, 0), (0, +8\text{px}), (0, -8\text{px})]$.
- **Spatial Flipping**: Evaluates Original, Horizontal Flip, and Vertical Flip passes per shift ($5 \times 3 = 15\text{ forward passes per candidate crop}$).
- **Ensemble Averaging**: Computes soft probability average maps to smooth boundary noise and sharpen true filament edges.

### Step 6: Non-Overlapping Collision Resolution & RLE Encoding ([`src/postprocess.py`](src/postprocess.py))
- **Greedy Mask Allocation**: Sorts predicted instance masks by confidence and assigns disputed overlapping pixels to the higher confidence instance, guaranteeing **zero pixel duplication**.
- **PyCOCO Encoding**: Encodes cleaned binary masks into lossless COCO RLE strings via `pycocotools.mask.encode(np.asfortranarray(mask))` and exports formatted `submission.csv`.

---

## 📊 3. Metric Evaluation: Panoptic Quality ($PQ$)

Predictions are evaluated against ground truth using **Panoptic Quality ($PQ$)**:

$$\text{PQ} = \text{SQ} \times \text{RQ} = \frac{\sum_{(y, \hat{y}) \in \text{TP}} \text{IoU}(y, \hat{y})}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

- **$\text{SQ}$ (Segmentation Quality)**: Average IoU score for true positive matches ($\text{IoU} > 0.50$).
- **$\text{RQ}$ (Recognition Quality)**: $F_1$-score of instance detection accuracy.
