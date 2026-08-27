# Setup & Execution Guide - Solar Filament Segmentation 2026

This guide provides instructions for running the **VisionX** solar filament segmentation pipeline on **Kaggle Notebooks** (Recommended) or locally.

---

## 1. Prerequisites & Dependencies

### Required Packages
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU acceleration recommended)
- `albumentations`
- `segmentation-models-pytorch`
- `pycocotools`
- `ultralytics`
- `opencv-python`
- `pandas` / `numpy` / `tqdm`

Install dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 2. Running on Kaggle Notebooks (Recommended)

### Step 1: Attach Dataset
1. Open your Kaggle Notebook.
2. In the right panel, click **+ Add Data**.
3. Search for and attach the competition dataset: `filament-segmentation-2026` or `MAGFiLO_1.0_Kaggle_2026`.
4. Ensure dataset path is mounted under `/kaggle/input/MAGFiLO_1.0_Kaggle_2026/` or `/kaggle/input/solarfilament/MAGFiLO_1.0_Kaggle_2026/`.

### Step 2: Environment Options
1. Set **Accelerator** to **GPU P100** or **T4 (Dual)**.
2. Set **Internet** to **On** (if installing packages dynamically).

### Step 3: Execution
Run the complete pipeline notebook or Python script:
```python
# In Kaggle Notebook Cell:
!python infer.py --data_root /kaggle/input/MAGFiLO_1.0_Kaggle_2026 --output_dir /kaggle/working
```

This generates `submission.csv` in `/kaggle/working/submission.csv` ready for direct submission.

---

## 3. Local Environment Setup

### Directory Setup
Place the competition dataset inside the workspace directory:
```
VisionX/
└── MAGFiLO_1.0_Kaggle_2026/
    ├── train/
    │   ├── train_images/
    │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
    └── test/
        └── test_images/
```

### Running Validation & Training
```bash
python train.py --data_root ./MAGFiLO_1.0_Kaggle_2026 --epochs 15 --batch_size 8
```

### Running Inference
```bash
python infer.py --data_root ./MAGFiLO_1.0_Kaggle_2026 --output_dir ./outputs
```

---

## 4. Verification Checklist

Before submitting to Kaggle, verify:
- [ ] `submission.csv` exists and is non-empty.
- [ ] Header consists of `filament_id,segmentation_rle`.
- [ ] Each `filament_id` matches the format `<image_stem>_<index>` (e.g. `20150125172714Mh_1`).
- [ ] RLE string counts are valid unquoted strings generated via `pycocotools`.
