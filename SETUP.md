# Setup & Execution Guide - Solar Filament Segmentation 2026 (v2)

This guide provides instructions for running the **VisionX v2** solar filament segmentation pipeline on **Kaggle Notebooks** (Recommended) or locally.

---

## 1. Prerequisites & Dependencies

### Required Packages
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU acceleration recommended)
- `scipy` $\ge 1.10.0$
- `scikit-image` $\ge 0.21.0$
- `albumentations` $\ge 1.4.0$
- `segmentation-models-pytorch` $\ge 0.3.3$
- `pycocotools` $\ge 2.0.6$
- `ultralytics` $\ge 8.0.0$
- `opencv-python-headless`
- `pandas` / `numpy` / `tqdm`

Install dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 2. Running on Kaggle Notebooks (Recommended)

### Step 1: Create Notebook & Enable GPU + Internet
1. Open [kaggle.com](https://www.kaggle.com) and click **+ Create Notebook**.
2. Under **Session options** on the right panel:
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   - Set **Internet** to **On**.

---

### Step 2: Single-Cell Execution Code (v2 Recommended)

Copy and run the following single cell in your Kaggle Notebook:

```bash
!pip install -q scipy scikit-image segmentation-models-pytorch albumentations ultralytics

# Clone repository if not present
!if [ ! -d "/kaggle/working/VisionX" ]; then git clone https://github.com/gopi470/VisionX.git /kaggle/working/VisionX; fi

# Navigate into repository and pull main
%cd /kaggle/working/VisionX
!git pull origin main

# Run v2 inference pipeline
!python infer.py --output_dir /kaggle/working
```

---

### Step 3: Optional Advanced Flags

- **Custom TTA Pass Count**:
  ```bash
  !python infer.py --tta_passes 3 --output_dir /kaggle/working
  ```
- **Ensemble Run** (if secondary refiner checkpoint available):
  ```bash
  !python infer.py --ensemble --secondary_weights /kaggle/working/refiner_v2b.pt --output_dir /kaggle/working
  ```
- **v1-Compatibility Run** (ablation baseline):
  ```bash
  !python infer.py --in_channels 3 --no_fixed_gsd --no_boundary_head --no_skeleton_repair --output_dir /kaggle/working
  ```

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

### Running Unit Tests (No GPU/Dataset Needed)
```bash
python test_suite.py
```

### Running v2 Model Training
```bash
python train.py --data_root ./MAGFiLO_1.0_Kaggle_2026 --epochs 40 --batch_size 4 --encoder_name convnext_small
```

### Running v2 Inference
```bash
python infer.py --data_root ./MAGFiLO_1.0_Kaggle_2026 --output_dir ./outputs
```

---

## 4. Verification Checklist

Before submitting to Kaggle, verify:
- [ ] `submission.csv` exists in `/kaggle/working/`.
- [ ] Header consists of `filament_id,segmentation_rle`.
- [ ] Each `filament_id` matches the format `<image_stem>_<index>` (e.g. `20150125172714Mh_1`).
- [ ] RLE string counts are valid unquoted strings generated via `pycocotools`.
