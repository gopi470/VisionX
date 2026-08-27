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

## 2. Running on Kaggle Notebooks (Detailed Step-by-Step Guide)

### Step 1: Create a New Kaggle Notebook
1. Open your browser and go to [kaggle.com](https://www.kaggle.com).
2. Log into your Kaggle account.
3. On the left sidebar, click the **+ Create** button (or go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**).
4. A new blank Jupyter Notebook editor will open.

---

### Step 2: Attach the Competition Dataset
1. In the right-hand panel of the Notebook interface, locate the **Input** section.
2. Click the **+ Add Data** button at the top right of that panel.
3. A search modal will appear. In the search box:
   - Search for `filament-segmentation-2026` or `Solar Filament Segmentation Challenge 2026`.
   - Alternatively, search for `MAGFiLO_1.0_Kaggle_2026`.
4. Click the **+** (Add) button next to the dataset.
5. Close the search panel. You will now see the dataset listed under **Input** in the right panel (usually at `/kaggle/input/filament-segmentation-2026` or `/kaggle/input/solarfilament/...`).

---

### Step 3: Configure GPU & Internet Settings
1. In the right-hand panel under **Notebook Options**:
2. Find the **Session options** or **Accelerator** dropdown:
   - Change **Accelerator** from `None` to **GPU T4 x2** or **GPU P100**.
3. Find the **Internet** toggle switch:
   - Switch **Internet** to **On** (this allows the notebook to clone GitHub repositories and install pip packages).
4. If prompted to restart the session, click **Confirm / Restart**.

---

### Step 4: Run Code Cells in Kaggle

#### Cell 1: Clone Repository & Install Packages
```python
!git clone https://github.com/gopi470/VisionX.git
%cd VisionX
!pip install -q -r requirements.txt
```

#### Cell 2: Verify Input Dataset Path
```python
import os
print("Mounted input folders:")
print(os.listdir('/kaggle/input'))
# Note: If this prints ['notebooks'] only, you still need to click '+ Add Input' 
# in the right panel and search for 'filament-segmentation-2026'!
```

#### Cell 3: Execute Inference & Generate Submission CSV
```python
# Adjust the dataset path according to what was printed in Cell 2
!python infer.py --data_root /kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 --output_dir /kaggle/working
```

#### Cell 4: Preview Submission CSV
```python
import pandas as pd
sub_path = '/kaggle/working/submission.csv'
if os.path.exists(sub_path):
    sub = pd.read_csv(sub_path)
    print(f"Submission generated successfully! Total rows: {len(sub)}")
    display(sub.head(10))
else:
    print("Submission file not found yet. Check logs above.")
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
