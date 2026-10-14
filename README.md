# VisionX: Solar Filament Instance Segmentation (MAGFiLO 2026)

Team VisionX, Department of Computer Science and Engineering, SSN College of Engineering, Chennai, India.
Authors: Haripriya, Ishana, GuruPrakash and Gopinath.

Instance segmentation of solar filaments in full-disk H-alpha images for the MAGFiLO 2026 Kaggle challenge. For every test image the submission must contain one run-length-encoded (RLE) mask per filament, with no pixel shared between two masks.

## Highlights

- Two successive solutions are documented: a classical-candidate baseline with a U-Net refiner (V1) and a YOLO11x-seg detector-segmenter (V2, final).
- V1: the refiner reaches a Dice of 0.822 when seeded with ground-truth boxes, but only 0.210 Dice and 0.108 panoptic quality end to end, emitting 22.0 masks per image against 7.1 annotated.
- V2: mask mAP50 of 0.273 and box mAP50 of 0.278 on 109 validation images (821 filaments), and 1,125 valid masks (6.25 per image) for all 180 test images.
- Public leaderboard score improved from 0.36 to 0.55 (rank 70 at the time of the screenshot; the rank may have changed since).

## Problem

Solar filaments are dark, elongated plasma structures seen against the disk in H-alpha images. A median filament covers about 0.029 % of a 2048 x 2048 image, and the disk also contains dark features (sunspots, noise) that are not annotated filaments. Reliable automatic extraction supports space-weather monitoring and long-term statistical studies.

## Repository contents

```
VisionX/
  v1_version_challenge1.ipynb              V1: classical candidates + ResNet34 U-Net refiner
  v2_magfilo_yolo11x.ipynb                 V2: YOLO11x-seg training and submission (final)
  v3_magfilo_yolo11x_fixed_with_output.ipynb   identical copy of V2 (same file contents)
  .gitignore
  README.md
```

The notebooks are written for Kaggle and keep their cell outputs, so training logs and validation tables can be read without re-running them. `v2_magfilo_yolo11x.ipynb` and `v3_magfilo_yolo11x_fixed_with_output.ipynb` have the same checksum; the V3 name is kept for traceability. The commit message of the repository records the final result: PQ 0.55 with YOLO11x-seg.

## Method

### V1 (baseline): classical candidates and U-Net refiner

| Stage | Technique | Output |
|-------|-----------|--------|
| A | Hough circle (blob fallback) for the solar disk | Disk centre and radius |
| B | Flat-fielding and dark-region threshold | Binary dark mask |
| C | Connected components and area filter | Candidate boxes |
| D | ResNet34 U-Net on a cropped box (raw, CLAHE and box-seed channels) | Per-box soft mask |
| E | Threshold, size filter, overlap resolution, RLE | Submission rows |

The refiner segments whatever it is given, so thresholded false candidates (sunspots, noise) pass through. This is why a high refiner Dice does not translate into end-to-end quality.

### V2 (final): YOLO11x-seg

| Module | Description |
|--------|-------------|
| M1 | Convert COCO polygons to normalised YOLO labels (x/W, y/H; classes 1-4 mapped to 0-3) |
| M2 | 90/10 train/validation split with seed 42 |
| M3 | Fine-tune COCO-pretrained YOLO11x-seg (1024 px, batch 8, 15 epochs, AdamW, cosine learning rate) |
| M4 | Inference one image at a time (confidence 0.20, NMS IoU 0.50; retry at 0.05 if nothing is found) |
| M5 | Fill predicted polygons directly on a 2048 x 2048 canvas |
| M6 | Confidence-ordered overlap removal: M'_k = M_k minus the union of M'_j for j < k |
| M7 | COCO compressed RLE through pycocotools, written to `submission.csv` |

The output of each module is the input of the next one.

## Dataset

MAGFiLO 2026: GONG H-alpha full-disk images with manually annotated filaments, provided through the Kaggle competition `filament-segmentation-2026`. The data is not redistributed in this repository.

| Item | Value |
|------|-------|
| Training COCO image entries | 1,154 |
| Annotated filaments | 8,199 (median 7 per image) |
| Test images (hidden labels) | 180 |
| Image size | 2048 x 2048 |
| Classes | Left, Right, Unidentifiable, Ambiguous |

## Results

V2 validation (109 images, 821 instances), separate re-validation of the best checkpoint:

| Class | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|-------|-------:|-------:|-----------:|--------------:|
| All | 0.292 | 0.468 | 0.273 | 0.092 |
| Left | 0.324 | 0.498 | 0.323 | 0.112 |
| Right | 0.269 | 0.534 | 0.270 | 0.096 |
| Unidentifiable | 0.282 | 0.371 | 0.226 | 0.068 |

Box mAP50 is 0.278. The same checkpoint scored a mask mAP50 of 0.257 at the end of training; the 0.016 gap between the two validation runs has not been isolated, so both numbers are reported in the paper.

Training used two Tesla T4 GPUs and took about 0.46 hours.

## How to run

The notebooks are designed for Kaggle, where the competition data is mounted automatically.

1. Create a new Kaggle notebook and join the competition `filament-segmentation-2026`, so that the data appears under `/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`.
2. Upload `v2_magfilo_yolo11x.ipynb` (V2) or `v1_version_challenge1.ipynb` (V1) to the notebook.
3. Enable a GPU accelerator (two T4 GPUs were used for the reported run).
4. Run all cells in order. The first cell installs the dependencies (`ultralytics`, `pycocotools`, `torchmetrics`, `opencv-python-headless`; V1 also installs `segmentation-models-pytorch` and `albumentations`).
5. The V2 notebook writes the converted dataset and checkpoints to `/kaggle/working/magfilo_yolo` and the final predictions to `submission.csv`. Submit that file to the competition.

To run outside Kaggle, change `DATA_ROOT` and `WORK_DIR` in the configuration cell (cell "3. CONFIGURATION" in the V2 notebook) to your local paths, which must follow the same `train/train_images`, `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` and `test/test_images` layout. Use Python 3.10 or newer with a CUDA GPU.

Key settings in the V2 configuration cell: `IMG_SIZE = 1024`, `BATCH = 8`, `EPOCHS = 15`, `VAL_SPLIT = 0.1`, `SEED = 42`, `MODEL_CFG = "yolo11x-seg.pt"`.

The submission has one row per mask with identifiers of the form `<image>_<k>` and a COCO compressed RLE string.

## Known limitations

- The split was made over 1,154 COCO entries, but the image directory held only 707 distinct files. At least 66 files may therefore appear in both train and validation, so validation scores are probably optimistic. A file-name split or k-fold cross-validation is needed.
- Training uses half the native resolution (1024 px). Full-resolution polygon filling gives sharper boundaries but cannot recover structure lost in downsampling.
- Only the first polygon of each annotation is used as a training target.
- The validation split contains no Ambiguous instances.
- V1 was scored on 15 training images, and V1 and V2 metrics are not directly comparable.
- Ablations for resolution, confidence threshold, NMS IoU and model size have not been run.

## Future work

Split by file name, train longer or on tiles at higher resolution, use all polygon parts of each annotation, tune thresholds on a clean validation set, add test-time augmentation or an ensemble, and use the spine and chirality annotations from MAGFiLO.

## Citation

```bibtex
@misc{visionx2026,
  author = {Haripriya and Ishana and GuruPrakash and Gopinath},
  title  = {VisionX: Solar Filament Instance Segmentation for MAGFiLO 2026},
  year   = {2026},
  url    = {https://github.com/gopi470/VisionX}
}
```

Dataset: A. Ahmadzadeh et al., "A dataset of manually annotated filaments from H-alpha observations," Scientific Data, vol. 11, 2024.

## Acknowledgements

The MAGFiLO and NSO/GONG teams for the data, and the challenge organisers for the Kaggle platform. The detector is built on Ultralytics YOLO11, and the V1 refiner on segmentation-models-pytorch.

## License

This repository does not currently include a license file, so all rights are reserved by the authors. The competition data is governed by the terms of the MAGFiLO 2026 Kaggle competition.
