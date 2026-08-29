"""
VisionX Utilities - RLE handling, COCO JSON parsing, and GroupKFold splitting.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def encode_rle(binary_mask: np.ndarray) -> str:
    """
    Encodes a 2D binary numpy array into Kaggle standard space-separated 1-indexed RLE string format.
    Column-first (Fortran) ordering standard for Kaggle vision competitions.
    """
    pixels = binary_mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def decode_rle(rle_str: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decodes a space-separated Kaggle RLE string OR COCO RLE string back into a binary numpy array of (height, width).
    """
    if not rle_str or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(rle_str, str) and (" " in rle_str or rle_str.isdigit()):
        s = rle_str.split()
        if not s:
            return np.zeros((height, width), dtype=np.uint8)
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape((width, height)).T
    else:
        # COCO RLE format fallback
        rle = {"counts": rle_str.encode("utf-8") if isinstance(rle_str, str) else rle_str, "size": [height, width]}
        return mask_utils.decode(rle).astype(np.uint8)


def polygon_to_mask(segmentation, height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Converts COCO polygon format or PyCOCO RLE into a 2D binary mask.
    """
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def load_coco_dataset(json_path: Path, images_dir: Path):
    """
    Parses COCO JSON annotations and maps them to image file paths.
    """
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    images_by_id = {img["id"]: img for img in coco_data["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    samples = []
    for image_id, img in images_by_id.items():
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            alt_suffix = ".jpg" if img_path.suffix.lower() == ".jpeg" else ".jpeg"
            alt_path = img_path.with_suffix(alt_suffix)
            if alt_path.exists():
                img_path = alt_path
            else:
                continue

        samples.append({
            "image_id": image_id,
            "file_name": img["file_name"],
            "path": img_path,
            "stem": img_path.stem,
            "height": img.get("height", 2048),
            "width": img.get("width", 2048),
            "annotations": anns_by_image[image_id],
        })

    return samples


def get_group_kfold_split(samples: list, val_ratio: float = 0.15, seed: int = 42):
    """
    Groups samples by observation timestamp stem (YYYYMMDDHHMMSS) to ensure zero data leakage across folds.
    """
    random.seed(seed)
    stems = sorted(list({s["stem"][:14] if len(s["stem"]) >= 14 else s["stem"] for s in samples}))
    random.shuffle(stems)

    n_val = max(1, int(len(stems) * val_ratio))
    val_stems = set(stems[:n_val])

    train_samples, val_samples = [], []
    for s in samples:
        stem_group = s["stem"][:14] if len(s["stem"]) >= 14 else s["stem"]
        if stem_group in val_stems:
            val_samples.append(s)
        else:
            train_samples.append(s)

    return train_samples, val_samples
