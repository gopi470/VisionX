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
    Encodes a 2D binary numpy array into COCO RLE string format.
    """
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")


def decode_rle(rle_str: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decodes a COCO RLE string back into a binary numpy array of (height, width).
    """
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
