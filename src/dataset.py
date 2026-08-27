"""
VisionX Dataset Loader & Cropped Refinement Dataset
"""

import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import polygon_to_mask


def calculate_crop_bounds(bbox: list[float], img_h: int, img_w: int, context_scale: float = 1.5, min_size: int = 96):
    """
    Computes a square crop bounding region centered around a candidate bounding box.
    """
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = int(math.ceil(max(bw, bh) * context_scale))
    side = max(side, min_size)

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side

    # Clamp bounds within image boundaries
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > img_w:
        x0 -= (x1 - img_w)
        x1 = img_w
    if y1 > img_h:
        y0 -= (y1 - img_h)
        y1 = img_h

    return max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)


def generate_seed_mask(img_h: int, img_w: int, bbox: list[float], pad_ratio: float = 0.40) -> np.ndarray:
    """
    Generates a filled bounding seed mask with margin padding.
    """
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1, y1 = min(img_w, int(x + bw + px)), min(img_h, int(y + bh + py))

    seed = np.zeros((img_h, img_w), dtype=np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


class RefineDataset(Dataset):
    """
    PyTorch dataset for training the cropped refinement network.
    Constructs a 3-channel input: [Grayscale Image, CLAHE Contrast Image, Candidate Seed Mask].
    """
    def __init__(self, annotation_samples: list, crop_size: int = 512, augment: bool = True):
        self.annotation_samples = annotation_samples
        self.crop_size = crop_size
        self.augment = augment
        self._image_cache = {}

    def __len__(self):
        return len(self.annotation_samples)

    def _load_gray_image(self, path):
        key = str(path)
        if key not in self._image_cache:
            img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read image at {path}")
            self._image_cache[key] = img
        return self._image_cache[key]

    def __getitem__(self, idx):
        sample, ann = self.annotation_samples[idx]
        raw_img = self._load_gray_image(sample["path"])
        h, w = raw_img.shape

        gt_mask = polygon_to_mask(ann["segmentation"], height=h, width=w)
        gt_bbox = ann["bbox"]

        seed_mask = generate_seed_mask(h, w, gt_bbox, pad_ratio=0.40)
        x0, y0, x1, y1 = calculate_crop_bounds(gt_bbox, h, w)

        # Resize crops to target crop size
        img_crop = cv2.resize(raw_img[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        gt_crop = cv2.resize(gt_mask[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        seed_crop = cv2.resize(seed_mask[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        # Generate CLAHE channel
        clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_crop = clahe_engine.apply(img_crop)

        # Random Flips Augmentation
        if self.augment and random.random() < 0.5:
            img_crop, clahe_crop = np.fliplr(img_crop).copy(), np.fliplr(clahe_crop).copy()
            gt_crop, seed_crop = np.fliplr(gt_crop).copy(), np.fliplr(seed_crop).copy()
        if self.augment and random.random() < 0.5:
            img_crop, clahe_crop = np.flipud(img_crop).copy(), np.flipud(clahe_crop).copy()
            gt_crop, seed_crop = np.flipud(gt_crop).copy(), np.flipud(seed_crop).copy()

        # Stack into 3-channel tensor
        input_tensor = np.stack([img_crop, clahe_crop, seed_crop * 255], axis=0).astype(np.float32) / 255.0

        # Normalize using standard ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        input_tensor = (input_tensor - mean) / std

        target_tensor = torch.from_numpy(gt_crop.astype(np.float32))[None]

        return torch.from_numpy(input_tensor), target_tensor
