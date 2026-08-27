"""
VisionX v2 Dataset Loader — Steps 1 & 3

v2 changes vs v1:
  - Step 1: Binary mask channels strictly use INTER_NEAREST; assertions after resize.
  - Step 3: Fixed-GSD variable crop sizing (768–1280px); 4-channel input with
            distance transform of seed mask as Ch3. CLAHE clipLimit unified to 2.0.
  - Albumentations-based augmentation pipeline (added in v1 bug-fix round) retained.
"""

import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import polygon_to_mask

try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False

try:
    import scipy.ndimage as _ndi
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ── Crop-bound helpers ────────────────────────────────────────────────────────

def calculate_crop_bounds(bbox: list, img_h: int, img_w: int, context_scale: float = 1.5, min_size: int = 96):
    """
    Computes a square crop region centred on a bounding box.
    (Unchanged from v1 — context_scale=1.5 gives the 40% margin.)
    """
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = int(math.ceil(max(bw, bh) * context_scale))
    side = max(side, min_size)

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side

    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > img_w:
        x0 -= (x1 - img_w); x1 = img_w
    if y1 > img_h:
        y0 -= (y1 - img_h); y1 = img_h

    return max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)


def generate_seed_mask(img_h: int, img_w: int, bbox: list, pad_ratio: float = 0.40) -> np.ndarray:
    """Generates a filled bounding seed mask with margin padding."""
    x, y, bw, bh = bbox
    px, py = bw * pad_ratio, bh * pad_ratio
    x0 = max(0, int(x - px)); y0 = max(0, int(y - py))
    x1 = min(img_w, int(x + bw + px)); y1 = min(img_h, int(y + bh + py))
    seed = np.zeros((img_h, img_w), dtype=np.uint8)
    seed[y0:y1, x0:x1] = 1
    return seed


# ── v2 Step 3: Fixed-GSD crop size computation ───────────────────────────────

def compute_gsd_crop_size(
    bbox_w: float,
    bbox_h: float,
    target_filament_width_px: float = 2.0,
    desired_px_per_filament: float = 4.0,
    min_crop: int = 768,
    max_crop: int = 1280,
) -> int:
    """
    Computes the output crop size so that the effective pixels-per-filament-width
    stays roughly constant regardless of how large the detected bbox is.

    Rationale: ground truth filaments are 1-2 px wide at native 2048×2048.
    At 1024px fixed crop (from a 1400px native region), scale ≈ 0.73×, so a
    2px filament becomes ~1.5px — very near the resolution floor.
    Fixed-GSD targets ~4px per filament width in the crop coordinate space.

    Output is clamped to [min_crop, max_crop].
    """
    # Approximate native filament width relative to bbox long edge
    # Heuristic: filaments span ~1-2px but bboxes are much larger
    long_side = max(bbox_w, bbox_h)
    if long_side <= 0:
        return min_crop

    # Scale factor: how many crop pixels should 1 native pixel map to?
    scale = desired_px_per_filament / target_filament_width_px
    crop_size = int(round(long_side * scale))
    return int(np.clip(crop_size, min_crop, max_crop))


# ── v2 Step 3: Distance transform channel ────────────────────────────────────

def compute_distance_transform_channel(seed_mask: np.ndarray) -> np.ndarray:
    """
    Computes and normalises the distance transform of a binary seed mask.
    Output: float32 array in [0, 1], same spatial shape as seed_mask.
    Uses scipy if available, falls back to cv2.distanceTransform.
    """
    binary = (seed_mask > 0).astype(np.uint8)
    if _SCIPY_AVAILABLE:
        dist = _ndi.distance_transform_edt(binary).astype(np.float32)
    else:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val > 0:
        dist = dist / max_val
    return dist


# ── Step 1: Strict binary mask assertion ─────────────────────────────────────

def _assert_binary(mask: np.ndarray, name: str = "mask") -> None:
    """Asserts that a mask is strictly binary (only 0 and 1 values)."""
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise AssertionError(
            f"[Step 1] {name} is not binary after resize — unique values: {unique}. "
            "Use INTER_NEAREST for binary mask channels."
        )


def _test_mask_binary_assertion() -> None:
    """Self-test: verify binary assertion catches bilinear-resized masks."""
    mask_float = np.zeros((8, 8), dtype=np.float32)
    mask_float[2:6, 2:6] = 1.0
    mask_float[3:5, 3:5] = 0.0
    
    # Linear interpolation on float array produces continuous values between 0.0 and 1.0
    resized_bilinear = cv2.resize(mask_float, (5, 5), interpolation=cv2.INTER_LINEAR)
    passed = False
    try:
        _assert_binary(resized_bilinear, "bilinear_mask")
    except AssertionError:
        passed = True
        
    mask_uint8 = mask_float.astype(np.uint8)
    resized_nearest = cv2.resize(mask_uint8, (5, 5), interpolation=cv2.INTER_NEAREST)
    _assert_binary(resized_nearest, "nearest_mask")
    assert passed, "Binary assertion failed to catch bilinear resize"
    print("[Step 1 test] Binary mask assertion: PASSED")




# ── Dataset ───────────────────────────────────────────────────────────────────

class RefineDataset(Dataset):
    """
    v2 PyTorch dataset for training the cropped refinement network.

    Input tensor (4 channels — v2):
      Ch0: raw grayscale crop
      Ch1: CLAHE-enhanced crop (clipLimit=2.0)
      Ch2: binary seed mask (INTER_NEAREST, strictly binary)
      Ch3: distance transform of seed mask (normalized, INTER_LINEAR OK)

    Set use_distance_transform=False for v1-compatible 3-channel input.
    """
    def __init__(
        self,
        annotation_samples: list,
        crop_size: int = 1024,
        augment: bool = True,
        use_fixed_gsd: bool = True,
        gsd_min_crop: int = 768,
        gsd_max_crop: int = 1280,
        use_distance_transform: bool = True,
    ):
        self.annotation_samples = annotation_samples
        self.crop_size = crop_size
        self.augment = augment
        self.use_fixed_gsd = use_fixed_gsd
        self.gsd_min_crop = gsd_min_crop
        self.gsd_max_crop = gsd_max_crop
        self.use_distance_transform = use_distance_transform
        self._image_cache = {}

        n_channels = 4 if use_distance_transform else 3

        if augment and _ALBUMENTATIONS_AVAILABLE:
            extra = {"clahe": "image", "seed": "mask"}
            if use_distance_transform:
                extra["dist"] = "image"
            self.aug_pipeline = A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.15, rotate_limit=180,
                        border_mode=cv2.BORDER_REFLECT_101, p=0.6,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
                    A.ElasticTransform(alpha=80, sigma=8, p=0.3),
                    A.GridDistortion(num_steps=4, distort_limit=0.2, p=0.25),
                    A.CoarseDropout(
                        num_holes_range=(1, 4), hole_height_range=(20, 40),
                        hole_width_range=(20, 40), fill=0, p=0.25,
                    ),
                ],
                additional_targets={**extra, "mask": "mask"},
            )
        else:
            self.aug_pipeline = None

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

        # Step 3: Fixed-GSD crop sizing
        bw, bh = gt_bbox[2], gt_bbox[3]
        if self.use_fixed_gsd:
            out_size = compute_gsd_crop_size(
                bw, bh, min_crop=self.gsd_min_crop, max_crop=self.gsd_max_crop,
            )
        else:
            out_size = self.crop_size

        # Photometric channels — bilinear OK (continuous values)
        img_crop = cv2.resize(
            raw_img[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_LINEAR,
        )

        # Step 1: Binary mask channels — INTER_NEAREST + strict assertion
        gt_crop = cv2.resize(
            gt_mask[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_NEAREST,
        )
        _assert_binary(gt_crop, "gt_crop")

        seed_crop = cv2.resize(
            seed_mask[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_NEAREST,
        )
        _assert_binary(seed_crop, "seed_crop")

        # Step 3 Ch3: Distance transform (continuous — bilinear resize OK)
        if self.use_distance_transform:
            dist_full = compute_distance_transform_channel(seed_mask)
            dist_crop = cv2.resize(
                dist_full[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        # CLAHE — clipLimit=2.0 (unified with pipeline.py after Bug #5 fix)
        clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_crop = clahe_engine.apply(img_crop)

        # Augmentation
        if self.augment:
            if self.aug_pipeline is not None:
                aug_kwargs = dict(image=img_crop, clahe=clahe_crop, mask=gt_crop, seed=seed_crop)
                if self.use_distance_transform:
                    aug_kwargs["dist"] = dist_crop
                augmented = self.aug_pipeline(**aug_kwargs)
                img_crop = augmented["image"]
                clahe_crop = augmented["clahe"]
                gt_crop = augmented["mask"]
                seed_crop = augmented["seed"]
                if self.use_distance_transform:
                    dist_crop = augmented["dist"]
                # Re-assert binary masks are still binary post-augmentation
                _assert_binary(gt_crop, "gt_crop_post_aug")
                _assert_binary(seed_crop, "seed_crop_post_aug")
            else:
                import random
                if random.random() < 0.5:
                    img_crop = np.fliplr(img_crop).copy()
                    clahe_crop = np.fliplr(clahe_crop).copy()
                    gt_crop = np.fliplr(gt_crop).copy()
                    seed_crop = np.fliplr(seed_crop).copy()
                    if self.use_distance_transform:
                        dist_crop = np.fliplr(dist_crop).copy()
                if random.random() < 0.5:
                    img_crop = np.flipud(img_crop).copy()
                    clahe_crop = np.flipud(clahe_crop).copy()
                    gt_crop = np.flipud(gt_crop).copy()
                    seed_crop = np.flipud(seed_crop).copy()
                    if self.use_distance_transform:
                        dist_crop = np.flipud(dist_crop).copy()

        # Build input tensor
        mean = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225, 0.25], dtype=np.float32)[:, None, None]

        channels = [
            img_crop.astype(np.float32) / 255.0,
            clahe_crop.astype(np.float32) / 255.0,
            seed_crop.astype(np.float32),
        ]
        if self.use_distance_transform:
            channels.append(dist_crop)

        mean_use = mean[:len(channels)]
        std_use  = std[:len(channels)]

        input_tensor = np.stack(channels, axis=0)   # (C, H, W)
        input_tensor = (input_tensor - mean_use) / std_use

        target_tensor = torch.from_numpy(gt_crop.astype(np.float32))[None]
        return torch.from_numpy(input_tensor.astype(np.float32)), target_tensor
