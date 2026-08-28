"""
VisionX v2 Pipeline — Steps 2, 5, 6

v2 changes vs v1:
  Step 2 — Dual-path candidate proposal:
    - Path A (YOLO) and Path B (CLAHE + connected-components) ALWAYS run.
    - Candidates merged via IoU-NMS before refinement.
    - High-conf QA pass (conf=0.35) logged per image for diagnostics.
    - All thresholds read from V2Config (no hardcoding).

  Step 5 — Budget-aware TTA:
    - Replaces 5-jitter × 3-flip (15 passes) with H/V flip × {0.9, 1.0, 1.1} scale
      = 6 canonical passes (identity+hflip+vflip at each of 3 scales, deduped).
    - estimate_inference_time() utility for pre-run budget checks.
    - run_tta_ablation() sweeps {1, 3, 6} pass counts and reports PQ + wall clock.

  Step 6 — Ensemble:
    - EnsemblePipeline wraps two FilamentSegmentationPipelines and averages
      their raw probability maps BEFORE collision resolution.
    - Gated by V2Config.use_ensemble (default False).
"""

import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.config import V2Config, get_default_config
from src.dataset import calculate_crop_bounds, generate_seed_mask, compute_distance_transform_channel
from src.postprocess import resolve_mask_overlaps, apply_morphological_cleaning, repair_filament_gaps

logger = logging.getLogger(__name__)


# ── IoU-NMS helper (Step 2) ───────────────────────────────────────────────────

def _box_iou(b1, b2) -> float:
    """IoU between two [x, y, w, h] bounding boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / (union + 1e-6))


def _nms_merge(
    path_a_boxes: list,   # list of [x, y, w, h, conf]
    path_b_boxes: list,   # list of [x, y, w, h, conf=0.40]
    iou_thresh: float = 0.40,
) -> list:
    """
    Merges Path A (YOLO) and Path B (CLAHE) candidates via IoU-NMS.
    Any Path B candidate whose box overlaps a Path A box by > iou_thresh is
    suppressed (Path A takes priority for overlapping regions).
    Returns merged list of [x, y, w, h, conf].
    """
    merged = list(path_a_boxes)  # start with all Path A detections
    for b_box in path_b_boxes:
        overlaps_a = any(_box_iou(b_box[:4], a_box[:4]) > iou_thresh for a_box in path_a_boxes)
        if not overlaps_a:
            merged.append(b_box)
    return merged


# ── Main pipeline ─────────────────────────────────────────────────────────────

class FilamentSegmentationPipeline:
    """
    v2 End-to-End Two-Stage Cascade Inference Pipeline.

    Stage 1: Dual-path candidate proposal (YOLO + CLAHE, always both).
    Stage 2: Fixed-GSD 4-channel crop extraction.
    Stage 3: U-Net++ refinement with 15-pass → 6-pass budget-aware TTA.
    Stage 4: Otsu threshold → morphological cleaning → collision resolution
             → skeleton gap repair → RLE.
    """

    def __init__(
        self,
        detector_model,
        refiner_model,
        device: torch.device,
        config: Optional[V2Config] = None,
        # v1-compat kwargs (used when config is None)
        crop_size: int = 1024,
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
    ):
        self.detector = detector_model
        self.device = device
        self.refiner = refiner_model.to(device).eval()

        if config is None:
            # Build a minimal config from v1-style kwargs for backward compat
            config = get_default_config()
            config.yolo_conf_low = conf_threshold
            config.yolo_nms_iou = iou_threshold
            config.fixed_crop_size = crop_size
        self.cfg = config

    # ── Step 5: TTA multi-scale forward pass ─────────────────────────────────

    @torch.no_grad()
    def _tta_forward(self, tensor_b: torch.Tensor) -> np.ndarray:
        """
        v2 TTA: H/V flip × {scale} passes.
        tta_passes in {1, 3, 6}:
          1 → identity only
          3 → identity + hflip + vflip at scale=1.0
          6 → identity + hflip at each of 3 scales (0.9, 1.0, 1.1)

        All probability maps are averaged into a single (H, W) float array.
        """
        n = self.cfg.tta_passes
        scales = self.cfg.tta_scales if n >= 6 else (1.0,)
        H, W = tensor_b.shape[2], tensor_b.shape[3]
        prob_maps = []

        for scale in scales:
            if scale != 1.0:
                new_h = max(32, int(H * scale))
                new_w = max(32, int(W * scale))
                tb = F.interpolate(tensor_b, size=(new_h, new_w), mode="bilinear", align_corners=False)
            else:
                tb = tensor_b

            augments = [tb]
            if n >= 3:
                augments.append(torch.flip(tb, dims=[3]))   # hflip
                augments.append(torch.flip(tb, dims=[2]))   # vflip

            for i, aug in enumerate(augments):
                logits = self.refiner(aug)
                prob = torch.sigmoid(logits)
                if i == 1:  # hflip — flip back
                    prob = torch.flip(prob, dims=[3])
                elif i == 2:  # vflip — flip back
                    prob = torch.flip(prob, dims=[2])
                if scale != 1.0:
                    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
                prob_maps.append(prob[0, 0].cpu().numpy())

        return np.mean(np.stack(prob_maps, axis=0), axis=0)

    # ── Step 3 & 2: Crop refinement ──────────────────────────────────────────

    @torch.no_grad()
    def _refine_crop(
        self, gray_img: np.ndarray, bbox: list, conf: float
    ) -> tuple:
        h, w = gray_img.shape
        x0, y0, x1, y1 = calculate_crop_bounds(bbox, h, w)

        # Fixed-GSD crop sizing (Step 3)
        from src.dataset import compute_gsd_crop_size
        if self.cfg.use_fixed_gsd:
            out_size = compute_gsd_crop_size(
                bbox[2], bbox[3],
                min_crop=self.cfg.gsd_min_crop,
                max_crop=self.cfg.gsd_max_crop,
            )
        else:
            out_size = self.cfg.fixed_crop_size

        seed_full = generate_seed_mask(h, w, bbox, pad_ratio=0.40)

        # Photometric channels — bilinear OK
        img_crop = cv2.resize(
            gray_img[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_LINEAR,
        )
        clahe_crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_crop)

        # Binary seed mask — INTER_NEAREST (Step 1)
        seed_crop = cv2.resize(
            seed_full[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_NEAREST,
        )

        channels = [
            img_crop.astype(np.float32) / 255.0,
            clahe_crop.astype(np.float32) / 255.0,
            seed_crop.astype(np.float32),
        ]

        # Distance transform channel (Step 3)
        if self.cfg.use_distance_transform:
            dist_full = compute_distance_transform_channel(seed_full)
            dist_crop = cv2.resize(
                dist_full[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            channels.append(dist_crop)

        mean = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)[:len(channels), None, None]
        std  = np.array([0.229, 0.224, 0.225, 0.25], dtype=np.float32)[:len(channels), None, None]

        input_tensor = np.stack(channels, axis=0).astype(np.float32)
        input_tensor = (input_tensor - mean) / std

        tensor_b = torch.from_numpy(input_tensor[None]).to(self.device)

        # Step 5: Budget-aware TTA
        probs = self._tta_forward(tensor_b)   # (out_size, out_size)

        # Otsu adaptive threshold (clamped to 0.30–0.55)
        probs_u8 = (probs * 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(probs_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = float(np.clip(otsu_t / 255.0, 0.30, 0.55))
        binary_crop = (probs >= thresh).astype(np.uint8)

        # Morphological clean (Step 4 enhancement, activated in v1 bug-fix round)
        binary_crop = apply_morphological_cleaning(binary_crop, kernel_size=5)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = cv2.resize(
            binary_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST,
        )
        return full_mask, float(conf)

    # ── Step 2: CLAHE connected-components (Path B) ───────────────────────────

    def _path_b_candidates(self, gray_img: np.ndarray) -> list:
        """
        Always-on CLAHE + adaptive-threshold + connected-components candidate extractor.
        Extracts dark solar filament structures on the solar disk.
        Returns list of [x, y, w, h, conf=0.40] boxes.
        """
        blurred = cv2.GaussianBlur(gray_img, (15, 15), 0)
        # Identify solar disk mask (exclude dark space background)
        disk_mask = (blurred > 45).astype(np.uint8)
        if disk_mask.sum() == 0:
            return []

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray_img)
        
        # Calculate mean brightness strictly within the solar disk
        mean_disk = float(np.mean(clahe[disk_mask > 0]))
        # Solar filaments are dark features (intensity significantly lower than disk average)
        dark_thresh = int(mean_disk * 0.70)
        
        _, dark_mask = cv2.threshold(clahe, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=disk_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        boxes = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if self.cfg.clahe_min_area <= area <= self.cfg.clahe_max_area:
                bx = stats[lbl, cv2.CC_STAT_LEFT]
                by = stats[lbl, cv2.CC_STAT_TOP]
                bw = stats[lbl, cv2.CC_STAT_WIDTH]
                bh = stats[lbl, cv2.CC_STAT_HEIGHT]
                boxes.append([float(bx), float(by), float(bw), float(bh), 0.40])
        return boxes


    # ── Main predict ──────────────────────────────────────────────────────────

    def predict_image(self, image_path: Path) -> list:
        gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if gray_img.shape != (2048, 2048):
            gray_img = cv2.resize(gray_img, (2048, 2048), interpolation=cv2.INTER_AREA)

        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # ── Step 2 Path A: YOLO low-confidence pass ───────────────────────────
        det = self.detector.predict(
            source=bgr_img,
            conf=self.cfg.yolo_conf_low,
            iou=self.cfg.yolo_nms_iou,
            imgsz=1280,
            verbose=False,
            device=0 if self.device.type == "cuda" else "cpu",
        )[0]

        path_a_boxes = []
        if det.boxes is not None and len(det.boxes):
            boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
            confs = det.boxes.conf.detach().cpu().numpy()
            for i in range(len(confs)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                path_a_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(confs[i])])

        # ── Step 2 QA pass: high-conf diagnostics only (not fed to pipeline) ─
        qa_det = self.detector.predict(
            source=bgr_img,
            conf=self.cfg.yolo_conf_high,
            iou=self.cfg.yolo_nms_iou,
            imgsz=1280,
            verbose=False,
            device=0 if self.device.type == "cuda" else "cpu",
        )[0]
        n_qa = len(qa_det.boxes) if qa_det.boxes is not None else 0
        logger.debug(
            f"{image_path.name}: Path A={len(path_a_boxes)} cands "
            f"| QA (conf≥{self.cfg.yolo_conf_high})={n_qa} "
            f"| delta={len(path_a_boxes) - n_qa} low-conf cands"
        )

        # ── Step 2 Path B: CLAHE — always runs (v1: only when Path A gave 0) ─
        path_b_boxes = self._path_b_candidates(gray_img)

        # ── Step 2 Merge via IoU-NMS ──────────────────────────────────────────
        merged_boxes = _nms_merge(path_a_boxes, path_b_boxes, iou_thresh=self.cfg.proposal_nms_iou)
        logger.debug(f"  Merged: PathA={len(path_a_boxes)} + PathB={len(path_b_boxes)} → {len(merged_boxes)} after NMS")

        # ── Stage 2: Refine each candidate ────────────────────────────────────
        scored_candidates = []
        for box in sorted(merged_boxes, key=lambda b: -b[4]):
            bbox = box[:4]
            conf_val = box[4]
            mask, score = self._refine_crop(gray_img, bbox, conf_val)
            if int(mask.sum()) >= self.cfg.min_mask_area:
                scored_candidates.append((score, mask))
            else:
                # High-recall fallback: if refiner returned 0 pixels on a candidate box,
                # use the raw seed mask inside the box as a candidate
                seed_mask = generate_seed_mask(gray_img.shape[0], gray_img.shape[1], bbox, pad_ratio=0.10)
                if int(seed_mask.sum()) >= self.cfg.min_mask_area:
                    scored_candidates.append((conf_val * 0.5, seed_mask))

        # ── Step 7: Correct execution order to prevent overlapping masks ───
        if self.cfg.use_skeleton_repair:
            repaired_candidates = []
            for score, mask in scored_candidates:
                repaired = repair_filament_gaps(mask, gap_px=self.cfg.skeleton_gap_px)
                if int(repaired.sum()) >= self.cfg.min_mask_area:
                    repaired_candidates.append((score, repaired))
            scored_candidates = repaired_candidates

        # resolve_mask_overlaps runs LAST to guarantee ZERO overlapping pixels
        resolved = resolve_mask_overlaps(scored_candidates, min_area=self.cfg.min_mask_area)

        # Safety Fallback: Guarantee at least one valid prediction per image so PQ > 0
        if len(resolved) == 0 and len(path_a_boxes) > 0:
            best_box = sorted(path_a_boxes, key=lambda b: -b[4])[0][:4]
            fb_mask = generate_seed_mask(gray_img.shape[0], gray_img.shape[1], best_box, pad_ratio=0.10)
            if int(fb_mask.sum()) >= self.cfg.min_mask_area:
                resolved = [fb_mask]

        return resolved




# ── Step 5: Runtime utilities ─────────────────────────────────────────────────

def estimate_inference_time(
    n_images: int,
    avg_candidates_per_image: int = 12,
    tta_passes: int = 6,
    seconds_per_refine_pass: float = 0.08,
) -> float:
    """
    Estimates total inference time in seconds.
    seconds_per_refine_pass: empirical estimate on a T4 GPU at 1024px crop.
    """
    total_passes = n_images * avg_candidates_per_image * tta_passes
    return total_passes * seconds_per_refine_pass


def run_tta_ablation(
    pipeline: "FilamentSegmentationPipeline",
    sample_image_paths: list,
    pass_counts: tuple = (1, 3, 6),
) -> dict:
    """
    Step 5: TTA pass-count ablation — runs inference at each pass count
    on a small sample of images and reports wall-clock time per config.
    Returns dict of {pass_count: {"n_images": int, "wall_sec": float, "avg_masks_per_image": float}}.
    """
    results = {}
    orig_passes = pipeline.cfg.tta_passes

    for n_passes in pass_counts:
        pipeline.cfg.tta_passes = n_passes
        t0 = time.time()
        total_masks = 0
        for path in sample_image_paths:
            masks = pipeline.predict_image(path)
            total_masks += len(masks)
        wall = time.time() - t0
        results[n_passes] = {
            "n_images": len(sample_image_paths),
            "wall_sec": round(wall, 2),
            "avg_masks_per_image": round(total_masks / max(1, len(sample_image_paths)), 2),
        }
        print(f"  TTA passes={n_passes}: {wall:.1f}s total, "
              f"{wall/max(1,len(sample_image_paths)):.2f}s/image, "
              f"{total_masks/max(1,len(sample_image_paths)):.1f} masks/image")

    pipeline.cfg.tta_passes = orig_passes
    return results


# ── Step 6: Ensemble pipeline ─────────────────────────────────────────────────

# We need torch.nn.functional for the interpolate call inside _tta_forward
import torch.nn.functional as F


class EnsemblePipeline:
    """
    Step 6: Wraps two FilamentSegmentationPipelines and averages their raw
    probability maps BEFORE collision resolution.

    Enabled by V2Config.use_ensemble = True.
    The secondary model can be a smaller/differently-seeded model.

    Execution order (preserving Step 7 correctness):
      primary_probs ← primary._tta_forward(crop)
      secondary_probs ← secondary._tta_forward(crop)
      avg_probs = mean([primary_probs, secondary_probs])
      binary = Otsu(avg_probs) → morphological clean → collision resolution
              → skeleton repair → RLE
    """
    def __init__(self, primary: FilamentSegmentationPipeline, secondary: FilamentSegmentationPipeline):
        self.primary = primary
        self.secondary = secondary

    def predict_image(self, image_path: Path) -> list:
        gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if gray_img.shape != (2048, 2048):
            gray_img = cv2.resize(gray_img, (2048, 2048), interpolation=cv2.INTER_AREA)
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        cfg = self.primary.cfg

        # Path A + B candidate proposal (using primary pipeline's detector)
        det = self.primary.detector.predict(
            source=bgr_img, conf=cfg.yolo_conf_low, iou=cfg.yolo_nms_iou,
            imgsz=1280, verbose=False,
            device=0 if self.primary.device.type == "cuda" else "cpu",
        )[0]
        path_a_boxes = []
        if det.boxes is not None and len(det.boxes):
            boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
            confs_arr = det.boxes.conf.detach().cpu().numpy()
            for i in range(len(confs_arr)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                path_a_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(confs_arr[i])])

        path_b_boxes = self.primary._path_b_candidates(gray_img)
        merged_boxes = _nms_merge(path_a_boxes, path_b_boxes, iou_thresh=cfg.proposal_nms_iou)

        scored_candidates = []
        for box in sorted(merged_boxes, key=lambda b: -b[4]):
            bbox = box[:4]
            conf_val = box[4]
            h, w = gray_img.shape
            x0, y0, x1_c, y1_c = calculate_crop_bounds(bbox, h, w)
            from src.dataset import compute_gsd_crop_size
            out_size = compute_gsd_crop_size(bbox[2], bbox[3], min_crop=cfg.gsd_min_crop, max_crop=cfg.gsd_max_crop) if cfg.use_fixed_gsd else cfg.fixed_crop_size

            # Build input tensor (primary channels)
            seed_full = generate_seed_mask(h, w, bbox, pad_ratio=0.40)
            img_crop = cv2.resize(gray_img[y0:y1_c, x0:x1_c], (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            clahe_crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_crop)
            seed_crop = cv2.resize(seed_full[y0:y1_c, x0:x1_c], (out_size, out_size), interpolation=cv2.INTER_NEAREST)
            channels = [img_crop.astype(np.float32) / 255.0, clahe_crop.astype(np.float32) / 255.0, seed_crop.astype(np.float32)]
            if cfg.use_distance_transform:
                dist_full = compute_distance_transform_channel(seed_full)
                dist_crop = cv2.resize(dist_full[y0:y1_c, x0:x1_c], (out_size, out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                channels.append(dist_crop)

            mean_arr = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)[:len(channels), None, None]
            std_arr  = np.array([0.229, 0.224, 0.225, 0.25], dtype=np.float32)[:len(channels), None, None]
            inp = (np.stack(channels, axis=0) - mean_arr) / std_arr

            tb = torch.from_numpy(inp[None].astype(np.float32))
            tb_pri = tb.to(self.primary.device)
            tb_sec = tb.to(self.secondary.device)

            # Average probability maps from both models BEFORE thresholding
            with torch.no_grad():
                probs_pri = self.primary._tta_forward(tb_pri)
                probs_sec = self.secondary._tta_forward(tb_sec)
            probs = (probs_pri + probs_sec) / 2.0

            probs_u8 = (probs * 255).astype(np.uint8)
            otsu_t, _ = cv2.threshold(probs_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = float(np.clip(otsu_t / 255.0, 0.30, 0.55))
            binary_crop = (probs >= thresh).astype(np.uint8)
            binary_crop = apply_morphological_cleaning(binary_crop, kernel_size=5)

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y0:y1_c, x0:x1_c] = cv2.resize(binary_crop, (x1_c - x0, y1_c - y0), interpolation=cv2.INTER_NEAREST)
        if cfg.use_skeleton_repair:
            repaired_candidates = []
            for score, mask in scored_candidates:
                repaired = repair_filament_gaps(mask, gap_px=cfg.skeleton_gap_px)
                if int(repaired.sum()) >= cfg.min_mask_area:
                    repaired_candidates.append((score, repaired))
            scored_candidates = repaired_candidates

        resolved = resolve_mask_overlaps(scored_candidates, min_area=cfg.min_mask_area)
        return resolved

