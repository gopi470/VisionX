"""
VisionX Pipeline Module
Coordinates Detector proposal, Cropped Refinement Network inference, and Overlap Resolution.
"""

from pathlib import Path
import cv2
import numpy as np
import torch

from src.dataset import calculate_crop_bounds, generate_seed_mask
from src.postprocess import resolve_mask_overlaps


class FilamentSegmentationPipeline:
    """
    End-to-End Two-Stage Cascade Inference Pipeline.
    Stage 1: Object Detector (YOLO or Region Proposal)
    Stage 2: Cropped Refinement Network (U-Net++ ConvNeXt Large at 1024x1024 resolution)
    """
    def __init__(self, detector_model, refiner_model, device: torch.device, crop_size: int = 1024, conf_threshold: float = 0.15, iou_threshold: float = 0.45):
        self.detector = detector_model
        self.refiner = refiner_model.to(device).eval()
        self.device = device
        self.crop_size = crop_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold


    @torch.no_grad()
    def _refine_crop(self, gray_img: np.ndarray, bbox: list[float], conf: float) -> tuple[np.ndarray, float]:
        h, w = gray_img.shape
        x0, y0, x1, y1 = calculate_crop_bounds(bbox, h, w)

        # Multi-Pass Box Jitter TTA: Shift candidate seed mask in 5 directions for higher precision
        shifts = [(0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)]
        prob_maps = []

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        for dx, dy in shifts:
            bbox_shifted = [bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]]
            seed_full = generate_seed_mask(h, w, bbox_shifted, pad_ratio=0.40)

            img_crop = cv2.resize(gray_img[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            seed_crop = cv2.resize(seed_full[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

            clahe_engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_crop = clahe_engine.apply(img_crop)

            input_tensor = np.stack([img_crop, clahe_crop, seed_crop * 255], axis=0).astype(np.float32) / 255.0
            input_tensor = (input_tensor - mean) / std

            tensor_b = torch.from_numpy(input_tensor[None]).to(self.device)

            # Horizontal & Vertical Flip TTA passes per shift
            logits_orig = self.refiner(tensor_b)
            logits_hflip = self.refiner(torch.flip(tensor_b, dims=[3]))
            logits_vflip = self.refiner(torch.flip(tensor_b, dims=[2]))

            p_orig = torch.sigmoid(logits_orig)
            p_hflip = torch.flip(torch.sigmoid(logits_hflip), dims=[3])
            p_vflip = torch.flip(torch.sigmoid(logits_vflip), dims=[2])

            p_avg = (p_orig + p_hflip + p_vflip) / 3.0
            prob_maps.append(p_avg[0, 0].cpu().numpy())

        # Average ensemble across all 5 shifted Box Jitter TTA passes
        probs = float(np.mean(prob_maps, axis=0)) if isinstance(np.mean(prob_maps, axis=0), float) else np.mean(prob_maps, axis=0)

        binary_crop = (probs >= 0.40).astype(np.uint8)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = cv2.resize(binary_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

        return full_mask, float(conf)



    def predict_image(self, image_path: Path) -> list[np.ndarray]:
        gray_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        if gray_img.shape != (2048, 2048):
            gray_img = cv2.resize(gray_img, (2048, 2048), interpolation=cv2.INTER_AREA)

        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Stage 1: Detector proposal stage
        detection_results = self.detector.predict(
            source=bgr_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=1280,
            verbose=False,
            device=0 if self.device.type == "cuda" else "cpu",
        )[0]

        scored_candidates = []
        if detection_results.boxes is not None and len(detection_results.boxes):
            boxes_xyxy = detection_results.boxes.xyxy.detach().cpu().numpy()
            confs = detection_results.boxes.conf.detach().cpu().numpy()

            for i in np.argsort(-confs):
                x1, y1, x2, y2 = boxes_xyxy[i]
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                bbox = [x1, y1, bw, bh]

                mask, score = self._refine_crop(gray_img, bbox, float(confs[i]))
                if int(mask.sum()) >= 150:
                    scored_candidates.append((score, mask))

        # Fallback Candidate Extraction (Adaptive Local Dark Region Thresholding)
        # Ensures zero empty predictions across test set if general detector misses dark solar filaments
        if len(scored_candidates) == 0:
            blurred = cv2.GaussianBlur(gray_img, (15, 15), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blurred)
            # Threshold dark solar features relative to mean solar disk brightness
            mean_val = np.mean(clahe)
            _, dark_mask = cv2.threshold(clahe, int(mean_val * 0.65), 255, cv2.THRESH_BINARY_INV)
            
            # Mask out outer space background (outside solar disk)
            disc_mask = (blurred > 30).astype(np.uint8)
            dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=disc_mask)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if 200 <= area <= 50000:
                    component_mask = (labels == label_idx).astype(np.uint8)
                    scored_candidates.append((0.40, component_mask))

        return resolve_mask_overlaps(scored_candidates)

