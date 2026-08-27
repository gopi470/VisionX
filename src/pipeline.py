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
    Stage 2: Cropped Refinement UNet
    """
    def __init__(self, detector_model, refiner_model, device: torch.device, crop_size: int = 512, conf_threshold: float = 0.28, iou_threshold: float = 0.50):
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

        seed_full = generate_seed_mask(h, w, bbox, pad_ratio=0.40)

        img_crop = cv2.resize(gray_img[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        seed_crop = cv2.resize(seed_full[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_crop = clahe_engine.apply(img_crop)

        input_tensor = np.stack([img_crop, clahe_crop, seed_crop * 255], axis=0).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        input_tensor = (input_tensor - mean) / std

        tensor_b = torch.from_numpy(input_tensor[None]).to(self.device)
        logits = self.refiner(tensor_b)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        binary_crop = (probs >= 0.50).astype(np.uint8)

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

        # Detector proposal stage
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
            boxes_xywh = detection_results.boxes.xywh.detach().cpu().numpy()
            boxes_xyxy = detection_results.boxes.xyxy.detach().cpu().numpy()
            confs = detection_results.boxes.conf.detach().cpu().numpy()

            for i in np.argsort(-confs):
                x1, y1, x2, y2 = boxes_xyxy[i]
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                bbox = [x1, y1, bw, bh]

                mask, score = self._refine_crop(gray_img, bbox, float(confs[i]))
                if int(mask.sum()) >= 150:
                    scored_candidates.append((score, mask))

        return resolve_mask_overlaps(scored_candidates)
