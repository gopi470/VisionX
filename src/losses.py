"""
VisionX v2 Loss Functions
- BoundaryLoss: distance-transform-weighted BCE on boundary pixels
- TverskyLoss: parameterized FP/FN penalty variant of Dice
- build_loss(): factory function returning composite loss from V2Config
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import V2Config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batch_distance_weight_map(targets: torch.Tensor) -> torch.Tensor:
    """
    Computes a per-pixel boundary weight map for a batch of binary masks.
    Pixels near the boundary (detected via Sobel in target space) receive
    higher weight. Fully vectorized on CPU then moved to target device.
    Returns tensor of shape (B, 1, H, W), values in [1.0, w_max].
    """
    device = targets.device
    B, _, H, W = targets.shape
    weight_maps = []
    for b in range(B):
        mask_np = targets[b, 0].cpu().numpy().astype(np.uint8)
        # Detect boundary pixels (Sobel magnitude > 0)
        gx = cv2.Sobel(mask_np, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mask_np, cv2.CV_32F, 0, 1, ksize=3)
        boundary = (np.sqrt(gx ** 2 + gy ** 2) > 0.5).astype(np.float32)
        # Distance transform from non-boundary — invert so boundary = far = low weight
        # Actually we want pixels NEAR boundary to have higher weight:
        dist = cv2.distanceTransform((1 - boundary).astype(np.uint8), cv2.DIST_L2, 3)
        # Normalize: boundary pixels → weight 4.0, far interior → 1.0
        dist_norm = dist / (dist.max() + 1e-6)
        weight = 1.0 + 3.0 * (1.0 - dist_norm)
        weight_maps.append(torch.from_numpy(weight).unsqueeze(0))
    return torch.stack(weight_maps, dim=0).to(device)  # (B, 1, H, W)


# ── Loss modules ──────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Distance-transform-weighted BCE loss that places 4× more weight on
    boundary pixels than interior pixels, forcing sharper filament edges.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = _batch_distance_weight_map(targets)          # (B,1,H,W)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce * weight).mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss: generalised Dice where alpha penalises FP and beta penalises FN.
    Set alpha < beta to focus on recall (recommended for thin filaments).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


class SoftDiceLoss(nn.Module):
    """Standard soft Dice loss for segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return dice


class CompositeLoss(nn.Module):
    """
    Composite loss: w_bce * BCE + w_dice * Dice + w_boundary * BoundaryLoss
    All weights should sum to 1.0.
    """
    def __init__(self, bce_weight: float = 0.30, dice_weight: float = 0.40, boundary_weight: float = 0.30):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.boundary_w = boundary_weight
        self.dice_loss = SoftDiceLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        return self.bce_w * bce + self.dice_w * dice + self.boundary_w * boundary


# ── Factory ───────────────────────────────────────────────────────────────────

def build_loss(config: "V2Config") -> nn.Module:
    """
    Factory: returns the correct loss module based on V2Config.
    - config.use_tversky_loss = True  → TverskyLoss (FP/FN parameterised)
    - default                         → CompositeLoss (BCE + Dice + Boundary)
    """
    if config.use_tversky_loss:
        return TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
    return CompositeLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        boundary_weight=config.boundary_weight,
    )
