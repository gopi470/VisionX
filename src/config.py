"""
VisionX V2Config — Central config dataclass for all pipeline toggles.
Every behaviour change in v2 is gated behind a flag here so v1 behaviour
can be recovered for ablation (set a flag back to its v1 default).
"""

from dataclasses import dataclass, field


@dataclass
class V2Config:
    # ── Step 2: Dual-path candidate proposal ──────────────────────────────────
    # Path A: YOLO low-confidence threshold (was 0.15 in v1)
    yolo_conf_low: float = 0.10
    # Path B: CLAHE always runs (v1: only when YOLO returns 0 candidates)
    clahe_path_always: bool = True
    # QA-only high-precision pass — logged but NOT fed to the main pipeline
    yolo_conf_high: float = 0.35
    # IoU cutoff for merging Path A ∪ Path B via NMS
    proposal_nms_iou: float = 0.40
    # YOLO NMS IoU threshold
    yolo_nms_iou: float = 0.45
    # CLAHE area filter — skip connected components outside this range (pixels)
    clahe_min_area: int = 100
    clahe_max_area: int = 150000
    # CLAHE threshold factor relative to mean solar disk brightness (was 0.70 hardcoded, now 0.85 adaptive)
    clahe_thresh_factor: float = 0.85

    # ── Step 3: Fixed-GSD crop extraction ────────────────────────────────────
    # Toggle fixed-GSD variable-size crops (v1 used fixed 1024)
    use_fixed_gsd: bool = True
    gsd_min_crop: int = 768
    gsd_max_crop: int = 1280
    # 4th input channel: normalized distance transform of seed mask
    use_distance_transform: bool = True

    # ── Step 4: Architecture & training ──────────────────────────────────────
    # Encoder backbone — "convnext_small" is the v2 default;
    # use "hrnet_w32" if timm supports it in the running environment
    encoder_name: str = "convnext_small"
    # Lightweight boundary-sharpening conv head on top of decoder
    use_boundary_head: bool = True
    # Loss weights (must sum to 1.0)
    bce_weight: float = 0.30
    dice_weight: float = 0.40
    boundary_weight: float = 0.30
    # Tversky loss as an alternate option (False = use default composite above)
    use_tversky_loss: bool = False
    tversky_alpha: float = 0.3   # FP penalty
    tversky_beta: float = 0.7    # FN penalty
    # EMA of model weights
    use_ema: bool = True
    ema_decay: float = 0.999
    # Stochastic Weight Averaging
    use_swa: bool = True
    swa_start_epoch: int = 30
    swa_anneal_epochs: int = 10
    # Hard-negative mining: oversample crops where predicted IoU < threshold
    hard_neg_start_epoch: int = 20
    hard_neg_iou_thresh: float = 0.30
    hard_neg_oversample_factor: int = 3

    # ── Step 5: Budget-aware TTA ──────────────────────────────────────────────
    # v2 TTA: H/V flip × {0.9, 1.0, 1.1} scale = 6 passes
    # Set tta_passes=1 to disable TTA (fastest, v1-equivalent baseline)
    tta_passes: int = 6
    tta_scales: tuple = (0.9, 1.0, 1.1)
    # Hard-fail if estimated total inference time exceeds this (seconds)
    tta_time_budget_seconds: int = 3600

    # ── Step 6: Ensemble ─────────────────────────────────────────────────────
    use_ensemble: bool = False
    secondary_encoder_name: str = "efficientnet-b4"
    secondary_model_weights: str = ""

    # ── Step 7: Post-processing ───────────────────────────────────────────────
    use_skeleton_repair: bool = True
    skeleton_gap_px: int = 4
    # Min pixel area for final accepted mask (after all post-processing)
    min_mask_area: int = 150

    # ── General ───────────────────────────────────────────────────────────────
    # Crop resolution used when use_fixed_gsd=False (v1 compat)
    fixed_crop_size: int = 1024
    # Number of input channels (3 = v1, 4 = v2 with distance transform)
    in_channels: int = 4


# Singleton convenience accessor
def get_default_config() -> V2Config:
    return V2Config()


def v1_compat_config() -> V2Config:
    """Returns a config that reproduces v1 pipeline behaviour for ablation."""
    return V2Config(
        clahe_path_always=False,
        use_fixed_gsd=False,
        use_distance_transform=False,
        in_channels=3,
        use_boundary_head=False,
        use_ema=False,
        use_swa=False,
        hard_neg_start_epoch=9999,
        tta_passes=15,          # v1 used 5-jitter × 3-flip
        use_ensemble=False,
        use_skeleton_repair=False,
        encoder_name="tu-convnext_large",
    )
