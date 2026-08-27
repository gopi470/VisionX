"""
VisionX v2 Package Initialization
"""

from src.utils import encode_rle, decode_rle, polygon_to_mask, load_coco_dataset, get_group_kfold_split
from src.metrics import (
    compute_panoptic_quality, evaluate_dataset_pq, compute_iou,
    evaluate_stratified_pq, measure_filament_width, classify_width_bucket,
    print_stratified_pq_report,
)
from src.postprocess import resolve_mask_overlaps, apply_morphological_cleaning, repair_filament_gaps
from src.dataset import RefineDataset, calculate_crop_bounds, generate_seed_mask, compute_gsd_crop_size
from src.models import build_refinement_model, CombinedBCEDiceLoss, BoundaryRefinementHead
from src.pipeline import FilamentSegmentationPipeline, EnsemblePipeline, estimate_inference_time
from src.losses import BoundaryLoss, TverskyLoss, CompositeLoss, build_loss
from src.config import V2Config, get_default_config, v1_compat_config

__all__ = [
    # utils
    "encode_rle", "decode_rle", "polygon_to_mask", "load_coco_dataset", "get_group_kfold_split",
    # metrics
    "compute_panoptic_quality", "evaluate_dataset_pq", "compute_iou",
    "evaluate_stratified_pq", "measure_filament_width", "classify_width_bucket",
    "print_stratified_pq_report",
    # postprocess
    "resolve_mask_overlaps", "apply_morphological_cleaning", "repair_filament_gaps",
    # dataset
    "RefineDataset", "calculate_crop_bounds", "generate_seed_mask", "compute_gsd_crop_size",
    # models
    "build_refinement_model", "CombinedBCEDiceLoss", "BoundaryRefinementHead",
    # pipeline
    "FilamentSegmentationPipeline", "EnsemblePipeline", "estimate_inference_time",
    # losses
    "BoundaryLoss", "TverskyLoss", "CompositeLoss", "build_loss",
    # config
    "V2Config", "get_default_config", "v1_compat_config",
]
