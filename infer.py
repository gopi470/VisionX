"""
VisionX v2 Inference & Submission Generator — Step 5 (runtime budget guard)

v2 changes vs v1:
  - Reads V2Config (CLI args override fields).
  - Step 5: Pre-run inference time estimation with hard-fail if over budget.
  - Step 6: Optional EnsemblePipeline (V2Config.use_ensemble=True).
  - Checkpoint metadata (encoder_name, in_channels) auto-detected from .pt file.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.config import V2Config, get_default_config
from src.utils import encode_rle
from src.models import build_refinement_model
from src.pipeline import FilamentSegmentationPipeline, EnsemblePipeline, estimate_inference_time


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VisionX v2 Inference & Submission Generator")
    p.add_argument("--data_root", default="MAGFiLO_1.0_Kaggle_2026")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--detector_weights", default="yolo11s.pt")
    p.add_argument("--refiner_weights", default=None)
    p.add_argument("--encoder_name", default="convnext_small",
                   help="Must match the backbone used during training")
    p.add_argument("--in_channels", type=int, default=4,
                   help="4 = v2 (distance-transform channel), 3 = v1-compat")
    p.add_argument("--crop_size", type=int, default=1024,
                   help="Fixed crop size (only used when --no_fixed_gsd is set)")
    p.add_argument("--conf_threshold", type=float, default=0.10)
    p.add_argument("--iou_threshold", type=float, default=0.45)
    p.add_argument("--tta_passes", type=int, default=6,
                   help="TTA passes: 1=off, 3=flips only, 6=flips+scales")
    p.add_argument("--time_budget", type=int, default=3600,
                   help="Max allowed inference time in seconds (hard-fail guard)")
    p.add_argument("--no_fixed_gsd", action="store_true",
                   help="Use fixed crop_size instead of GSD-adaptive sizing")
    p.add_argument("--no_boundary_head", action="store_true")
    p.add_argument("--no_skeleton_repair", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Enable detailed per-image live console progress logging")
    p.add_argument("--quiet", action="store_true",
                   help="Disable per-image verbose progress logging")
    # Ensemble
    p.add_argument("--ensemble", action="store_true",
                   help="Enable EnsemblePipeline with a secondary model")
    p.add_argument("--secondary_weights", default="",
                   help="Path to secondary refiner checkpoint (.pt)")
    p.add_argument("--secondary_encoder", default="efficientnet-b4",
                   help="Encoder for secondary model")
    return p.parse_args()


# ── Dataset path resolver (unchanged from v1) ─────────────────────────────────

def find_test_images_dir(data_root: Path) -> Path:
    """Flexible dataset path resolver for Kaggle input structures."""
    search_roots = [data_root, Path("/kaggle/input")]
    for root in search_roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_dir() and p.name in ("test_images", "test"):
                    imgs = (list(p.glob("*.jpeg")) + list(p.glob("*.jpg"))
                            + list(p.glob("*.JPEG")) + list(p.glob("*.JPG")))
                    if imgs:
                        return p
    raise FileNotFoundError(
        f"Could not locate test_images dir in {data_root} or /kaggle/input"
    )


# ── Checkpoint metadata loader ────────────────────────────────────────────────

def _load_refiner(weights_path: str, encoder_name: str, in_channels: int,
                  use_boundary_head: bool, device: torch.device):
    """Loads refiner model, auto-detecting architecture from checkpoint if available."""
    refiner = build_refinement_model(
        encoder_name=encoder_name,
        encoder_weights=None if weights_path else "imagenet",
        in_channels=in_channels,
        use_boundary_head=use_boundary_head,
    )
    if weights_path and Path(weights_path).exists():
        print(f"Loading refiner weights from {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=False)
        # Auto-detect architecture from checkpoint metadata
        if isinstance(state, dict):
            saved_enc = state.get("encoder_name", encoder_name)
            saved_ch = state.get("in_channels", in_channels)
            saved_bh = state.get("use_boundary_head", use_boundary_head)
            if saved_enc != encoder_name or saved_ch != in_channels:
                print(f"  [Note] Checkpoint metadata: encoder={saved_enc}, "
                      f"in_channels={saved_ch}, boundary_head={saved_bh}. "
                      f"Rebuilding model to match checkpoint.")
                refiner = build_refinement_model(
                    encoder_name=saved_enc,
                    encoder_weights=None,
                    in_channels=saved_ch,
                    use_boundary_head=saved_bh,
                )
            model_state = state.get("model", state)
        else:
            model_state = state
        refiner.load_state_dict(model_state, strict=False)
    return refiner


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    verbose = args.verbose and not args.quiet
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_images_dir = find_test_images_dir(data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    raw_paths = (
        list(test_images_dir.glob("*.jpeg"))
        + list(test_images_dir.glob("*.jpg"))
        + list(test_images_dir.glob("*.JPEG"))
        + list(test_images_dir.glob("*.JPG"))
    )
    # Deduplicate while preserving deterministic path sorting
    test_image_paths = sorted(list({p.resolve(): p for p in raw_paths}.values()))
    print(f"Found {len(test_image_paths)} unique test images in {test_images_dir}")

    # ── Step 5: Pre-run budget check ─────────────────────────────────────────
    estimated_sec = estimate_inference_time(
        n_images=len(test_image_paths),
        avg_candidates_per_image=15,
        tta_passes=args.tta_passes,
    )
    print(f"Estimated inference time: {estimated_sec:.0f}s "
          f"(budget: {args.time_budget}s)")
    if estimated_sec > args.time_budget:
        raise RuntimeError(
            f"[Step 5 Budget Guard] Estimated inference time {estimated_sec:.0f}s "
            f"exceeds configured budget {args.time_budget}s. "
            f"Reduce --tta_passes (current: {args.tta_passes}) or increase --time_budget."
        )

    # ── Build V2Config ────────────────────────────────────────────────────────
    cfg = get_default_config()
    cfg.yolo_conf_low = args.conf_threshold
    cfg.yolo_nms_iou = args.iou_threshold
    cfg.tta_passes = args.tta_passes
    cfg.tta_time_budget_seconds = args.time_budget
    cfg.use_fixed_gsd = not args.no_fixed_gsd
    cfg.fixed_crop_size = args.crop_size
    cfg.use_boundary_head = not args.no_boundary_head
    cfg.use_skeleton_repair = not args.no_skeleton_repair
    cfg.use_distance_transform = (args.in_channels == 4)
    cfg.in_channels = args.in_channels
    cfg.use_ensemble = args.ensemble

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device} | TTA passes: {cfg.tta_passes} | "
          f"Encoder: {args.encoder_name} | Channels: {args.in_channels}")

    detector = YOLO(args.detector_weights)

    refiner = _load_refiner(
        args.refiner_weights, args.encoder_name,
        args.in_channels, cfg.use_boundary_head, device,
    )

    pipeline = FilamentSegmentationPipeline(
        detector_model=detector,
        refiner_model=refiner,
        device=device,
        config=cfg,
    )

    # ── Step 6: Ensemble ──────────────────────────────────────────────────────
    if cfg.use_ensemble and args.secondary_weights:
        print(f"Building ensemble with secondary model: {args.secondary_encoder}")
        secondary_refiner = _load_refiner(
            args.secondary_weights, args.secondary_encoder,
            args.in_channels, cfg.use_boundary_head, device,
        )
        secondary_pipeline = FilamentSegmentationPipeline(
            detector_model=detector,
            refiner_model=secondary_refiner,
            device=device,
            config=cfg,
        )
        active_pipeline = EnsemblePipeline(pipeline, secondary_pipeline)
        print("Ensemble pipeline active.")
    else:
        active_pipeline = pipeline

    # ── Inference loop ────────────────────────────────────────────────────────
    submission_rows = []
    t_start = time.time()
    total_imgs = len(test_image_paths)
    print(f"\n🚀 Running Inference on {total_imgs} images...\n", flush=True)

    for idx, path in enumerate(test_image_paths, start=1):
        if verbose:
            print(f"[{idx}/{total_imgs}] 🖼️ Image: {path.name}", flush=True)
        image_stem = path.stem
        predicted_masks = active_pipeline.predict_image(path, verbose=verbose)
        for i, mask in enumerate(predicted_masks, start=1):
            submission_rows.append({
                "filament_id": f"{image_stem}_{i}",
                "segmentation_rle": encode_rle(mask),
            })
        if verbose:
            print(f"   💾 Recorded {len(predicted_masks)} filament rows.\n", flush=True)

    t_elapsed = time.time() - t_start
    print(f"\nInference complete: {t_elapsed:.1f}s total "
          f"({t_elapsed / max(1, len(test_image_paths)):.2f}s/image)")

    sub_df = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    submission_path = output_dir / "submission.csv"
    sub_df.to_csv(submission_path, index=False)

    n_images_covered = (
        sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique()
        if len(sub_df) > 0 else 0
    )
    print(f"Saved: {submission_path}")
    print(f"Total filament rows: {len(sub_df)} | Images covered: {n_images_covered}")


if __name__ == "__main__":
    main()
