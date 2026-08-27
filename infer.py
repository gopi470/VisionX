"""
VisionX Kaggle Submission & Test Set Inference Entrypoint
Runs complete two-stage pipeline on test set and formats submission CSV.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm

from src.utils import encode_rle
from src.models import build_refinement_model
from src.pipeline import FilamentSegmentationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="VisionX Test Set Inference & Submission Generator")
    parser.add_argument("--data_root", type=str, default="MAGFiLO_1.0_Kaggle_2026", help="Path to competition dataset root")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for submission.csv")
    parser.add_argument("--detector_weights", type=str, default="yolov8n.pt", help="Path or name of detector weights")
    parser.add_argument("--refiner_weights", type=str, default=None, help="Path to refiner weights (.pt)")
    parser.add_argument("--conf_threshold", type=float, default=0.28, help="Detector confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.50, help="Detector NMS IoU threshold")
    return parser.parse_args()


def find_test_images_dir(data_root: Path) -> Path:
    """Flexible dataset path resolver for Kaggle input structures."""
    search_roots = [data_root, Path("/kaggle/input")]
    
    for root in search_roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_dir() and p.name in ("test_images", "test"):
                    imgs = list(p.glob("*.jpeg")) + list(p.glob("*.jpg")) + list(p.glob("*.JPEG")) + list(p.glob("*.JPG"))
                    if len(imgs) > 0:
                        return p

    raise FileNotFoundError(f"Could not locate test_images directory containing images in {data_root} or /kaggle/input")




def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_images_dir = find_test_images_dir(data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    test_image_paths = sorted(list(test_images_dir.glob("*.jpeg"))) + sorted(list(test_images_dir.glob("*.jpg")))
    print(f"Found {len(test_image_paths)} test set images in {test_images_dir}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on device: {device}")

    detector = YOLO(args.detector_weights)

    refiner = build_refinement_model(encoder_name="resnet34", encoder_weights=None if args.refiner_weights else "imagenet", in_channels=3)
    if args.refiner_weights and Path(args.refiner_weights).exists():
        print(f"Loading refiner weights from {args.refiner_weights}")
        state = torch.load(args.refiner_weights, map_location=device, weights_only=False)
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        refiner.load_state_dict(model_state)

    pipeline = FilamentSegmentationPipeline(
        detector_model=detector,
        refiner_model=refiner,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )

    submission_rows = []
    print("Running inference across test set...")
    for path in tqdm(test_image_paths, desc="Inference"):
        image_stem = path.stem
        predicted_masks = pipeline.predict_image(path)

        for i, mask in enumerate(predicted_masks, start=1):
            filament_id = f"{image_stem}_{i}"
            rle_str = encode_rle(mask)
            submission_rows.append({
                "filament_id": filament_id,
                "segmentation_rle": rle_str,
            })

    sub_df = pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"])
    submission_path = output_dir / "submission.csv"
    sub_df.to_csv(submission_path, index=False)

    num_images_predicted = sub_df["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).nunique() if len(sub_df) > 0 else 0
    print(f"\nSuccessfully generated {submission_path}")
    print(f"Total Filament Rows: {len(sub_df)} | Unique Images Covered: {num_images_predicted}")


if __name__ == "__main__":
    main()
