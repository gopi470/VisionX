"""
VisionX Model Training Entrypoint
Trains the Refinement U-Net Model on crop samples with mixed precision (AMP).
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_coco_dataset, get_group_kfold_split
from src.dataset import RefineDataset
from src.models import build_refinement_model, CombinedBCEDiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train VisionX Refinement Network")
    parser.add_argument("--data_root", type=str, default="MAGFiLO_1.0_Kaggle_2026", help="Path to competition dataset root")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (extended fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training at 1024x1024 high resolution")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop resolution (1024x1024)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()



def build_annotation_list(samples):
    ann_list = []
    for s in samples:
        for ann in s["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            if not ann.get("segmentation") and not ann.get("bbox"):
                continue
            ann_list.append((s, ann))
    return ann_list


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    images_dir = data_root / "train" / "train_images"

    if not json_path.exists():
        print(f"Dataset JSON not found at {json_path}. Please verify path.")
        return

    print("Loading COCO dataset...")
    samples = load_coco_dataset(json_path, images_dir)
    train_samples, val_samples = get_group_kfold_split(samples, val_ratio=0.15, seed=args.seed)

    train_anns = build_annotation_list(train_samples)
    val_anns = build_annotation_list(val_samples)

    print(f"Dataset split: Train samples={len(train_samples)} ({len(train_anns)} crops) | Val samples={len(val_samples)} ({len(val_anns)} crops)")

    train_ds = RefineDataset(train_anns, crop_size=args.crop_size, augment=True)
    val_ds = RefineDataset(val_anns, crop_size=args.crop_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_refinement_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3).to(device)

    criterion = CombinedBCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_loss = float("inf")
    print(f"Starting training for {args.epochs} epochs on {device}...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:02d} Summary: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = output_dir / "refiner_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": avg_val_loss}, ckpt_path)
            print(f"--> Saved best model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
