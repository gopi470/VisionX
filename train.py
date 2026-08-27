"""
VisionX v2 Training Entrypoint — Step 4

v2 changes vs v1:
  - Reads V2Config (CLI overrides specific fields).
  - Composite loss from src/losses.py (BCE + Dice + BoundaryLoss by default).
  - EMA (exponential moving average) of model weights.
  - SWA (stochastic weight averaging) over final swa_anneal_epochs epochs.
  - Hard-negative mining starting at epoch hard_neg_start_epoch:
      After each qualifying epoch, evaluate training crops, identify those with
      predicted IoU < hard_neg_iou_thresh, and oversample them in the next epoch.
  - RefineDataset v2: 4-channel input, fixed-GSD crops, albumentations pipeline.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.config import V2Config, get_default_config
from src.utils import load_coco_dataset, get_group_kfold_split
from src.dataset import RefineDataset
from src.models import build_refinement_model
from src.losses import build_loss
from src.metrics import compute_iou


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VisionX v2 Training")
    p.add_argument("--data_root", default="MAGFiLO_1.0_Kaggle_2026")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    # V2Config overrides
    p.add_argument("--encoder_name", default="tu-convnext_small",
                   help="Encoder backbone. v2 default: tu-convnext_small")
    p.add_argument("--no_boundary_head", action="store_true",
                   help="Disable boundary refinement head (v1-compat mode)")
    p.add_argument("--no_ema", action="store_true", help="Disable EMA")
    p.add_argument("--no_swa", action="store_true", help="Disable SWA")
    p.add_argument("--no_fixed_gsd", action="store_true",
                   help="Use fixed 1024px crops instead of GSD-adaptive sizing")
    p.add_argument("--use_tversky", action="store_true",
                   help="Use TverskyLoss instead of composite BCE+Dice+Boundary")
    p.add_argument("--in_channels", type=int, default=4,
                   help="Model input channels (4=v2 with dist-transform, 3=v1-compat)")
    return p.parse_args()


# ── EMA helper ────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Manual exponential moving average of model parameters.
    Avoids external dependency on torch-ema.
    Usage:
        ema = EMAModel(model, decay=0.999)
        after each optimizer step: ema.update(model)
        to evaluate: ema.apply_shadow(); eval...; ema.restore()
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ── Hard-negative mining ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_hard_neg_weights(
    model: nn.Module,
    dataset: RefineDataset,
    device: torch.device,
    iou_thresh: float = 0.30,
    oversample: int = 3,
) -> list:
    """
    Runs a quick forward pass over the training dataset, identifies samples
    where predicted mask IoU < iou_thresh, and returns a weight list for
    WeightedRandomSampler (oversampling hard negatives by `oversample` factor).
    """
    model.eval()
    weights = [1.0] * len(dataset)

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    idx = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        for b in range(x.shape[0]):
            pred_np = preds[b, 0].cpu().numpy().astype(np.uint8)
            gt_np = y[b, 0].cpu().numpy().astype(np.uint8)
            iou = compute_iou(pred_np, gt_np)
            if iou < iou_thresh:
                weights[idx] = float(oversample)
            idx += 1
            if idx >= len(dataset):
                break
        if idx >= len(dataset):
            break

    model.train()
    return weights


# ── Annotation list ───────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build config from defaults + CLI overrides
    cfg = get_default_config()
    cfg.encoder_name = args.encoder_name
    cfg.use_boundary_head = not args.no_boundary_head
    cfg.use_ema = not args.no_ema
    cfg.use_swa = not args.no_swa
    cfg.use_fixed_gsd = not args.no_fixed_gsd
    cfg.use_tversky_loss = args.use_tversky
    cfg.in_channels = args.in_channels
    cfg.use_distance_transform = (args.in_channels == 4)

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
    print(f"Split: Train={len(train_samples)} imgs ({len(train_anns)} crops) | "
          f"Val={len(val_samples)} imgs ({len(val_anns)} crops)")

    # v2 datasets with GSD crops and distance transform
    ds_kwargs = dict(
        use_fixed_gsd=cfg.use_fixed_gsd,
        gsd_min_crop=cfg.gsd_min_crop,
        gsd_max_crop=cfg.gsd_max_crop,
        use_distance_transform=cfg.use_distance_transform,
    )
    train_ds = RefineDataset(train_anns, augment=True, **ds_kwargs)
    val_ds = RefineDataset(val_anns, augment=False, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building model: {cfg.encoder_name} | in_channels={cfg.in_channels} | "
          f"boundary_head={cfg.use_boundary_head} | device={device}")

    model = build_refinement_model(
        encoder_name=cfg.encoder_name,
        encoder_weights="imagenet",
        in_channels=cfg.in_channels,
        use_boundary_head=cfg.use_boundary_head,
    ).to(device)

    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Cosine annealing base scheduler
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # EMA
    ema = EMAModel(model, decay=cfg.ema_decay) if cfg.use_ema else None

    # SWA (over the final swa_anneal_epochs epochs)
    swa_model = torch.optim.swa_utils.AveragedModel(model) if cfg.use_swa else None
    swa_scheduler = None
    if cfg.use_swa:
        swa_start = max(1, args.epochs - cfg.swa_anneal_epochs)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, swa_lr=1e-5, anneal_epochs=cfg.swa_anneal_epochs,
        )
        cfg.swa_start_epoch = swa_start
        print(f"SWA will start at epoch {swa_start}")

    best_val_loss = float("inf")
    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # ── Hard-negative mining: rebuild sampler from epoch hard_neg_start_epoch ─
        if epoch == cfg.hard_neg_start_epoch:
            print(f"[Epoch {epoch}] Starting hard-negative mining (IoU<{cfg.hard_neg_iou_thresh})...")
        if epoch >= cfg.hard_neg_start_epoch and (epoch - cfg.hard_neg_start_epoch) % 5 == 0:
            weights = compute_hard_neg_weights(
                model, train_ds, device,
                iou_thresh=cfg.hard_neg_iou_thresh,
                oversample=cfg.hard_neg_oversample_factor,
            )
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=sampler,
                num_workers=2, pin_memory=True, drop_last=True,
            )
            n_hard = sum(1 for w in weights if w > 1.0)
            print(f"  Hard-neg sampler rebuilt: {n_hard}/{len(weights)} oversampled crops")

        # ── Train ─────────────────────────────────────────────────────────────
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
            if ema:
                ema.update(model)
            train_losses.append(loss.item())

        # ── SWA update ────────────────────────────────────────────────────────
        if cfg.use_swa and swa_model and epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            base_scheduler.step()

        # ── Validate (using EMA weights if enabled) ───────────────────────────
        if ema:
            ema.apply_shadow(model)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        if ema:
            ema.restore(model)

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d}: Train={avg_train:.4f} | Val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = output_dir / "refiner_best.pt"
            save_model = swa_model if (cfg.use_swa and swa_model and epoch >= cfg.swa_start_epoch) else model
            torch.save({
                "model": save_model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
                "encoder_name": cfg.encoder_name,
                "in_channels": cfg.in_channels,
                "use_boundary_head": cfg.use_boundary_head,
            }, ckpt_path)
            print(f"  --> Saved best checkpoint: {ckpt_path}")

    # ── Final SWA BN update ───────────────────────────────────────────────────
    if cfg.use_swa and swa_model:
        print("Updating SWA batch-norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({
            "model": swa_model.state_dict(),
            "epoch": args.epochs,
            "val_loss": best_val_loss,
            "encoder_name": cfg.encoder_name,
            "in_channels": cfg.in_channels,
            "use_boundary_head": cfg.use_boundary_head,
        }, output_dir / "refiner_swa_final.pt")
        print(f"  --> SWA model saved: {output_dir / 'refiner_swa_final.pt'}")


if __name__ == "__main__":
    main()
