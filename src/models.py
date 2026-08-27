"""
VisionX v2 Neural Network Architectures & Loss Functions — Step 3 & 4

v2 changes:
  - Step 3: in_channels=4 default; mean-initialise 4th conv channel from existing 3.
  - Step 4: ConvNeXt-Small default (vs Large in v1 — fits at 1024px on T4 GPU).
            Optional lightweight BoundaryRefinementHead on decoder output.
  - build_loss() moved to src/losses.py; kept re-exported here for back-compat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ── Loss (kept for back-compat; canonical implementations in src/losses.py) ───

class CombinedBCEDiceLoss(nn.Module):
    """v1 combined BCE + Soft Dice loss — retained for checkpoint compatibility."""
    def __init__(self, bce_weight: float = 0.45, dice_weight: float = 0.55, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ── v2 Step 4: Lightweight Boundary Refinement Head ──────────────────────────

class BoundaryRefinementHead(nn.Module):
    """
    Lightweight 3-layer conv block appended after the UNet++ decoder output.
    Sharpens boundary predictions without changing decoder topology.
    Configurable via V2Config.use_boundary_head.

    Architecture:
      Conv3x3(in→in, BN, ReLU) → Conv3x3(in→in//2, BN, ReLU) → Conv1x1(in//2→1)
    The final logit is added residually to the original decoder logit for stability.
    """
    def __init__(self, in_channels: int = 16):
        super().__init__()
        mid = max(8, in_channels // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(mid, 1, 1, bias=True)

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        x = self.conv1(features)
        x = self.conv2(x)
        refinement = self.out(x)
        return base_logits + refinement  # residual addition for training stability


# ── v2 Step 3 & 4: Model builder ─────────────────────────────────────────────

def _mean_init_extra_channel(model: nn.Module, old_channels: int = 3) -> None:
    """
    When upgrading from 3→4 input channels, initialize the 4th conv kernel
    as the mean of the existing 3 channels instead of random init.
    This preserves learned magnitude of pretrained features.
    Searches for the first Conv2d with in_channels == old_channels+1.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == old_channels + 1:
            with torch.no_grad():
                existing = module.weight[:, :old_channels, :, :]   # (out, 3, kH, kW)
                new_ch_weight = existing.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
                module.weight[:, old_channels:, :, :] = new_ch_weight
            print(f"  [v2] Mean-initialised 4th input channel weights in '{name}'")
            break


class RefinementModel(nn.Module):
    """
    Wraps SMP UNet++ with an optional BoundaryRefinementHead.
    Exposes a single forward() that returns final logits.
    """
    def __init__(self, base_model: nn.Module, use_boundary_head: bool = True):
        super().__init__()
        self.base = base_model
        self.use_boundary_head = use_boundary_head
        if use_boundary_head:
            # Boundary head operates on the raw decoder feature map (16ch SMP output)
            self.boundary_head = BoundaryRefinementHead(in_channels=16)
        else:
            self.boundary_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_boundary_head and self.boundary_head is not None:
            features = self.base.encoder(x)
            decoder_output = self.base.decoder(features)          # (B, 16, H, W)
            base_logits = self.base.segmentation_head(decoder_output)  # (B, 1, H, W)
            return self.boundary_head(decoder_output, base_logits)
        return self.base(x)



def build_refinement_model(
    encoder_name: str = "convnext_small",
    encoder_weights: str = "imagenet",
    in_channels: int = 4,
    use_boundary_head: bool = True,
) -> nn.Module:
    """
    Builds the v2 refinement model:
      - UNet / UNet++ decoder
      - ConvNeXt-Small or ResNet34 backbone
      - 4-channel input with mean-initialised extra channel
    """
    def _build(enc_name: str):
        # SMP Unet has robust channel matching across timm ConvNeXt backbones
        return smp.Unet(
            encoder_name=enc_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    try:
        base = _build(encoder_name)
    except Exception as e:
        print(f"  [v2] Encoder '{encoder_name}' failed ({e}), falling back to resnet34")
        base = _build("resnet34")

    if in_channels == 4 and encoder_weights is not None:
        _mean_init_extra_channel(base, old_channels=3)

    return base

