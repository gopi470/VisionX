"""
VisionX Neural Network Architectures & Loss Functions
Supports SMP UNet / DeepLabV3+ with custom loss combining BCE & Soft Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined BCE and Soft Dice Loss for segmentation boundary optimization.
    """
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


def build_refinement_model(encoder_name: str = "tu-convnext_large", encoder_weights: str = "imagenet", in_channels: int = 3) -> nn.Module:
    """
    Constructs a SOTA U-Net++ (Nested U-Net) architecture with heavy ConvNeXt Large / Swin backbone
    for maximum feature extraction capability and boundary precision.
    """
    try:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    except Exception:
        # Fallback to EfficientNet-B7 if timm convnext weights are loading offline
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    return model


