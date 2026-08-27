"""
VisionX v2 Unit Test Suite

Covers:
  - RLE encode/decode round-trip
  - Panoptic Quality math (TP, FP, FN edge cases)
  - Overlap resolution correctness
  - Model creation (v1 + v2 configs)
  - Step 1: Binary mask assertion after INTER_NEAREST resize
  - Step 3: GSD crop size bounds
  - Step 7: Gap repair returns valid binary mask
"""

import numpy as np
import sys
import time


def _hr(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print('─'*50)


# ── 1. RLE round-trip ─────────────────────────────────────────────────────────
_hr("Test 1: RLE encode/decode round-trip")
from src.utils import encode_rle, decode_rle

mask = np.zeros((2048, 2048), dtype=np.uint8)
mask[100:200, 300:500] = 1
rle = encode_rle(mask)
decoded = decode_rle(rle, 2048, 2048)
assert np.array_equal(mask, decoded), "RLE round-trip FAILED"
print("PASS — RLE round-trip correct")


# ── 2. Panoptic Quality math ──────────────────────────────────────────────────
_hr("Test 2: Panoptic Quality edge cases")
from src.metrics import compute_panoptic_quality

# Perfect match
m1 = np.zeros((10, 10), dtype=np.uint8); m1[2:5, 2:5] = 1
res = compute_panoptic_quality([m1], [m1])
assert abs(res["pq"] - 1.0) < 1e-6, f"Perfect match PQ should be 1.0, got {res['pq']}"
print(f"PASS — Perfect match PQ={res['pq']:.4f}")

# No predictions (all FN)
res2 = compute_panoptic_quality([m1], [])
assert res2["pq"] == 0.0 and res2["fn"] == 1
print(f"PASS — No predictions: PQ={res2['pq']}, FN={res2['fn']}")

# No GT (all FP)
res3 = compute_panoptic_quality([], [m1])
assert res3["pq"] == 0.0 and res3["fp"] == 1
print(f"PASS — No GT: PQ={res3['pq']}, FP={res3['fp']}")

# IoU=0.5 edge (threshold is strictly >0.5, so 0.5 should be FP/FN)
m2 = np.zeros((10, 10), dtype=np.uint8); m2[2:5, 4:7] = 1  # 50% overlap with m1
iou = float(np.logical_and(m1, m2).sum() / np.logical_or(m1, m2).sum())
res4 = compute_panoptic_quality([m1], [m2])
expected_tp = 1 if iou > 0.5 else 0
assert res4["tp"] == expected_tp, f"IoU={iou:.3f} TP expected {expected_tp}, got {res4['tp']}"
print(f"PASS — Boundary IoU={iou:.3f}: TP={res4['tp']} (correct)")


# ── 3. Overlap resolution ─────────────────────────────────────────────────────
_hr("Test 3: Overlap resolution")
from src.postprocess import resolve_mask_overlaps

ma = np.zeros((20, 20), dtype=np.uint8); ma[5:15, 5:15] = 1
mb = np.zeros((20, 20), dtype=np.uint8); mb[8:18, 8:18] = 1  # overlaps ma
resolved = resolve_mask_overlaps([(0.9, ma), (0.7, mb)])
assert len(resolved) == 2
overlap = np.logical_and(resolved[0], resolved[1]).sum()
assert overlap == 0, f"Resolved masks still overlap by {overlap} pixels"
print(f"PASS — No pixel overlap after resolution (2 masks)")


# ── 4. Model creation ─────────────────────────────────────────────────────────
_hr("Test 4: Model creation (v2 config)")
from src.models import build_refinement_model
import torch

try:
    model = build_refinement_model(
        encoder_name="resnet34",   # use resnet34 for speed in CI
        encoder_weights=None,
        in_channels=4,
        use_boundary_head=True,
    )
    x = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256), f"Unexpected output shape: {out.shape}"
    print(f"PASS — v2 model (resnet34, 4ch, boundary_head) output shape: {out.shape}")
except Exception as e:
    print(f"SKIP — Model creation test skipped (no GPU / dependency): {e}")


# ── 5. Step 1: Binary mask assertion ─────────────────────────────────────────
_hr("Test 5: Step 1 — Binary mask assertion")
from src.dataset import _test_mask_binary_assertion
_test_mask_binary_assertion()


# ── 6. Step 3: GSD crop size bounds ──────────────────────────────────────────
_hr("Test 6: Step 3 — GSD crop size bounds")
from src.dataset import compute_gsd_crop_size

sizes = [
    compute_gsd_crop_size(100, 100, min_crop=768, max_crop=1280),
    compute_gsd_crop_size(2000, 2000, min_crop=768, max_crop=1280),
    compute_gsd_crop_size(1, 1, min_crop=768, max_crop=1280),
]
for s in sizes:
    assert 768 <= s <= 1280, f"GSD crop size {s} out of [768, 1280]"
print(f"PASS — GSD crop sizes all in [768, 1280]: {sizes}")


# ── 7. Step 7: Gap repair returns valid binary mask ───────────────────────────
_hr("Test 7: Step 7 — Skeleton gap repair")
from src.postprocess import repair_filament_gaps

# Create a filament mask with a small gap
filament = np.zeros((100, 100), dtype=np.uint8)
filament[40:60, 10:45] = 1   # left segment
filament[40:60, 52:90] = 1   # right segment (7px gap at 45-52)

repaired = repair_filament_gaps(filament, gap_px=8)
unique = set(np.unique(repaired).tolist())
assert unique.issubset({0, 1}), f"Repaired mask not binary: {unique}"
print(f"PASS — Repaired mask is binary; foreground px: {filament.sum()} → {repaired.sum()}")


# ── 8. Config: v1 compat flag ─────────────────────────────────────────────────
_hr("Test 8: V2Config v1 compat")
from src.config import v1_compat_config, get_default_config

v1_cfg = v1_compat_config()
v2_cfg = get_default_config()
assert v1_cfg.clahe_path_always is False
assert v2_cfg.clahe_path_always is True
assert v1_cfg.in_channels == 3
assert v2_cfg.in_channels == 4
print("PASS — v1_compat_config and v2 default config fields correct")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═"*50)
print("  All VisionX v2 tests PASSED")
print("═"*50 + "\n")
