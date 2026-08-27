"""
VisionX v2 Unit Test Suite
Runs without requiring torch or segmentation_models_pytorch — mocks missing ML dependencies for fast local execution.
"""

import sys
import os
import types
import numpy as np

# Direct module imports (bypass src/__init__.py which needs ML packages)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock torch & segmentation_models_pytorch if not available locally
if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "no_grad"):
    mock_torch = types.ModuleType("torch")
    mock_torch.Tensor = object
    mock_torch.device = object

    class MockNoGrad:
        def __call__(self, fn):
            return fn
        def __enter__(self): pass
        def __exit__(self, *a): pass

    mock_torch.no_grad = MockNoGrad
    mock_torch.nn = types.ModuleType("torch.nn")
    class MockModule:
        def __init__(self, *args, **kwargs): pass
    mock_torch.nn.Module = MockModule
    mock_torch.nn.functional = types.ModuleType("torch.nn.functional")
    class MockDataset: pass
    mock_torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=MockDataset))
    sys.modules["torch"] = mock_torch
    sys.modules["torch.nn"] = mock_torch.nn
    sys.modules["torch.nn.functional"] = mock_torch.nn.functional
    sys.modules["torch.utils"] = mock_torch.utils
    sys.modules["torch.utils.data"] = mock_torch.utils.data

if "segmentation_models_pytorch" not in sys.modules:
    sys.modules["segmentation_models_pytorch"] = types.ModuleType("segmentation_models_pytorch")


def _hr(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ── 1. RLE round-trip ─────────────────────────────────────────────────────────
_hr("Test 1: RLE encode/decode round-trip")
from src.utils import encode_rle, decode_rle

mask = np.zeros((2048, 2048), dtype=np.uint8)
mask[100:200, 300:500] = 1
rle = encode_rle(mask)
decoded = decode_rle(rle, 2048, 2048)
assert np.array_equal(mask, decoded), "RLE round-trip FAILED"
print("PASS -- RLE round-trip correct")


# ── 2. Panoptic Quality math ──────────────────────────────────────────────────
_hr("Test 2: Panoptic Quality edge cases")
from src.metrics import compute_panoptic_quality

m1 = np.zeros((10, 10), dtype=np.uint8); m1[2:5, 2:5] = 1
res = compute_panoptic_quality([m1], [m1])
assert abs(res["pq"] - 1.0) < 1e-6, f"Perfect match PQ should be 1.0, got {res['pq']}"
print(f"PASS -- Perfect match PQ={res['pq']:.4f}")

res2 = compute_panoptic_quality([m1], [])
assert res2["pq"] == 0.0 and res2["fn"] == 1
print(f"PASS -- No predictions: PQ={res2['pq']}, FN={res2['fn']}")

res3 = compute_panoptic_quality([], [m1])
assert res3["pq"] == 0.0 and res3["fp"] == 1
print(f"PASS -- No GT: PQ={res3['pq']}, FP={res3['fp']}")

m2 = np.zeros((10, 10), dtype=np.uint8); m2[2:5, 4:7] = 1
iou_val = float(np.logical_and(m1, m2).sum() / np.logical_or(m1, m2).sum())
res4 = compute_panoptic_quality([m1], [m2])
expected_tp = 1 if iou_val > 0.5 else 0
assert res4["tp"] == expected_tp
print(f"PASS -- Boundary IoU={iou_val:.3f}: TP={res4['tp']} (correct)")


# ── 3. Overlap resolution ─────────────────────────────────────────────────────
_hr("Test 3: Overlap resolution")
from src.postprocess import resolve_mask_overlaps

ma = np.zeros((20, 20), dtype=np.uint8); ma[5:15, 5:15] = 1
mb = np.zeros((20, 20), dtype=np.uint8); mb[8:18, 8:18] = 1
resolved = resolve_mask_overlaps([(0.9, ma), (0.7, mb)], min_area=1)
assert len(resolved) == 2
overlap = np.logical_and(resolved[0], resolved[1]).sum()
assert overlap == 0, f"Resolved masks still overlap by {overlap} pixels"
print(f"PASS -- No pixel overlap after resolution")


# ── 4. Step 1: Binary mask assertion ─────────────────────────────────────────
_hr("Test 4: Step 1 -- Binary mask assertion")
from src.dataset import _assert_binary, _test_mask_binary_assertion
_test_mask_binary_assertion()


# ── 5. Step 3: GSD crop size bounds ──────────────────────────────────────────
_hr("Test 5: Step 3 -- GSD crop size bounds")
from src.dataset import compute_gsd_crop_size

sizes = [
    compute_gsd_crop_size(100, 100, min_crop=768, max_crop=1280),
    compute_gsd_crop_size(2000, 2000, min_crop=768, max_crop=1280),
    compute_gsd_crop_size(1, 1, min_crop=768, max_crop=1280),
]
for s in sizes:
    assert 768 <= s <= 1280, f"GSD crop size {s} out of [768, 1280]"
print(f"PASS -- GSD crop sizes all in [768, 1280]: {sizes}")


# ── 6. Step 7: Gap repair returns valid binary mask ───────────────────────────
_hr("Test 6: Step 7 -- Skeleton gap repair")
from src.postprocess import repair_filament_gaps

filament = np.zeros((100, 100), dtype=np.uint8)
filament[40:60, 10:45] = 1
filament[40:60, 52:90] = 1  # 7px gap

repaired = repair_filament_gaps(filament, gap_px=8)
unique = set(np.unique(repaired).tolist())
assert unique.issubset({0, 1}), f"Repaired mask not binary: {unique}"
print(f"PASS -- Repaired mask binary; px: {filament.sum()} -> {repaired.sum()}")


# ── 7. Config flags ───────────────────────────────────────────────────────────
_hr("Test 7: V2Config flags")
from src.config import v1_compat_config, get_default_config

v1 = v1_compat_config()
v2 = get_default_config()
assert v1.clahe_path_always is False
assert v2.clahe_path_always is True
assert v1.in_channels == 3
assert v2.in_channels == 4
assert v1.use_boundary_head is False
assert v2.use_boundary_head is True
print("PASS -- V2Config and v1_compat_config fields correct")


# ── 8. Stratified PQ (numpy-only) ────────────────────────────────────────────
_hr("Test 8: Stratified PQ utility")
from src.metrics import evaluate_stratified_pq, measure_filament_width, classify_width_bucket

thin = np.zeros((100, 100), dtype=np.uint8)
thin[49:51, 10:60] = 1

thick = np.zeros((100, 100), dtype=np.uint8)
thick[40:50, 20:70] = 1

w_thin = measure_filament_width(thin)
w_thick = measure_filament_width(thick)
print(f"  Thin filament measured width: {w_thin:.2f}px")
print(f"  Thick filament measured width: {w_thick:.2f}px")
assert classify_width_bucket(w_thin) == "thin", f"Expected thin, got {classify_width_bucket(w_thin)}"
assert classify_width_bucket(w_thick) in ("medium", "thick")

strat = evaluate_stratified_pq([[thin, thick]], [[thin, thick]])
assert "thin" in strat and "overall" in strat
print(f"PASS -- Stratified PQ: overall PQ={strat['overall']['mean_pq']:.4f}")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  All VisionX v2 tests PASSED")
print("="*50 + "\n")
