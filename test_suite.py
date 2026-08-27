"""
VisionX Unit Test Suite
Verifies Panoptic Quality math, RLE encode/decode round-trip, and model architecture initialization.
"""

import numpy as np
import torch
from src.utils import encode_rle, decode_rle
from src.metrics import compute_panoptic_quality, compute_iou
from src.models import build_refinement_model, CombinedBCEDiceLoss
from src.postprocess import resolve_mask_overlaps


def test_rle_roundtrip():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[100:300, 200:400] = 1

    rle_str = encode_rle(mask)
    assert isinstance(rle_str, str), "RLE output must be string"

    decoded = decode_rle(rle_str, 2048, 2048)
    assert np.array_equal(mask, decoded), "Decoded mask does not match original binary mask"
    print("✓ test_rle_roundtrip passed")


def test_iou_and_pq():
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_a[10:50, 10:50] = 1

    mask_b = np.zeros((100, 100), dtype=np.uint8)
    mask_b[10:50, 10:50] = 1

    iou = compute_iou(mask_a, mask_b)
    assert iou == 1.0, f"Expected IoU 1.0, got {iou}"

    pq_res = compute_panoptic_quality([mask_a], [mask_b], iou_threshold=0.5)
    assert pq_res["pq"] == 1.0, f"Expected PQ 1.0, got {pq_res['pq']}"
    print("✓ test_iou_and_pq passed")


def test_overlap_resolution():
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:60, 10:60] = 1

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[40:80, 40:80] = 1

    scored = [(0.90, mask1), (0.80, mask2)]
    resolved = resolve_mask_overlaps(scored, min_area=50)

    assert len(resolved) == 2, f"Expected 2 resolved masks, got {len(resolved)}"
    # Check no pixel overlap remains
    overlap = np.logical_and(resolved[0], resolved[1]).sum()
    assert overlap == 0, f"Expected 0 pixel overlap, got {overlap}"
    print("✓ test_overlap_resolution passed")


def test_model_build():
    model = build_refinement_model(encoder_name="resnet34", encoder_weights=None, in_channels=3)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    assert output.shape == (2, 1, 512, 512), f"Expected output shape (2, 1, 512, 512), got {output.shape}"
    print("✓ test_model_build passed")


if __name__ == "__main__":
    print("Running VisionX test suite...")
    test_rle_roundtrip()
    test_iou_and_pq()
    test_overlap_resolution()
    test_model_build()
    print("All tests passed successfully!")
