"""
VisionX v2 Post-Processing & Overlap Resolution — Step 7

v2 changes:
  - Step 7: Added repair_filament_gaps() using skimage skeletonisation + morphological
            dilation to close small gaps in predicted filament masks before RLE encoding.
  - Confirmed correct execution order:
      TTA/ensemble prob-map avg → Otsu threshold → resolve_mask_overlaps()
                                → repair_filament_gaps() → RLE encode
  - apply_morphological_cleaning() unchanged (now called in pipeline.py post-threshold).
"""

import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def resolve_mask_overlaps(scored_masks: list, min_area: int = 150) -> list:
    """
    Greedily resolves spatial collisions across predicted candidate masks.
    Input : list of (confidence_score, binary_mask) — any order.
    Output: list of non-overlapping binary masks, sorted by confidence desc.

    Execution order note (Step 7):
      This function runs AFTER TTA/ensemble probability averaging and AFTER
      per-crop thresholding — never on per-pass masks before averaging.
    """
    if not scored_masks:
        return []

    scored_masks = sorted(scored_masks, key=lambda x: x[0], reverse=True)
    occupied = np.zeros_like(scored_masks[0][1], dtype=bool)
    final_masks = []

    for score, mask in scored_masks:
        cleaned = mask.copy()
        cleaned[occupied] = 0
        if int(cleaned.sum()) >= min_area:
            occupied |= cleaned.astype(bool)
            final_masks.append(cleaned.astype(np.uint8))

    return final_masks


def apply_morphological_cleaning(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies morphological open→close to remove isolated speckles and fill
    small interior holes in a binary mask.
    Called in pipeline._refine_crop() after Otsu thresholding.
    kernel_size: 5 gives ~5px radius closing — appropriate for 1024px crops.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def repair_filament_gaps(mask: np.ndarray, gap_px: int = 4) -> np.ndarray:
    """
    Step 7: Skeleton-based connectivity repair.

    Problem: collision resolution can leave small gaps in elongated filament
    masks where neighbouring instances "ate" disputed pixels. This function:
      1. Skeletonises the binary mask to find the medial axis.
      2. Dilates the skeleton by gap_px pixels to bridge nearby disconnected ends.
      3. Unions the dilated skeleton with the original mask to fill gaps.
      4. Re-applies morphological closing to smooth the result.

    If scikit-image is not available, falls back to simple dilation+erosion
    gap-closing (less precise but functional).

    Parameters
    ----------
    mask   : binary np.ndarray (uint8, values 0/1)
    gap_px : maximum gap width in pixels to attempt to close

    Returns
    -------
    Repaired binary mask (uint8, values 0/1), same shape as input.
    """
    if mask.sum() == 0:
        return mask

    mask_u8 = (mask > 0).astype(np.uint8)

    if _SKIMAGE_AVAILABLE:
        # Skeletonise → dilate → union with original
        skeleton = _skimage_skeletonize(mask_u8).astype(np.uint8)
        if skeleton.sum() == 0:
            return mask_u8
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px * 2 + 1, gap_px * 2 + 1))
        dilated_skeleton = cv2.dilate(skeleton, kernel, iterations=1)
        repaired = np.clip(mask_u8.astype(np.int32) + dilated_skeleton.astype(np.int32), 0, 1).astype(np.uint8)
    else:
        # Fallback: dilate + erode (morphological closing) to bridge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px * 2 + 1, gap_px * 2 + 1))
        repaired = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Final smooth pass
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, smooth_kernel)

    return repaired.astype(np.uint8)
