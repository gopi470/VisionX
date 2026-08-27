"""
VisionX Post-Processing & Overlap Resolution Module
Resolves pixel overlaps between instance candidates and thresholds output masks.
"""

import numpy as np


def resolve_mask_overlaps(scored_masks: list[tuple[float, np.ndarray]], min_area: int = 150) -> list[np.ndarray]:
    """
    Greedily resolves spatial collisions across predicted candidate masks.
    Input: list of (confidence_score, binary_mask) sorted by confidence descending.
    Output: list of non-overlapping binary masks.
    """
    if not scored_masks:
        return []

    # Sort descending by confidence score
    scored_masks = sorted(scored_masks, key=lambda x: x[0], reverse=True)
    
    occupied = np.zeros_like(scored_masks[0][1], dtype=bool)
    final_masks = []

    for score, mask in scored_masks:
        # Subtract already occupied regions
        cleaned_mask = mask.copy()
        cleaned_mask[occupied] = 0
        
        # Filter out small disconnected remnants
        if int(cleaned_mask.sum()) >= min_area:
            occupied |= cleaned_mask.astype(bool)
            final_masks.append(cleaned_mask.astype(np.uint8))

    return final_masks


def apply_morphological_cleaning(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies optional morphological closing/opening to remove tiny speckles and fill small holes.
    """
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned
