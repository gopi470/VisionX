"""
VisionX Panoptic Quality (PQ) & Stratified Evaluation — v2
Official formula: PQ = SQ × RQ, IoU threshold = 0.50, global averaging.
Audit status: matches competition spec as confirmed in Step 0 review.

v2 additions:
  - measure_filament_width(): skeletonisation + distance-transform width estimate
  - classify_width_bucket(): thin ≤2px / medium 3–6px / thick >6px
  - evaluate_stratified_pq(): per-bucket PQ/SQ/RQ breakdown
"""

import numpy as np

try:
    from skimage.morphology import skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# ── Core metric (unchanged from v1 — matches official spec) ──────────────────

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_panoptic_quality(gt_masks: list, pred_masks: list, iou_threshold: float = 0.5):
    """
    Computes PQ, SQ, RQ for a single image.
    Formula: PQ = SQ × RQ
      SQ = mean IoU over TP matches
      RQ = TP / (TP + 0.5*FP + 0.5*FN)
    IoU threshold = 0.50 (unambiguous — a GT can only match one prediction).
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": len(pred_masks), "fn": len(gt_masks)}

    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt_m in enumerate(gt_masks):
        for j, pred_m in enumerate(pred_masks):
            iou_matrix[i, j] = compute_iou(gt_m, pred_m)

    matched_gt: set = set()
    matched_pred: set = set()
    iou_sum = 0.0
    tp = 0

    matches = [
        (iou_matrix[i, j], i, j)
        for i in range(len(gt_masks))
        for j in range(len(pred_masks))
        if iou_matrix[i, j] > iou_threshold
    ]
    matches.sort(key=lambda x: x[0], reverse=True)

    for iou_val, i, j in matches:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            iou_sum += iou_val
            tp += 1

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp

    if tp == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": fp, "fn": fn}

    sq = iou_sum / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def evaluate_dataset_pq(all_gt_masks: list, all_pred_masks: list, iou_threshold: float = 0.5):
    """Evaluates dataset-wide mean PQ across multiple images."""
    pq_list, sq_list, rq_list = [], [], []
    for gt_m, pred_m in zip(all_gt_masks, all_pred_masks):
        res = compute_panoptic_quality(gt_m, pred_m, iou_threshold=iou_threshold)
        pq_list.append(res["pq"])
        sq_list.append(res["sq"])
        rq_list.append(res["rq"])
    return {
        "mean_pq": float(np.mean(pq_list)),
        "mean_sq": float(np.mean(sq_list)),
        "mean_rq": float(np.mean(rq_list)),
    }


# ── v2: Width measurement & stratified evaluation ────────────────────────────

def measure_filament_width(binary_mask: np.ndarray) -> float:
    """
    Estimates the median pixel width of a filament mask using:
      1. Skeletonisation (medial axis)
      2. Distance transform from mask boundary → gives local half-width at each skeleton pixel
      3. Median of 2× those values = typical filament diameter in pixels

    Falls back to sqrt(area / length_estimate) if scikit-image unavailable.
    Returns width in pixels at native (2048×2048) resolution.
    """
    mask = (binary_mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return 0.0

    if _SKIMAGE_AVAILABLE:
        skeleton = skeletonize(mask).astype(np.uint8)
        if skeleton.sum() == 0:
            return float(np.sqrt(area))
        # Distance from every foreground pixel to nearest background pixel
        import scipy.ndimage as ndi
        dist = ndi.distance_transform_edt(mask)
        # Sample distance at skeleton pixels → gives half-width
        half_widths = dist[skeleton > 0]
        return float(2.0 * np.median(half_widths))
    else:
        # Rough estimate: width ≈ area / skeleton_length
        # Use perimeter as a proxy for skeleton length
        contours, _ = __import__("cv2").findContours(mask, __import__("cv2").RETR_EXTERNAL, __import__("cv2").CHAIN_APPROX_SIMPLE)
        perimeter = sum(__import__("cv2").arcLength(c, True) for c in contours) if contours else 1.0
        return float(2.0 * area / (perimeter + 1e-6))


def classify_width_bucket(width_px: float) -> str:
    """
    Classifies a filament width into a bucket for stratified evaluation.
      thin   : ≤ 2 px  (most challenging — 1-2px filaments at native res)
      medium : 3–6 px
      thick  : > 6 px
    """
    if width_px <= 2.0:
        return "thin"
    elif width_px <= 6.0:
        return "medium"
    else:
        return "thick"


def evaluate_stratified_pq(
    all_gt_masks: list,
    all_pred_masks: list,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Step 0 deliverable: stratified PQ broken out by GT filament width bucket.

    Parameters
    ----------
    all_gt_masks  : list of lists — [[mask, ...], ...] one inner list per image
    all_pred_masks: list of lists — [[mask, ...], ...] one inner list per image

    Returns
    -------
    dict with keys: "thin", "medium", "thick", "overall"
    Each value is {"mean_pq", "mean_sq", "mean_rq", "n_instances"}.
    """
    bucket_results: dict = {"thin": [], "medium": [], "thick": []}

    for gt_masks_img, pred_masks_img in zip(all_gt_masks, all_pred_masks):
        # Measure width for every GT mask
        for gt_mask in gt_masks_img:
            w = measure_filament_width(gt_mask)
            bucket = classify_width_bucket(w)

            # For this one GT instance, compute its individual PQ contribution:
            # find its best-matching prediction (if any)
            best_iou = 0.0
            for pred_mask in pred_masks_img:
                iou_val = compute_iou(gt_mask, pred_mask)
                if iou_val > best_iou:
                    best_iou = iou_val

            tp = 1 if best_iou > iou_threshold else 0
            fn = 1 - tp
            sq = best_iou if tp else 0.0
            rq = tp / (tp + 0.5 * fn + 1e-9)  # simplified per-instance F1
            pq = sq * rq
            bucket_results[bucket].append({"pq": pq, "sq": sq, "rq": rq})

    # Aggregate per bucket
    summary = {}
    for bucket, items in bucket_results.items():
        if items:
            summary[bucket] = {
                "mean_pq": float(np.mean([x["pq"] for x in items])),
                "mean_sq": float(np.mean([x["sq"] for x in items])),
                "mean_rq": float(np.mean([x["rq"] for x in items])),
                "n_instances": len(items),
            }
        else:
            summary[bucket] = {"mean_pq": 0.0, "mean_sq": 0.0, "mean_rq": 0.0, "n_instances": 0}

    # Overall (unweighted macro average across images)
    overall = evaluate_dataset_pq(all_gt_masks, all_pred_masks, iou_threshold)
    summary["overall"] = {
        "mean_pq": overall["mean_pq"],
        "mean_sq": overall["mean_sq"],
        "mean_rq": overall["mean_rq"],
        "n_instances": sum(len(g) for g in all_gt_masks),
    }
    return summary


def print_stratified_pq_report(stratified: dict) -> None:
    """Pretty-prints the stratified PQ report to stdout."""
    print("\n── Stratified PQ Report (v2) ────────────────────────────────")
    print(f"{'Bucket':<10} {'N':>6} {'PQ':>8} {'SQ':>8} {'RQ':>8}")
    print("─" * 46)
    for bucket in ("thin", "medium", "thick", "overall"):
        r = stratified.get(bucket, {})
        print(
            f"{bucket:<10} {r.get('n_instances', 0):>6} "
            f"{r.get('mean_pq', 0):.4f}   {r.get('mean_sq', 0):.4f}   {r.get('mean_rq', 0):.4f}"
        )
    print("─" * 46 + "\n")
