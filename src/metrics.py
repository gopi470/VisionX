"""
VisionX Panoptic Quality (PQ) & Dice Metric Implementation
Follows CVPR panoptic quality formula: PQ = SQ * RQ
"""

import numpy as np


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_panoptic_quality(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_threshold: float = 0.5):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
    between a set of ground truth instance masks and predicted instance masks for a single image.
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": len(pred_masks), "fn": len(gt_masks)}

    # Compute pairwise IoU matrix
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt_m in enumerate(gt_masks):
        for j, pred_m in enumerate(pred_masks):
            iou_matrix[i, j] = compute_iou(gt_m, pred_m)

    # Match true positives where IoU > threshold (unambiguous matching for iou > 0.5)
    matched_gt = set()
    matched_pred = set()
    iou_sum = 0.0
    tp = 0

    # Sort matches by IoU descending
    matches = []
    for i in range(len(gt_masks)):
        for j in range(len(pred_masks)):
            if iou_matrix[i, j] > iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
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
    pq = sq * rq

    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def evaluate_dataset_pq(all_gt_masks: list[list[np.ndarray]], all_pred_masks: list[list[np.ndarray]], iou_threshold: float = 0.5):
    """
    Evaluates dataset-wide mean Panoptic Quality across multiple images.
    """
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
