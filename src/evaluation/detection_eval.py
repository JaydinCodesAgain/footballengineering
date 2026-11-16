"""Bounding-box IoU matching, AP/mAP calculation, and detection evaluation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.annotation_workflow import FrameAnnotation
from src.schemas import DetectionRecord


def compute_iou(box_a: Sequence[float | int], box_b: Sequence[float | int]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two boxes `[x1, y1, x2, y2]`.
    Returns 0.0 for invalid or non-overlapping boxes.
    """
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = area_a + area_b - inter_area
    return float(inter_area / union_area) if union_area > 0.0 else 0.0


def compute_iou_matrix(
    pred_boxes: Sequence[Sequence[float | int]],
    gt_boxes: Sequence[Sequence[float | int]],
) -> np.ndarray:
    """Compute `(len(pred_boxes), len(gt_boxes))` pairwise IoU matrix."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    mat = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            mat[i, j] = compute_iou(p, g)
    return mat


def match_detections_one_to_one(
    pred_boxes: Sequence[Sequence[float | int]],
    pred_confidences: Sequence[float],
    gt_boxes: Sequence[Sequence[float | int]],
    iou_threshold: float = 0.50,
    method: str = "greedy_confidence",
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Perform strict one-to-one matching between predicted and ground-truth boxes.

    Args:
        pred_boxes: Predicted `[x1, y1, x2, y2]` boxes.
        pred_confidences: Confidence score for each predicted box.
        gt_boxes: Ground-truth `[x1, y1, x2, y2]` boxes.
        iou_threshold: Minimum IoU required for a true positive match.
        method: `'greedy_confidence'` (COCO/VOC standard) or `'hungarian'`.

    Returns:
        `(matches, fp_pred_indices, fn_gt_indices)` where each match is `(pred_idx, gt_idx, iou)`.
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0 and n_gt == 0:
        return [], [], []
    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    matches: List[Tuple[int, int, float]] = []
    matched_preds = set()
    matched_gts = set()

    if method == "hungarian":
        cost = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        # Sort accepted pairs by confidence descending for consistent ordering
        pair_list = []
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_threshold:
                pair_list.append((int(r), int(c), float(iou_mat[r, c])))
        pair_list.sort(key=lambda item: pred_confidences[item[0]], reverse=True)
        for r, c, iou_val in pair_list:
            matches.append((r, c, iou_val))
            matched_preds.add(r)
            matched_gts.add(c)
    else:
        # Standard PASCAL VOC / COCO confidence-sorted greedy matching
        order = sorted(range(n_pred), key=lambda i: float(pred_confidences[i]), reverse=True)
        for p_idx in order:
            best_gt = -1
            best_iou = -1.0
            for g_idx in range(n_gt):
                if g_idx in matched_gts:
                    continue
                val = float(iou_mat[p_idx, g_idx])
                if val >= iou_threshold and val > best_iou:
                    best_iou = val
                    best_gt = g_idx
            if best_gt >= 0:
                matches.append((p_idx, best_gt, best_iou))
                matched_preds.add(p_idx)
                matched_gts.add(best_gt)

    fp_indices = [i for i in range(n_pred) if i not in matched_preds]
    fn_indices = [j for j in range(n_gt) if j not in matched_gts]
    return matches, fp_indices, fn_indices


def compute_average_precision(
    confidences: Sequence[float],
    is_tp: Sequence[bool],
    total_gt: int,
) -> float:
    """
    Compute all-point interpolated Average Precision (AP) from confidence-ranked detections.
    """
    if total_gt <= 0 or len(confidences) == 0:
        return 0.0

    order = np.argsort(-np.asarray(confidences, dtype=np.float64))
    tp_sorted = np.asarray(is_tp, dtype=np.float64)[order]
    fp_sorted = 1.0 - tp_sorted

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # All-point interpolation envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate_class_detections(
    frame_predictions: Dict[int, List[DetectionRecord]],
    verified_annotations: Sequence[FrameAnnotation],
    class_name: str,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate predictions for a single object class (`'player'` or `'ball'`) across all
    manually verified ground-truth frames.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    all_confs: List[float] = []
    all_is_tp: List[bool] = []
    frame_diagnostics: List[Dict[str, Any]] = []

    for ann in verified_annotations:
        if not ann.verified or ann.source != "manual":
            raise ValueError(
                f"Frame {ann.frame_number} does not have verified manual ground-truth provenance "
                f"(verified={ann.verified}, source='{ann.source}')."
            )

        fno = ann.frame_number
        gt_boxes = [b.bbox for b in ann.boxes if b.class_name == class_name]
        preds = [p for p in frame_predictions.get(fno, []) if p.class_name == class_name]
        pred_boxes = [list(p.bbox) for p in preds]
        pred_confs = [float(p.confidence) for p in preds]

        matches, fp_idxs, fn_idxs = match_detections_one_to_one(
            pred_boxes=pred_boxes,
            pred_confidences=pred_confs,
            gt_boxes=gt_boxes,
            iou_threshold=iou_threshold,
            method="greedy_confidence",
        )

        matched_pred_set = {m[0] for m in matches}
        for p_idx, conf in enumerate(pred_confs):
            all_confs.append(conf)
            all_is_tp.append(p_idx in matched_pred_set)

        tp = len(matches)
        fp = len(fp_idxs)
        fn = len(fn_idxs)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_boxes)

        frame_diagnostics.append(
            {
                "frame_number": fno,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matched_pairs": [
                    {
                        "pred_bbox": pred_boxes[p_i],
                        "gt_bbox": gt_boxes[g_i],
                        "iou": round(iou_v, 4),
                        "confidence": round(pred_confs[p_i], 4),
                    }
                    for p_i, g_i, iou_v in matches
                ],
                "false_positive_boxes": [
                    {"bbox": pred_boxes[i], "confidence": round(pred_confs[i], 4)} for i in fp_idxs
                ],
                "missed_ground_truth_boxes": [gt_boxes[j] for j in fn_idxs],
            }
        )

    precision = float(total_tp) / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp) / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    ap50 = compute_average_precision(all_confs, all_is_tp, total_gt)

    # Compute mAP@0.50:0.95
    ap_thresholds = np.linspace(0.50, 0.95, 10)
    aps: List[float] = []
    for t_iou in ap_thresholds:
        t_confs: List[float] = []
        t_tp: List[bool] = []
        for ann in verified_annotations:
            fno = ann.frame_number
            gt_b = [b.bbox for b in ann.boxes if b.class_name == class_name]
            preds = [p for p in frame_predictions.get(fno, []) if p.class_name == class_name]
            p_b = [list(p.bbox) for p in preds]
            p_c = [float(p.confidence) for p in preds]
            m_t, _, _ = match_detections_one_to_one(p_b, p_c, gt_b, iou_threshold=float(t_iou))
            m_set = {m[0] for m in m_t}
            for idx_p, c_val in enumerate(p_c):
                t_confs.append(c_val)
                t_tp.append(idx_p in m_set)
        aps.append(compute_average_precision(t_confs, t_tp, total_gt))

    map_50_95 = float(np.mean(aps)) if aps else 0.0

    return {
        "class_name": class_name,
        "evaluated_frames": len(verified_annotations),
        "total_ground_truth": int(total_gt),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ap50": round(ap50, 4),
        "map50_95": round(map_50_95, 4),
        "frame_diagnostics": frame_diagnostics,
    }


def evaluate_detections_against_ground_truth(
    frame_predictions: Dict[int, List[DetectionRecord]],
    verified_annotations: Sequence[FrameAnnotation],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Run full bounding-box IoU evaluation for both players and footballs.
    If no verified manual bounding-box annotations exist, clearly marks metrics as unavailable.
    """
    verified = [a for a in verified_annotations if a.verified and a.source == "manual"]
    if not verified:
        return {
            "metrics_available": False,
            "status": "unavailable_missing_verified_ground_truth_boxes",
            "message": (
                "No manually verified ground-truth bounding boxes are present yet. "
                "Use DetectionAnnotationStore / `detection_boxes.csv` to provide manual bounding boxes. "
                "Model predictions are never used as synthetic ground truth."
            ),
            "player": None,
            "ball": None,
            "mAP50": None,
        }

    player_res = evaluate_class_detections(frame_predictions, verified, "player", iou_threshold)
    ball_res = evaluate_class_detections(frame_predictions, verified, "ball", iou_threshold)

    active_aps = [
        r["ap50"] for r in (player_res, ball_res) if r["total_ground_truth"] > 0
    ]
    mean_ap50 = round(float(np.mean(active_aps)), 4) if active_aps else 0.0

    return {
        "metrics_available": True,
        "status": "verified_iou_evaluation",
        "iou_threshold": iou_threshold,
        "evaluated_frames": len(verified),
        "player": player_res,
        "ball": ball_res,
        "mAP50": mean_ap50,
    }


def evaluate_legacy_manual_counts(
    predicted_counts_by_frame: Dict[int, Dict[str, int]],
    legacy_ground_truth: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Reproduce the legacy Cell 13 manual count evaluation (`person_tp`, `person_fn`, `ball_tp`, `ball_fn`)
    while explicitly flagging that count-based inspection does not verify spatial IoU localization.
    """
    results: Dict[str, Any] = {
        "evaluation_mode": "legacy_manual_visual_count_review",
        "iou_verified": False,
        "disclaimer": (
            "Metrics derived from manual per-frame TP/FN visual counts (Cell 13) on an 11-frame sample. "
            "Does not perform bounding-box IoU matching and must not be treated as full-match IoU performance."
        ),
        "frames_evaluated": 0,
        "classes": {},
    }

    common_frames = sorted(set(predicted_counts_by_frame.keys()) & set(legacy_ground_truth.keys()))
    results["frames_evaluated"] = len(common_frames)

    for obj_name, pred_key in (("player", "people"), ("ball", "balls")):
        prefix = "person" if obj_name == "player" else "ball"
        tp = 0
        fp = 0
        fn = 0
        for fno in common_frames:
            truth = legacy_ground_truth[fno]
            correct = int(truth[f"{prefix}_tp"])
            missed = int(truth[f"{prefix}_fn"])
            predicted = int(predicted_counts_by_frame[fno][pred_key])
            # If detector settings changed slightly from when visual counts were logged, clamp fp >= 0
            tp_eff = min(correct, predicted)
            fp_eff = max(0, predicted - tp_eff)
            fn_eff = missed + max(0, correct - predicted)
            tp += tp_eff
            fp += fp_eff
            fn += fn_eff

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results["classes"][obj_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return results


def run_confidence_threshold_sweep(
    raw_frame_predictions: Dict[int, List[DetectionRecord]],
    thresholds: Sequence[float] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60),
    verified_annotations: Optional[Sequence[FrameAnnotation]] = None,
    iou_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Sweep confidence thresholds for player and ball detections, reporting detection rates,
    frames with ball detected, and (when verified ground-truth boxes exist) IoU precision/recall/F1.
    """
    total_frames = max(1, len(raw_frame_predictions))
    verified = [a for a in (verified_annotations or []) if a.verified and a.source == "manual"]

    rows: List[Dict[str, Any]] = []
    for thresh in thresholds:
        t_val = float(thresh)
        filtered: Dict[int, List[DetectionRecord]] = {}
        player_count = 0
        ball_count = 0
        frames_with_ball = 0

        for fno, dets in raw_frame_predictions.items():
            f_dets = [d for d in dets if d.confidence >= t_val]
            filtered[fno] = f_dets
            p_c = sum(1 for d in f_dets if d.class_name == "player")
            b_c = sum(1 for d in f_dets if d.class_name == "ball")
            player_count += p_c
            ball_count += b_c
            if b_c > 0:
                frames_with_ball += 1

        row: Dict[str, Any] = {
            "confidence_threshold": round(t_val, 2),
            "total_frames": len(raw_frame_predictions),
            "total_players_detected": player_count,
            "avg_players_per_frame": round(player_count / float(total_frames), 2),
            "total_balls_detected": ball_count,
            "frames_with_ball": frames_with_ball,
            "ball_frame_coverage": round(frames_with_ball / float(total_frames), 4),
        }

        if verified:
            eval_res = evaluate_detections_against_ground_truth(filtered, verified, iou_threshold)
            row["player_precision"] = eval_res["player"]["precision"]
            row["player_recall"] = eval_res["player"]["recall"]
            row["player_f1"] = eval_res["player"]["f1"]
            row["ball_precision"] = eval_res["ball"]["precision"]
            row["ball_recall"] = eval_res["ball"]["recall"]
            row["ball_f1"] = eval_res["ball"]["f1"]
        else:
            row["player_precision"] = None
            row["player_recall"] = None
            row["player_f1"] = None
            row["ball_precision"] = None
            row["ball_recall"] = None
            row["ball_f1"] = None

        rows.append(row)

    return rows

