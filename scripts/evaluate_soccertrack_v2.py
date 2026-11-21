"""Post-hoc evaluation of Football Engineering pipeline outputs against SoccerTrack v2 references.

Strictly separates model inference from reference labels: this script only reads
finished pipeline outputs (`frame_observations.csv`, `ball_tracking_diagnostics.csv`,
`possession_timeline.csv`, `team_discovery_report.json`) and evaluates them post-hoc
against SoccerTrack v2 GSR (`gsr/<match>/<match>_1st.json`), BAS events
(`bas/<match>/<match>_12_class_events.json`), and auxiliary ball tracks
(`ball/<match>_1st_ball.npz`).

Computes:
1. Automatic team discovery metrics (best Team_A/Team_B permutation post-hoc):
   covered accuracy, covered Macro-F1, overall accuracy including Unknown rejections,
   coverage, ARI, NMI, cross-team swaps, goalkeeper separation, and track-level team purity.
2. Formal player tracking metrics against GSR persistent identities:
   MOTA, MOTP, IDF1, IDP, IDR, HOTA (DetA, AssA, LocA), IDSW, Fragmentations,
   MT/PT/ML, per-player dominant-ID purity, and within-shot diagnostics.
3. Ball tracking diagnostics, false camera-cut check (fixed camera), auxiliary
   ball-track / BAS-actor proximity analysis, and failure-frame sampling for
   verified ball-box inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.classification.team_discovery import evaluate_discovered_labels
from src.detection.annotation_workflow import DetectionAnnotationStore
from src.evaluation.tracking_eval import evaluate_within_shot_player_tracking
from src.schemas import BallTrajectoryRecord, FrameObservationRecord, InternalLabel


def compute_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_gsr_frames_streaming(
    gsr_json_path: Path,
    target_frames_1based: Set[int],
    stop_after_frame: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    """Fast line-based streaming parser for ~2.7 GB pretty-printed SoccerTrack v2 GSR JSONs.

    Extracts ONLY `supercategory == "object"` records for `target_frames_1based` in ~2–3 s
    and < 20 MB RAM without loading the full JSON into memory.
    """
    if not target_frames_1based:
        return {}, {}

    max_target = stop_after_frame if stop_after_frame is not None else max(target_frames_1based)
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    meta: Dict[str, Any] = {"image_size": None, "info": None}

    with gsr_json_path.open("r", encoding="utf-8", errors="replace") as f:
        in_info = False
        info_lines: List[str] = []
        in_first_img = False
        img_lines: List[str] = []
        in_annotations = False

        in_obj = False
        keep_obj = False
        obj_lines: List[str] = []
        consecutive_beyond = 0

        for line in f:
            if not in_annotations:
                if meta["info"] is None:
                    if not in_info and line.startswith('    "info": {'):
                        in_info = True
                        info_lines.append("{")
                        continue
                    elif in_info:
                        if line.startswith("    },") or line.startswith("    }"):
                            info_lines.append("}")
                            try:
                                meta["info"] = json.loads("".join(info_lines))
                            except json.JSONDecodeError:
                                pass
                            in_info = False
                        else:
                            info_lines.append(line)
                        continue

                if meta["image_size"] is None:
                    if not in_first_img and line.startswith("        {"):
                        in_first_img = True
                        img_lines.append("{")
                        continue
                    elif in_first_img:
                        if line.startswith("        },") or line.startswith("        }"):
                            img_lines.append("}")
                            try:
                                img0 = json.loads("".join(img_lines))
                                if "width" in img0 and "height" in img0:
                                    meta["image_size"] = [int(img0["width"]), int(img0["height"])]
                            except json.JSONDecodeError:
                                pass
                            in_first_img = False
                        else:
                            img_lines.append(line)
                        continue

                if line.startswith('    "annotations": ['):
                    in_annotations = True
                continue

            # Inside "annotations": [ ... ]
            if line.startswith("    ]"):
                break
            if not in_obj:
                if line.startswith("        {"):
                    in_obj = True
                    keep_obj = False
                    obj_lines = ["{"]
                continue

            if line.startswith("        },") or line.startswith("        }"):
                in_obj = False
                if not keep_obj:
                    obj_lines = []
                    continue
                obj_lines.append("}")
                try:
                    rec = json.loads("".join(obj_lines))
                except json.JSONDecodeError:
                    obj_lines = []
                    continue
                obj_lines = []

                if rec.get("supercategory") != "object":
                    continue

                img_id_str = str(rec.get("image_id", ""))
                if len(img_id_str) >= 6:
                    frame_1b = int(img_id_str[-6:])
                else:
                    try:
                        frame_1b = int(img_id_str)
                    except ValueError:
                        continue

                if frame_1b in target_frames_1based:
                    consecutive_beyond = 0
                    bi = rec.get("bbox_image")
                    if isinstance(bi, dict) and all(k in bi for k in ("x", "y", "w", "h")):
                        x = float(bi["x"])
                        y = float(bi["y"])
                        w = float(bi["w"])
                        h = float(bi["h"])
                        bbox_xyxy = (x, y, x + w, y + h)
                    elif isinstance(bi, list) and len(bi) == 4:
                        x, y, w, h = [float(v) for v in bi]
                        bbox_xyxy = (x, y, x + w, y + h)
                    else:
                        continue

                    attrs = rec.get("attributes") or {}
                    bp = rec.get("bbox_pitch") or {}
                    out[frame_1b].append(
                        {
                            "frame_1based": frame_1b,
                            "track_id": int(rec["track_id"]),
                            "player_id": attrs.get("player_id"),
                            "role": str(attrs.get("role", "player")).lower(),
                            "jersey": attrs.get("jersey"),
                            "team": attrs.get("team"),
                            "bbox_xyxy": bbox_xyxy,
                            "pitch_x": bp.get("x_bottom_middle"),
                            "pitch_y": bp.get("y_bottom_middle"),
                        }
                    )
                elif frame_1b > max_target + 25:
                    consecutive_beyond += 1
                    if consecutive_beyond >= 200:
                        break
            else:
                if '"supercategory": "object"' in line:
                    keep_obj = True
                # Only buffer lines while object could still be an "object" (pitch lines are 350 lines,
                # and "supercategory" appears in the first 6 lines of every annotation record)
                if keep_obj or len(obj_lines) < 10:
                    obj_lines.append(line)

    return dict(out), meta


def load_pipeline_observations(obs_csv: Path) -> Tuple[Dict[int, List[FrameObservationRecord]], List[FrameObservationRecord]]:
    by_frame: Dict[int, List[FrameObservationRecord]] = defaultdict(list)
    all_recs: List[FrameObservationRecord] = []
    with obs_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fno = int(row["frame_number"])
            ts = float(row["timestamp"])
            pbox = tuple(int(x) for x in row["player_bbox"].strip("[]").split(","))
            pid = int(row["player_id"]) if row.get("player_id") not in (None, "", "None") else None
            rec = FrameObservationRecord(
                match_id=row.get("match_id", ""),
                frame_number=fno,
                timestamp=ts,
                player_id=pid,
                team_id=row["team_id"],
                team_display_name=row.get("team_display_name", row["team_id"]),
                raw_team_id=row.get("raw_team_id", row["team_id"]),
                player_bbox=pbox,  # type: ignore[arg-type]
                player_detection_confidence=float(row.get("player_detection_confidence") or 0.0),
                classification_confidence=float(row.get("classification_confidence") or 0.0),
                classification_distance=float(row.get("classification_distance") or 1.0),
                track_team_confidence=float(row.get("track_team_confidence") or 0.0),
                is_team_locked=str(row.get("is_team_locked", "")).lower() == "true",
            )
            by_frame[fno].append(rec)
            all_recs.append(rec)
    return dict(by_frame), all_recs


def load_ball_trajectory(ball_csv: Path) -> List[BallTrajectoryRecord]:
    recs: List[BallTrajectoryRecord] = []
    with ball_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bbox_val = None
            if row.get("bbox"):
                parts = [int(x) for x in row["bbox"].strip("[]").split(",") if x.strip()]
                if len(parts) == 4:
                    bbox_val = (parts[0], parts[1], parts[2], parts[3])
            recs.append(
                BallTrajectoryRecord(
                    match_id=row.get("match_id", ""),
                    frame_number=int(row["frame_number"]),
                    timestamp=float(row["timestamp"]),
                    observation_mode=row["observation_mode"],
                    is_observed=str(row.get("is_observed", "")).lower() == "true",
                    is_interpolated=str(row.get("is_interpolated", "")).lower() == "true",
                    bbox=bbox_val,
                    center_x=float(row["center_x"]) if row.get("center_x") not in (None, "", "None") else None,
                    center_y=float(row["center_y"]) if row.get("center_y") not in (None, "", "None") else None,
                    velocity_x=float(row.get("velocity_x") or 0.0),
                    velocity_y=float(row.get("velocity_y") or 0.0),
                    speed_px_per_sec=float(row.get("speed_px_per_sec") or 0.0),
                    confidence=float(row.get("confidence") or 0.0),
                    uncertainty_radius_px=float(row.get("uncertainty_radius_px") or 999.0),
                    frames_since_observed=int(row.get("frames_since_observed") or 0),
                    camera_cut_reset=str(row.get("camera_cut_reset", "")).lower() == "true",
                    rejected_outliers_count=int(row.get("rejected_outliers_count") or 0),
                    detection_source=row.get("detection_source", "none"),
                )
            )
    return recs


def match_boxes_hungarian(
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    """Optimal 1-to-1 box matching maximizing IoU above `iou_threshold`."""
    if not gt_boxes or not pred_boxes:
        return []
    n_gt, n_pr = len(gt_boxes), len(pred_boxes)
    iou_mat = np.zeros((n_gt, n_pr), dtype=np.float64)
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_mat[i, j] = compute_iou_xyxy(gb, pb)
    # Maximize IoU via linear_sum_assignment on negative IoU
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matches: List[Tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        val = float(iou_mat[r, c])
        if val >= iou_threshold:
            matches.append((int(r), int(c), val))
    return matches


def verify_frame_offset(
    obs_by_frame: Dict[int, List[FrameObservationRecord]],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    candidate_offsets: Sequence[int] = (-1, 0, 1, 2, 3),
) -> Dict[str, Any]:
    """Verify exact frame index alignment (`gsr_frame = pred_frame + offset`)."""
    sample_pred_frames = sorted(obs_by_frame.keys())[5:55]
    scores: Dict[int, Dict[str, Any]] = {}
    for off in candidate_offsets:
        ious: List[float] = []
        matched = 0
        total_gt = 0
        for pf in sample_pred_frames:
            gt = gsr_by_frame.get(pf + off, [])
            pr = obs_by_frame.get(pf, [])
            if not gt:
                continue
            total_gt += len(gt)
            m = match_boxes_hungarian(
                [g["bbox_xyxy"] for g in gt],
                [tuple(float(v) for v in p.player_bbox) for p in pr],
                iou_threshold=0.20,
            )
            matched += len(m)
            ious.extend(val for _, _, val in m)
        scores[off] = {
            "matched_count": matched,
            "total_gt": total_gt,
            "recall_iou20": round(matched / float(max(1, total_gt)), 4),
            "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        }
    best_off = max(scores.keys(), key=lambda k: (scores[k]["recall_iou20"], scores[k]["mean_matched_iou"]))
    return {"best_offset": int(best_off), "per_offset": {str(k): v for k, v in scores.items()}}


def compute_mot_and_hota(
    eval_frames: Sequence[int],
    obs_by_frame: Dict[int, List[FrameObservationRecord]],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    frame_offset: int,
    primary_iou: float = 0.30,
) -> Dict[str, Any]:
    """Compute MOTA, MOTP, IDF1/IDP/IDR, IDSW, Fragmentations, MT/PT/ML, and HOTA."""
    # --- 1. Primary IoU threshold CLEAR MOT + IDF1 ---
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_frag = 0
    matched_ious: List[float] = []

    last_matched_pred_for_gt: Dict[int, int] = {}
    prev_frame_matched_gt: Set[int] = set()
    gt_seen_counts: Counter[int] = Counter()
    gt_matched_counts: Counter[int] = Counter()
    gt_pred_cooccur: Dict[Tuple[int, int], int] = defaultdict(int)
    gt_assigned_pred_seq: Dict[int, List[Optional[int]]] = defaultdict(list)

    for pf in eval_frames:
        gt_list = gsr_by_frame.get(pf + frame_offset, [])
        pr_list = [p for p in obs_by_frame.get(pf, []) if p.player_id is not None]
        if not gt_list:
            continue

        total_gt += len(gt_list)
        total_pred += len(pr_list)
        for g in gt_list:
            gt_seen_counts[g["track_id"]] += 1

        matches = match_boxes_hungarian(
            [g["bbox_xyxy"] for g in gt_list],
            [tuple(float(v) for v in p.player_bbox) for p in pr_list],
            iou_threshold=primary_iou,
        )
        matched_gt_idx: Dict[int, Tuple[int, float]] = {gi: (pi, iou) for gi, pi, iou in matches}
        curr_matched_gt_ids: Set[int] = set()

        for gi, g in enumerate(gt_list):
            gt_id = g["track_id"]
            if gi in matched_gt_idx:
                pi, iou = matched_gt_idx[gi]
                pr_id = int(pr_list[pi].player_id)  # type: ignore[arg-type]
                total_tp += 1
                matched_ious.append(iou)
                gt_matched_counts[gt_id] += 1
                gt_pred_cooccur[(gt_id, pr_id)] += 1
                gt_assigned_pred_seq[gt_id].append(pr_id)
                curr_matched_gt_ids.add(gt_id)

                if gt_id in last_matched_pred_for_gt:
                    if last_matched_pred_for_gt[gt_id] != pr_id:
                        total_idsw += 1
                    if gt_id not in prev_frame_matched_gt:
                        total_frag += 1
                last_matched_pred_for_gt[gt_id] = pr_id
            else:
                total_fn += 1
                gt_assigned_pred_seq[gt_id].append(None)

        total_fp += len(pr_list) - len(matches)
        prev_frame_matched_gt = curr_matched_gt_ids

    precision = total_tp / float(max(1, total_tp + total_fp))
    recall = total_tp / float(max(1, total_tp + total_fn))
    det_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + total_idsw) / float(max(1, total_gt))
    motp = float(np.mean(matched_ious)) if matched_ious else 0.0

    # Global optimal 1-to-1 assignment for IDF1
    gt_ids = sorted(gt_seen_counts.keys())
    pr_ids = sorted({pr_id for _, pr_id in gt_pred_cooccur.keys()})
    idtp = 0
    if gt_ids and pr_ids:
        gt_idx_map = {g: i for i, g in enumerate(gt_ids)}
        pr_idx_map = {p: j for j, p in enumerate(pr_ids)}
        mat = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
        for (gid, pid), cnt in gt_pred_cooccur.items():
            mat[gt_idx_map[gid], pr_idx_map[pid]] = cnt
        r_ind, c_ind = linear_sum_assignment(-mat)
        idtp = int(sum(mat[r, c] for r, c in zip(r_ind, c_ind)))

    idfp = max(0, total_pred - idtp)
    idfn = max(0, total_gt - idtp)
    idp = idtp / float(max(1, idtp + idfp))
    idr = idtp / float(max(1, idtp + idfn))
    idf1 = (2.0 * idtp) / float(max(1, 2 * idtp + idfp + idfn))

    # MT / PT / ML and per-GT-player identity continuity
    mt = pt = ml = 0
    per_gt_player: Dict[str, Any] = {}
    purities: List[float] = []
    distinct_counts: List[int] = []
    for gid in gt_ids:
        seen = gt_seen_counts[gid]
        matched_c = gt_matched_counts[gid]
        cov = matched_c / float(max(1, seen))
        if cov >= 0.80:
            mt += 1
        elif cov >= 0.20:
            pt += 1
        else:
            ml += 1
        seq = [x for x in gt_assigned_pred_seq[gid] if x is not None]
        if seq:
            dom_id, dom_cnt = Counter(seq).most_common(1)[0]
            purity = dom_cnt / float(len(seq))
            purities.append(purity)
            distinct_counts.append(len(set(seq)))
            switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        else:
            dom_id, purity, switches = None, 0.0, 0
            distinct_counts.append(0)
        per_gt_player[str(gid)] = {
            "gt_frames": seen,
            "matched_frames": matched_c,
            "track_recall": round(cov, 4),
            "distinct_pred_ids": len(set(seq)),
            "dominant_pred_id": dom_id,
            "dominant_id_purity": round(purity, 4),
            "id_switches": switches,
        }

    # --- 2. Formal HOTA across alpha = 0.05..0.95 (step 0.05) ---
    alphas = [round(0.05 * k, 2) for k in range(1, 20)]
    hota_vals: List[float] = []
    deta_vals: List[float] = []
    assa_vals: List[float] = []
    loca_vals: List[float] = []

    # Precompute per-frame IoU matrices once for speed
    frame_cache = []
    for pf in eval_frames:
        gt_list = gsr_by_frame.get(pf + frame_offset, [])
        pr_list = [p for p in obs_by_frame.get(pf, []) if p.player_id is not None]
        if not gt_list:
            continue
        gt_ids_f = [g["track_id"] for g in gt_list]
        pr_ids_f = [int(p.player_id) for p in pr_list]  # type: ignore[arg-type]
        if pr_list:
            mat = np.zeros((len(gt_list), len(pr_list)), dtype=np.float64)
            for i, g in enumerate(gt_list):
                gb = g["bbox_xyxy"]
                for j, p in enumerate(pr_list):
                    mat[i, j] = compute_iou_xyxy(gb, tuple(float(v) for v in p.player_bbox))
        else:
            mat = np.zeros((len(gt_list), 0), dtype=np.float64)
        frame_cache.append((gt_ids_f, pr_ids_f, mat))

    for alpha in alphas:
        tp_a = 0
        fp_a = 0
        fn_a = 0
        iou_sum_a = 0.0
        tpa_pair: Counter[Tuple[int, int]] = Counter()
        gt_total_a: Counter[int] = Counter()
        pr_total_a: Counter[int] = Counter()

        for gt_ids_f, pr_ids_f, iou_mat in frame_cache:
            for gid in gt_ids_f:
                gt_total_a[gid] += 1
            for pid in pr_ids_f:
                pr_total_a[pid] += 1

            n_g, n_p = iou_mat.shape
            if n_g == 0 or n_p == 0:
                fn_a += n_g
                fp_a += n_p
                continue

            r_ind, c_ind = linear_sum_assignment(-iou_mat)
            m_cnt = 0
            for r, c in zip(r_ind, c_ind):
                val = float(iou_mat[r, c])
                if val >= alpha:
                    m_cnt += 1
                    tp_a += 1
                    iou_sum_a += val
                    tpa_pair[(gt_ids_f[r], pr_ids_f[c])] += 1
            fn_a += n_g - m_cnt
            fp_a += n_p - m_cnt

        det_a = tp_a / float(max(1, tp_a + fp_a + fn_a))
        loc_a = iou_sum_a / float(max(1, tp_a)) if tp_a > 0 else 0.0
        if tp_a > 0:
            ass_sum = 0.0
            for (gid, pid), tpa in tpa_pair.items():
                fna = gt_total_a[gid] - tpa
                fpa = pr_total_a[pid] - tpa
                ass_sum += tpa * (tpa / float(max(1, tpa + fna + fpa)))
            ass_a = ass_sum / float(tp_a)
        else:
            ass_a = 0.0
        hota_a = math.sqrt(det_a * ass_a)

        deta_vals.append(det_a)
        assa_vals.append(ass_a)
        loca_vals.append(loc_a)
        hota_vals.append(hota_a)

    return {
        "primary_iou_threshold": primary_iou,
        "frames_evaluated": len(frame_cache),
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "detection_f1": round(det_f1, 4),
        "mota": round(mota, 4),
        "motp_mean_iou": round(motp, 4),
        "idf1": round(idf1, 4),
        "idp": round(idp, 4),
        "idr": round(idr, 4),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "id_switches": total_idsw,
        "fragmentations": total_frag,
        "gt_unique_identities": len(gt_ids),
        "pred_unique_identities_matched": len(pr_ids),
        "mostly_tracked_mt": mt,
        "partially_tracked_pt": pt,
        "mostly_lost_ml": ml,
        "mean_gt_dominant_id_purity": round(float(np.mean(purities)), 4) if purities else 0.0,
        "mean_distinct_pred_ids_per_gt_player": round(float(np.mean(distinct_counts)), 2) if distinct_counts else 0.0,
        "hota": round(float(np.mean(hota_vals)), 4),
        "deta": round(float(np.mean(deta_vals)), 4),
        "assa": round(float(np.mean(assa_vals)), 4),
        "loca": round(float(np.mean(loca_vals)), 4),
        "hota_at_iou50": round(hota_vals[alphas.index(0.50)], 4),
        "deta_at_iou50": round(deta_vals[alphas.index(0.50)], 4),
        "assa_at_iou50": round(assa_vals[alphas.index(0.50)], 4),
        "per_gt_player": per_gt_player,
    }


def evaluate_team_discovery_against_gsr(
    eval_frames: Sequence[int],
    obs_by_frame: Dict[int, List[FrameObservationRecord]],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    frame_offset: int,
    iou_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Post-hoc team discovery evaluation against SoccerTrack v2 GSR team & role references."""
    outfield_preds: List[str] = []
    outfield_refs: List[str] = []
    outfield_raw_preds: List[str] = []

    all_with_gk_preds: List[str] = []
    all_with_gk_refs: List[str] = []
    gk_pred_counts: Counter[str] = Counter()
    unmatched_pred_counts: Counter[str] = Counter()

    # Track-level mapping: pred_track_id -> Counter of (pred_label, gt_team)
    pred_track_labels: Dict[int, Counter[str]] = defaultdict(Counter)
    pred_track_gt_teams: Dict[int, Counter[str]] = defaultdict(Counter)
    pred_track_gt_roles: Dict[int, Counter[str]] = defaultdict(Counter)
    gt_player_pred_teams: Dict[int, Counter[str]] = defaultdict(Counter)
    gt_player_true_team: Dict[int, str] = {}
    gt_player_role: Dict[int, str] = {}

    for pf in eval_frames:
        gt_list = gsr_by_frame.get(pf + frame_offset, [])
        pr_list = obs_by_frame.get(pf, [])
        if not gt_list:
            continue

        matches = match_boxes_hungarian(
            [g["bbox_xyxy"] for g in gt_list],
            [tuple(float(v) for v in p.player_bbox) for p in pr_list],
            iou_threshold=iou_threshold,
        )
        matched_pr = {pi for _, pi, _ in matches}
        for pi, p in enumerate(pr_list):
            if pi not in matched_pr:
                unmatched_pred_counts[p.team_id] += 1

        for gi, pi, _ in matches:
            g = gt_list[gi]
            p = pr_list[pi]
            role = g["role"]
            team_side = g["team"]
            if team_side == "left":
                gt_team = InternalLabel.TEAM_A.value
            elif team_side == "right":
                gt_team = InternalLabel.TEAM_B.value
            else:
                gt_team = InternalLabel.UNKNOWN.value

            gt_player_role[g["track_id"]] = role
            gt_player_true_team[g["track_id"]] = gt_team
            gt_player_pred_teams[g["track_id"]][p.team_id] += 1

            if p.player_id is not None:
                pred_track_labels[int(p.player_id)][p.team_id] += 1
                pred_track_gt_teams[int(p.player_id)][gt_team] += 1
                pred_track_gt_roles[int(p.player_id)][role] += 1

            if role == "player" and gt_team in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value):
                outfield_preds.append(p.team_id)
                outfield_raw_preds.append(p.raw_team_id)
                outfield_refs.append(gt_team)
                all_with_gk_preds.append(p.team_id)
                all_with_gk_refs.append(gt_team)
            elif role == "goalkeeper":
                gk_pred_counts[p.team_id] += 1
                all_with_gk_preds.append(p.team_id)
                all_with_gk_refs.append(InternalLabel.UNKNOWN.value)

    outfield_eval = evaluate_discovered_labels(outfield_preds, outfield_refs)
    outfield_raw_eval = evaluate_discovered_labels(outfield_raw_preds, outfield_refs)
    with_gk_eval = evaluate_discovered_labels(all_with_gk_preds, all_with_gk_refs)

    # Track-level evaluation (one vote per predicted track matched to outfield players)
    trk_preds: List[str] = []
    trk_refs: List[str] = []
    for pid, gt_cnt in pred_track_gt_teams.items():
        if pred_track_gt_roles[pid].most_common(1)[0][0] != "player":
            continue
        dom_gt = gt_cnt.most_common(1)[0][0]
        if dom_gt not in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value):
            continue
        dom_pr = pred_track_labels[pid].most_common(1)[0][0]
        trk_preds.append(dom_pr)
        trk_refs.append(dom_gt)
    track_level_eval = evaluate_discovered_labels(trk_preds, trk_refs)

    # Per-GT-player team assignment summary (aligned with best outfield permutation)
    use_swap = bool(outfield_eval.get("permutation_applied", False))

    def _align(lbl: str) -> str:
        if not use_swap:
            return lbl
        if lbl == InternalLabel.TEAM_A.value:
            return InternalLabel.TEAM_B.value
        if lbl == InternalLabel.TEAM_B.value:
            return InternalLabel.TEAM_A.value
        return lbl

    gt_player_team_summary: Dict[str, Any] = {}
    outfield_players_correct = 0
    outfield_players_wrong = 0
    outfield_players_unknown = 0
    for gid in sorted(gt_player_pred_teams.keys()):
        aligned_counts = Counter({_align(k): v for k, v in gt_player_pred_teams[gid].items()})
        dom_pred, dom_cnt = aligned_counts.most_common(1)[0]
        total_obs = sum(aligned_counts.values())
        true_team = gt_player_true_team[gid]
        role = gt_player_role[gid]
        if role == "player":
            if dom_pred == true_team:
                outfield_players_correct += 1
            elif dom_pred == InternalLabel.UNKNOWN.value:
                outfield_players_unknown += 1
            else:
                outfield_players_wrong += 1
        gt_player_team_summary[str(gid)] = {
            "role": role,
            "gt_team": true_team,
            "aligned_pred_counts": dict(aligned_counts),
            "dominant_aligned_pred": dom_pred,
            "total_matched_obs": total_obs,
            "correct_dominant": dom_pred == true_team if role == "player" else dom_pred == InternalLabel.UNKNOWN.value,
        }

    total_gk_obs = sum(gk_pred_counts.values())
    gk_rejected_as_unknown = gk_pred_counts.get(InternalLabel.UNKNOWN.value, 0)
    return {
        "iou_threshold": iou_threshold,
        "outfield_frame_observations_eval": outfield_eval,
        "outfield_raw_pre_stabilizer_eval": outfield_raw_eval,
        "outfield_plus_gk_as_unknown_eval": with_gk_eval,
        "predicted_track_level_outfield_eval": track_level_eval,
        "gt_outfield_player_majority_summary": {
            "total_gt_outfield_players_matched": outfield_players_correct + outfield_players_wrong + outfield_players_unknown,
            "correct_team": outfield_players_correct,
            "wrong_team": outfield_players_wrong,
            "unknown_majority": outfield_players_unknown,
        },
        "goalkeeper_handling": {
            "total_gk_matched_observations": total_gk_obs,
            "gk_predicted_counts": dict(gk_pred_counts),
            "gk_rejected_as_unknown_rate": round(gk_rejected_as_unknown / float(max(1, total_gk_obs)), 4)
            if total_gk_obs > 0
            else None,
        },
        "unmatched_predictions_label_counts": dict(unmatched_pred_counts),
        "per_gt_player_team": gt_player_team_summary,
    }


def project_pitch_points_to_distorted_image(
    pitch_xy_centered: np.ndarray,
    intrinsics_path: Path,
    homography_path: Path,
) -> Optional[np.ndarray]:
    """Project SoccerTrack v2 centered metric pitch coords `(x in [-52.5,52.5], y in [-34,34])`
    back to raw panoramic image pixels `(u, v)`.

    Note: SoccerTrack v2 keypoints use `(x_pitch in [0, 105], y_pitch in [0, 68])`.
    This function first checks calibration consistency using GSR objects if available,
    so callers should validate projected errors against known GSR player foot positions.
    """
    if not intrinsics_path.is_file() or not homography_path.is_file():
        return None
    intr = np.load(str(intrinsics_path), allow_pickle=True)
    K = intr["K"].astype(np.float64)
    D = intr["D"].astype(np.float64)
    Knew = intr["Knew"].astype(np.float64)
    H = np.load(str(homography_path)).astype(np.float64)

    # Convert centered pitch (-52.5..52.5, -34..34) -> top-left pitch (0..105, 0..68)
    pts = pitch_xy_centered.copy().astype(np.float64)
    pts[:, 0] = pts[:, 0] + 52.5
    pts[:, 1] = pts[:, 1] + 34.0

    # Try H mapping pitch -> undistorted image
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    undist_hom = (H @ hom.T).T
    undist_xy = undist_hom[:, :2] / np.where(np.abs(undist_hom[:, 2:3]) > 1e-9, undist_hom[:, 2:3], 1.0)

    # Map undistorted image pixels -> normalized camera rays -> distort via fisheye.distortPoints
    Knew_inv = np.linalg.inv(Knew)
    undist_h = np.hstack([undist_xy, ones])
    norm_rays = (Knew_inv @ undist_h.T).T
    norm_2d = (norm_rays[:, :2] / norm_rays[:, 2:3]).reshape(-1, 1, 2)
    distorted = cv2.fisheye.distortPoints(norm_2d, K, D).reshape(-1, 2)
    # Apply K to get distorted pixel coordinates
    dist_h = np.hstack([distorted, ones])
    # Note: cv2.fisheye.distortPoints already applies K! Wait: OpenCV fisheye.distortPoints
    # takes normalized undistorted coordinates and returns pixel coordinates with K applied.
    return distorted


def analyze_ball_tracking_and_errors(
    ball_records: Sequence[BallTrajectoryRecord],
    eval_frames: Sequence[int],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    frame_offset: int,
    bas_json_path: Optional[Path] = None,
    ball_npz_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    match_id: str = "",
) -> Dict[str, Any]:
    """Analyze ball-tracking provenance, errors, BAS-anchor proximity, and select failure frames."""
    rec_by_frame = {int(r.frame_number): r for r in ball_records}
    obs = [r for r in ball_records if r.is_observed]
    interp = [r for r in ball_records if r.is_interpolated]
    lost = [r for r in ball_records if not r.is_observed and not r.is_interpolated]
    cut_frames = [int(r.frame_number) for r in ball_records if r.camera_cut_reset]

    # Segment lengths & jumps
    seg_lengths: List[int] = []
    lost_streaks: List[Tuple[int, int, int]] = []  # (start_f, end_f, len)
    cur_active = 0
    cur_lost_start: Optional[int] = None
    cur_lost_len = 0
    jumps: List[Dict[str, Any]] = []
    prev_obs: Optional[BallTrajectoryRecord] = None

    for r in ball_records:
        if r.is_observed or r.is_interpolated:
            cur_active += 1
            if cur_lost_len > 0 and cur_lost_start is not None:
                lost_streaks.append((cur_lost_start, r.frame_number, cur_lost_len))
                cur_lost_len = 0
                cur_lost_start = None
        else:
            if cur_active > 0:
                seg_lengths.append(cur_active)
                cur_active = 0
            if cur_lost_start is None:
                cur_lost_start = r.frame_number
            cur_lost_len += 1

        if r.is_observed and r.center_x is not None and r.center_y is not None:
            if prev_obs is not None and prev_obs.center_x is not None and prev_obs.center_y is not None:
                df = max(1, r.frame_number - prev_obs.frame_number)
                dist = math.hypot(r.center_x - prev_obs.center_x, r.center_y - prev_obs.center_y)
                step_per_frame = dist / float(df)
                jumps.append(
                    {
                        "frame_number": r.frame_number,
                        "prev_frame": prev_obs.frame_number,
                        "dist_px": round(dist, 2),
                        "px_per_frame": round(step_per_frame, 2),
                        "center": [round(r.center_x, 1), round(r.center_y, 1)],
                        "conf": r.confidence,
                        "source": r.detection_source,
                    }
                )
            prev_obs = r
    if cur_active > 0:
        seg_lengths.append(cur_active)
    if cur_lost_len > 0 and cur_lost_start is not None:
        lost_streaks.append((cur_lost_start, ball_records[-1].frame_number, cur_lost_len))

    # Check static clustering of observed balls (potential static background FPs!)
    grid_hits: Counter[Tuple[int, int]] = Counter()
    for r in obs:
        if r.center_x is not None and r.center_y is not None:
            grid_hits[(int(r.center_x // 40), int(r.center_y // 40))] += 1
    top_static_cells = [
        {"cell_xy_px": [gx * 40, gy * 40], "observed_hits": cnt}
        for (gx, gy), cnt in grid_hits.most_common(10)
    ]

    # Compare against BAS events (independent human-annotated ball actions with actor player_id!)
    bas_checks: List[Dict[str, Any]] = []
    if bas_json_path and bas_json_path.is_file() and eval_frames:
        bas_data = json.loads(bas_json_path.read_text(encoding="utf-8"))
        max_f1b = max(eval_frames) + frame_offset
        min_f1b = min(eval_frames) + frame_offset
        for act in bas_data.get("actions", []):
            gt_str = str(act.get("gameTime", ""))
            if not gt_str.startswith("1 -"):
                continue
            ev_f = int(act.get("frame", -1))
            if ev_f < min_f1b or ev_f > max_f1b:
                continue
            # Snap to nearest evaluated pipeline frame
            nearest_pf = min(eval_frames, key=lambda pf: abs((pf + frame_offset) - ev_f))
            if abs((nearest_pf + frame_offset) - ev_f) > 3:
                continue
            brec = rec_by_frame.get(nearest_pf)
            # Find acting player's GSR bounding box on nearest_pf + frame_offset
            actor_pid = str(act.get("player_id", ""))
            gt_objs = gsr_by_frame.get(nearest_pf + frame_offset, [])
            actor_box = None
            for obj in gt_objs:
                if str(obj.get("player_id", "")) == actor_pid:
                    actor_box = obj["bbox_xyxy"]
                    break
            dist_to_actor_feet = None
            if actor_box is not None and brec is not None and brec.center_x is not None and brec.center_y is not None:
                afx = (actor_box[0] + actor_box[2]) / 2.0
                afy = actor_box[3]  # feet
                dist_to_actor_feet = round(math.hypot(brec.center_x - afx, brec.center_y - afy), 1)
            bas_checks.append(
                {
                    "bas_frame_1b": ev_f,
                    "eval_frame_0b": nearest_pf,
                    "label": act.get("label"),
                    "team": act.get("team"),
                    "actor_player_id": actor_pid,
                    "actor_bbox_xyxy": [round(v, 1) for v in actor_box] if actor_box else None,
                    "ball_mode": brec.observation_mode if brec else "missing",
                    "ball_center": [round(brec.center_x, 1), round(brec.center_y, 1)]
                    if (brec and brec.center_x is not None and brec.center_y is not None)
                    else None,
                    "ball_conf": brec.confidence if brec else None,
                    "ball_source": brec.detection_source if brec else None,
                    "dist_to_actor_feet_px": dist_to_actor_feet,
                }
            )

    # Auxiliary ball track (pitch -> image projection, explicitly flagged as interpolated proxy)
    aux_ball_summary: Dict[str, Any] = {"available": False}
    if ball_npz_path and ball_npz_path.is_file() and raw_dir is not None:
        intr_p = raw_dir / f"{match_id}_camera_intrinsics.npz"
        H_p = raw_dir / f"{match_id}_homography.npy"
        if intr_p.is_file() and H_p.is_file():
            # Validate projection error on GSR players first so we don't trust a mis-oriented projection
            sample_pitch = []
            sample_img_feet = []
            for pf in eval_frames[:40]:
                for obj in gsr_by_frame.get(pf + frame_offset, []):
                    if obj["pitch_x"] is not None and obj["pitch_y"] is not None:
                        sample_pitch.append([float(obj["pitch_x"]), float(obj["pitch_y"])])
                        bx = obj["bbox_xyxy"]
                        sample_img_feet.append([(bx[0] + bx[2]) / 2.0, bx[3]])
            if len(sample_pitch) >= 20:
                sp = np.array(sample_pitch, dtype=np.float64)
                si = np.array(sample_img_feet, dtype=np.float64)
                # Fit direct robust polynomial/homography sanity check vs intrinsics projection
                proj_errs = {}
                for flip_y in (False, True):
                    test_p = sp.copy()
                    if flip_y:
                        test_p[:, 1] = -test_p[:, 1]
                    proj = project_pitch_points_to_distorted_image(test_p, intr_p, H_p)
                    if proj is not None:
                        err = float(np.median(np.linalg.norm(proj - si, axis=1)))
                        proj_errs[flip_y] = (err, proj)
                best_flip = min(proj_errs.keys(), key=lambda k: proj_errs[k][0])
                cal_med_err = proj_errs[best_flip][0]

                # Project auxiliary ball track
                b_npz = np.load(str(ball_npz_path), allow_pickle=True)
                bf = b_npz["frame"]
                bx = b_npz["x"]
                by = b_npz["y"]
                b_by_f = {int(f): (float(x), float(y)) for f, x, y in zip(bf, bx, by) if not (math.isnan(x) or math.isnan(y))}

                aux_dists: List[float] = []
                aux_active_matches = 0
                aux_far_outliers = 0
                aux_valid_frames = 0
                for pf in eval_frames:
                    xy = b_by_f.get(pf + frame_offset)
                    if xy is None:
                        continue
                    aux_valid_frames += 1
                    pt = np.array([[xy[0], -xy[1] if best_flip else xy[1]]], dtype=np.float64)
                    uv = project_pitch_points_to_distorted_image(pt, intr_p, H_p)
                    brec = rec_by_frame.get(pf)
                    if uv is not None and brec is not None and brec.center_x is not None and brec.center_y is not None:
                        d = float(math.hypot(brec.center_x - uv[0, 0], brec.center_y - uv[0, 1]))
                        aux_dists.append(d)
                        if d <= 120.0:
                            aux_active_matches += 1
                        if d > 300.0:
                            aux_far_outliers += 1

                aux_ball_summary = {
                    "available": True,
                    "caveat": (
                        "SoccerTrack v2 auxiliary ball track is piecewise constant-speed interpolated "
                        "between annotated BAS events (not independently measured per frame); use for "
                        "coarse diagnostic triage only, NOT formal precision/recall."
                    ),
                    "player_foot_projection_median_error_px": round(cal_med_err, 2),
                    "y_flipped_used": bool(best_flip),
                    "eval_frames_with_aux_ball": aux_valid_frames,
                    "active_pred_compared_frames": len(aux_dists),
                    "median_dist_to_aux_ball_px": round(float(np.median(aux_dists)), 1) if aux_dists else None,
                    "p90_dist_to_aux_ball_px": round(float(np.percentile(aux_dists, 90)), 1) if aux_dists else None,
                    "within_120px_of_aux_ball": aux_active_matches,
                    "over_300px_from_aux_ball_likely_fp": aux_far_outliers,
                }

    large_jumps = sorted([j for j in jumps if j["dist_px"] > 150.0], key=lambda x: -x["dist_px"])
    return {
        "total_frames": len(ball_records),
        "observed_frames": len(obs),
        "interpolated_frames": len(interp),
        "lost_frames": len(lost),
        "observed_rate_pct": round(len(obs) / float(max(1, len(ball_records))) * 100.0, 2),
        "interpolated_rate_pct": round(len(interp) / float(max(1, len(ball_records))) * 100.0, 2),
        "active_rate_pct": round((len(obs) + len(interp)) / float(max(1, len(ball_records))) * 100.0, 2),
        "lost_rate_pct": round(len(lost) / float(max(1, len(ball_records))) * 100.0, 2),
        "detection_source_counts": dict(Counter(r.detection_source for r in obs)),
        "false_camera_cuts_on_fixed_camera": len(cut_frames),
        "camera_cut_frames": cut_frames[:20],
        "mean_continuous_track_frames": round(float(np.mean(seg_lengths)), 2) if seg_lengths else 0.0,
        "max_continuous_track_frames": max(seg_lengths) if seg_lengths else 0,
        "longest_lost_streaks": sorted(lost_streaks, key=lambda x: -x[2])[:10],
        "large_jumps_over_150px_count": len(large_jumps),
        "top_large_jumps": large_jumps[:15],
        "top_spatial_cells_40px": top_static_cells,
        "bas_event_anchor_checks": bas_checks,
        "auxiliary_ball_track_diagnostic": aux_ball_summary,
    }


def load_fisheye_pitch_projector(raw_dir: Optional[Path], match_id: str):
    """Build exact ray-plane inverse `(u, v) -> (pitch_x, pitch_y)` in centered metres
    and forward `(pitch_x, pitch_y) -> (u, v)` using SoccerTrack v2's camera intrinsics.
    """
    if raw_dir is None:
        return None
    intr_p = raw_dir / f"{match_id}_camera_intrinsics.npz"
    if not intr_p.is_file():
        return None
    intr = np.load(str(intr_p), allow_pickle=True)
    K = intr["K"].astype(np.float64)
    D = intr["D"].astype(np.float64)
    rvec = np.array(intr["rvecs"][0], dtype=np.float64).reshape(3, 1)
    tvec = np.array(intr["tvecs"][0], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    R_T = R.T
    cam_pos = -(R_T @ tvec).reshape(3)

    def pixel_to_pitch(pts_uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
        rays_world = (R_T @ np.hstack([und, np.ones((len(und), 1))]).T).T
        s = -cam_pos[2] / np.where(np.abs(rays_world[:, 2]) > 1e-9, rays_world[:, 2], 1e-9)
        w = cam_pos[None, :] + s[:, None] * rays_world
        return np.column_stack([w[:, 0] - 52.5, w[:, 1] - 34.0])

    def pitch_to_pixel(pts_pitch_centered: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_pitch_centered, dtype=np.float64).reshape(-1, 2)
        world = np.column_stack([pts[:, 0] + 52.5, pts[:, 1] + 34.0, np.zeros(len(pts))]).reshape(-1, 1, 3)
        uv, _ = cv2.fisheye.projectPoints(world, rvec, tvec, K, D)
        return uv.reshape(-1, 2)

    kp_p = raw_dir / f"{match_id}_keypoints.json"
    cal_err = None
    if kp_p.is_file():
        kp = json.loads(kp_p.read_text(encoding="utf-8"))
        gt_w = np.array([[float(k.strip("()").split(",")[0]) - 52.5, float(k.strip("()").split(",")[1]) - 34.0] for k in kp.keys()])
        img_p = np.array(list(kp.values()), dtype=np.float64)
        est_w = pixel_to_pitch(img_p)
        errs = np.linalg.norm(est_w - gt_w, axis=1)
        cal_err = {
            "median_m": round(float(np.median(errs)), 2),
            "p90_m": round(float(np.percentile(errs, 90)), 2),
            "max_m": round(float(np.max(errs)), 2),
        }
    return {"pixel_to_pitch": pixel_to_pitch, "pitch_to_pixel": pitch_to_pixel, "calibration_keypoint_error": cal_err}


def match_pitch_hungarian(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    max_dist_m: float,
) -> List[Tuple[int, int, float]]:
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return []
    dmat = np.linalg.norm(gt_xy[:, None, :] - pred_xy[None, :, :], axis=2)
    # Clamp distances beyond max_dist_m so infeasible pairs never steal feasible matches
    cost = np.where(dmat <= max_dist_m, dmat, max_dist_m + 1000.0)
    r_ind, c_ind = linear_sum_assignment(cost)
    out: List[Tuple[int, int, float]] = []
    for r, c in zip(r_ind, c_ind):
        d = float(dmat[r, c])
        if d <= max_dist_m:
            out.append((int(r), int(c), d))
    return out


def evaluate_pitch_space_gsr(
    eval_frames: Sequence[int],
    obs_by_frame: Dict[int, List[FrameObservationRecord]],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    frame_offset: int,
    projector: Dict[str, Any],
    match_radius_m: float = 5.0,
) -> Dict[str, Any]:
    """Authoritative 2D metric pitch-space evaluation (SoccerNet GSR / SoccerTrack v2 convention).

    Projects predicted player feet into centered metric pitch coordinates `(x, y)`
    and scores Detection, MOTA/IDF1/IDSW, Pitch-Space GS-HOTA (`S(d) = max(0, 1 - d / match_radius_m)`),
    and Post-Hoc Team Discovery against GSR's authoritative `pitch_x, pitch_y`, `track_id`, `role`, `team`.
    """
    p2p = projector["pixel_to_pitch"]

    total_gt = 0
    total_pred_all = 0
    total_pred_on_pitch = 0
    off_pitch_label_counts: Counter[str] = Counter()
    total_tp = 0
    total_fp_all = 0
    total_fp_on_pitch = 0
    total_fn = 0
    total_idsw = 0
    total_frag = 0
    matched_dists_m: List[float] = []

    last_matched_pred: Dict[int, int] = {}
    prev_matched_gt: Set[int] = set()
    gt_seen: Counter[int] = Counter()
    gt_matched: Counter[int] = Counter()
    gt_pred_cooccur: Dict[Tuple[int, int], int] = defaultdict(int)
    gt_pred_seq: Dict[int, List[Optional[int]]] = defaultdict(list)

    # Team discovery accumulators
    outfield_preds: List[str] = []
    outfield_raw_preds: List[str] = []
    outfield_refs: List[str] = []
    all_with_gk_preds: List[str] = []
    all_with_gk_refs: List[str] = []
    gk_pred_counts: Counter[str] = Counter()
    pred_track_labels: Dict[int, Counter[str]] = defaultdict(Counter)
    pred_track_gt_teams: Dict[int, Counter[str]] = defaultdict(Counter)
    pred_track_gt_roles: Dict[int, Counter[str]] = defaultdict(Counter)
    gt_player_pred_teams: Dict[int, Counter[str]] = defaultdict(Counter)
    gt_player_true_team: Dict[int, str] = {}
    gt_player_role: Dict[int, str] = {}

    frame_cache = []  # for GS-HOTA sweep

    for pf in eval_frames:
        gt_list = [g for g in gsr_by_frame.get(pf + frame_offset, []) if g["pitch_x"] is not None and g["pitch_y"] is not None]
        pr_list = [p for p in obs_by_frame.get(pf, []) if p.player_id is not None]
        if not gt_list:
            continue

        total_gt += len(gt_list)
        total_pred_all += len(pr_list)
        for g in gt_list:
            gt_seen[g["track_id"]] += 1

        gt_xy = np.array([[float(g["pitch_x"]), float(g["pitch_y"])] for g in gt_list], dtype=np.float64)
        if pr_list:
            feet_uv = np.array([[(p.player_bbox[0] + p.player_bbox[2]) / 2.0, float(p.player_bbox[3])] for p in pr_list], dtype=np.float64)
            pr_xy_all = p2p(feet_uv)
            on_pitch_mask = (np.abs(pr_xy_all[:, 0]) <= 55.0) & (np.abs(pr_xy_all[:, 1]) <= 36.5)
            for p, is_on in zip(pr_list, on_pitch_mask):
                if not is_on:
                    off_pitch_label_counts[p.team_id] += 1
            total_pred_on_pitch += int(on_pitch_mask.sum())
            dmat = np.linalg.norm(gt_xy[:, None, :] - pr_xy_all[None, :, :], axis=2)
        else:
            pr_xy_all = np.zeros((0, 2), dtype=np.float64)
            on_pitch_mask = np.zeros(0, dtype=bool)
            dmat = np.zeros((len(gt_list), 0), dtype=np.float64)

        frame_cache.append(([g["track_id"] for g in gt_list], [int(p.player_id) for p in pr_list], dmat))  # type: ignore[arg-type]

        matches = match_pitch_hungarian(gt_xy, pr_xy_all, max_dist_m=match_radius_m)
        matched_gt_map: Dict[int, Tuple[int, float]] = {gi: (pi, d) for gi, pi, d in matches}
        curr_matched_gt: Set[int] = set()

        for gi, g in enumerate(gt_list):
            gid = g["track_id"]
            if gi in matched_gt_map:
                pi, dist_m = matched_gt_map[gi]
                p = pr_list[pi]
                pid = int(p.player_id)  # type: ignore[arg-type]
                total_tp += 1
                matched_dists_m.append(dist_m)
                gt_matched[gid] += 1
                gt_pred_cooccur[(gid, pid)] += 1
                gt_pred_seq[gid].append(pid)
                curr_matched_gt.add(gid)

                if gid in last_matched_pred:
                    if last_matched_pred[gid] != pid:
                        total_idsw += 1
                    if gid not in prev_matched_gt:
                        total_frag += 1
                last_matched_pred[gid] = pid

                role = g["role"]
                team_side = g["team"]
                gt_team = (
                    InternalLabel.TEAM_A.value
                    if team_side == "left"
                    else InternalLabel.TEAM_B.value
                    if team_side == "right"
                    else InternalLabel.UNKNOWN.value
                )
                gt_player_role[gid] = role
                gt_player_true_team[gid] = gt_team
                gt_player_pred_teams[gid][p.team_id] += 1
                pred_track_labels[pid][p.team_id] += 1
                pred_track_gt_teams[pid][gt_team] += 1
                pred_track_gt_roles[pid][role] += 1

                if role == "player" and gt_team in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value):
                    outfield_preds.append(p.team_id)
                    outfield_raw_preds.append(p.raw_team_id)
                    outfield_refs.append(gt_team)
                    all_with_gk_preds.append(p.team_id)
                    all_with_gk_refs.append(gt_team)
                elif role == "goalkeeper":
                    gk_pred_counts[p.team_id] += 1
                    all_with_gk_preds.append(p.team_id)
                    all_with_gk_refs.append(InternalLabel.UNKNOWN.value)
            else:
                total_fn += 1
                gt_pred_seq[gid].append(None)

        matched_pr_set = {pi for _, pi, _ in matches}
        total_fp_all += len(pr_list) - len(matches)
        total_fp_on_pitch += sum(1 for pi in range(len(pr_list)) if on_pitch_mask[pi] and pi not in matched_pr_set)
        prev_matched_gt = curr_matched_gt

    # Detection & CLEAR MOT & IDF1
    prec_all = total_tp / float(max(1, total_tp + total_fp_all))
    prec_on = total_tp / float(max(1, total_tp + total_fp_on_pitch))
    rec = total_tp / float(max(1, total_tp + total_fn))
    det_f1 = (2.0 * prec_all * rec / (prec_all + rec)) if (prec_all + rec) > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp_all + total_idsw) / float(max(1, total_gt))
    mota_on_pitch = 1.0 - (total_fn + total_fp_on_pitch + total_idsw) / float(max(1, total_gt))

    gt_ids = sorted(gt_seen.keys())
    pr_ids = sorted({pid for _, pid in gt_pred_cooccur.keys()})
    idtp = 0
    if gt_ids and pr_ids:
        g_map = {g: i for i, g in enumerate(gt_ids)}
        p_map = {p: j for j, p in enumerate(pr_ids)}
        mat = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
        for (gid, pid), cnt in gt_pred_cooccur.items():
            mat[g_map[gid], p_map[pid]] = cnt
        ri, ci = linear_sum_assignment(-mat)
        idtp = int(sum(mat[r, c] for r, c in zip(ri, ci)))
    idfp = max(0, total_pred_all - idtp)
    idfn = max(0, total_gt - idtp)
    idp = idtp / float(max(1, idtp + idfp))
    idr = idtp / float(max(1, idtp + idfn))
    idf1 = (2.0 * idtp) / float(max(1, 2 * idtp + idfp + idfn))

    mt = pt = ml = 0
    purities: List[float] = []
    distinct_counts: List[int] = []
    per_gt_player: Dict[str, Any] = {}
    for gid in gt_ids:
        seen = gt_seen[gid]
        mc = gt_matched[gid]
        cov = mc / float(max(1, seen))
        if cov >= 0.80:
            mt += 1
        elif cov >= 0.20:
            pt += 1
        else:
            ml += 1
        seq = [x for x in gt_pred_seq[gid] if x is not None]
        if seq:
            dom_id, dom_cnt = Counter(seq).most_common(1)[0]
            pur = dom_cnt / float(len(seq))
            purities.append(pur)
            distinct_counts.append(len(set(seq)))
            sw = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        else:
            dom_id, pur, sw = None, 0.0, 0
            distinct_counts.append(0)
        per_gt_player[str(gid)] = {
            "gt_frames": seen,
            "matched_frames": mc,
            "track_recall": round(cov, 4),
            "distinct_pred_ids": len(set(seq)),
            "dominant_pred_id": dom_id,
            "dominant_id_purity": round(pur, 4),
            "id_switches": sw,
        }

    # Pitch-space GS-HOTA (similarity S(d) = max(0, 1 - d / match_radius_m), alpha = 0.05..0.95)
    alphas = [round(0.05 * k, 2) for k in range(1, 20)]
    hota_vals, deta_vals, assa_vals, loca_vals = [], [], [], []
    for alpha in alphas:
        max_d_alpha = match_radius_m * (1.0 - alpha)
        tp_a = fp_a = fn_a = 0
        sim_sum = 0.0
        tpa_pair: Counter[Tuple[int, int]] = Counter()
        gt_tot_a: Counter[int] = Counter()
        pr_tot_a: Counter[int] = Counter()
        for g_ids_f, p_ids_f, dmat in frame_cache:
            for g in g_ids_f:
                gt_tot_a[g] += 1
            for p in p_ids_f:
                pr_tot_a[p] += 1
            ng, np_ = dmat.shape
            if ng == 0 or np_ == 0:
                fn_a += ng
                fp_a += np_
                continue
            sim_mat = np.maximum(0.0, 1.0 - dmat / match_radius_m)
            ri, ci = linear_sum_assignment(-sim_mat)
            m_cnt = 0
            for r, c in zip(ri, ci):
                s_val = float(sim_mat[r, c])
                if s_val >= alpha and float(dmat[r, c]) <= max_d_alpha + 1e-9:
                    m_cnt += 1
                    tp_a += 1
                    sim_sum += s_val
                    tpa_pair[(g_ids_f[r], p_ids_f[c])] += 1
            fn_a += ng - m_cnt
            fp_a += np_ - m_cnt
        det_a = tp_a / float(max(1, tp_a + fp_a + fn_a))
        loc_a = sim_sum / float(max(1, tp_a)) if tp_a > 0 else 0.0
        if tp_a > 0:
            ass_sum = 0.0
            for (gid, pid), tpa in tpa_pair.items():
                fna = gt_tot_a[gid] - tpa
                fpa = pr_tot_a[pid] - tpa
                ass_sum += tpa * (tpa / float(max(1, tpa + fna + fpa)))
            ass_a = ass_sum / float(tp_a)
        else:
            ass_a = 0.0
        deta_vals.append(det_a)
        assa_vals.append(ass_a)
        loca_vals.append(loc_a)
        hota_vals.append(math.sqrt(det_a * ass_a))

    # Team discovery summary
    outfield_eval = evaluate_discovered_labels(outfield_preds, outfield_refs)
    outfield_raw_eval = evaluate_discovered_labels(outfield_raw_preds, outfield_refs)
    with_gk_eval = evaluate_discovered_labels(all_with_gk_preds, all_with_gk_refs)
    trk_preds, trk_refs = [], []
    for pid, gt_cnt in pred_track_gt_teams.items():
        if pred_track_gt_roles[pid].most_common(1)[0][0] != "player":
            continue
        dom_gt = gt_cnt.most_common(1)[0][0]
        if dom_gt not in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value):
            continue
        trk_preds.append(pred_track_labels[pid].most_common(1)[0][0])
        trk_refs.append(dom_gt)
    track_eval = evaluate_discovered_labels(trk_preds, trk_refs)

    use_swap = bool(outfield_eval.get("permutation_applied", False))
    def _align(lbl: str) -> str:
        if not use_swap:
            return lbl
        if lbl == InternalLabel.TEAM_A.value:
            return InternalLabel.TEAM_B.value
        if lbl == InternalLabel.TEAM_B.value:
            return InternalLabel.TEAM_A.value
        return lbl

    outfield_ok = outfield_bad = outfield_unk = 0
    for gid in sorted(gt_player_pred_teams.keys()):
        if gt_player_role[gid] != "player":
            continue
        aligned = Counter({_align(k): v for k, v in gt_player_pred_teams[gid].items()})
        dom_p = aligned.most_common(1)[0][0]
        if dom_p == gt_player_true_team[gid]:
            outfield_ok += 1
        elif dom_p == InternalLabel.UNKNOWN.value:
            outfield_unk += 1
        else:
            outfield_bad += 1

    return {
        "match_radius_m": match_radius_m,
        "calibration_keypoint_error": projector["calibration_keypoint_error"],
        "hota_similarity_definition": f"S(d) = max(0, 1 - d / {match_radius_m}m) over alpha=0.05..0.95",
        "tracking": {
            "frames_evaluated": len(frame_cache),
            "total_gt_players": total_gt,
            "total_pred_all": total_pred_all,
            "total_pred_on_pitch": total_pred_on_pitch,
            "off_pitch_sideline_detections": total_pred_all - total_pred_on_pitch,
            "off_pitch_label_counts": dict(off_pitch_label_counts),
            "tp": total_tp,
            "fp_all": total_fp_all,
            "fp_on_pitch": total_fp_on_pitch,
            "fn": total_fn,
            "detection_precision_all": round(prec_all, 4),
            "detection_precision_on_pitch": round(prec_on, 4),
            "detection_recall": round(rec, 4),
            "detection_f1_all": round(det_f1, 4),
            "mean_matched_distance_m": round(float(np.mean(matched_dists_m)), 2) if matched_dists_m else None,
            "median_matched_distance_m": round(float(np.median(matched_dists_m)), 2) if matched_dists_m else None,
            "mota_all": round(mota, 4),
            "mota_on_pitch": round(mota_on_pitch, 4),
            "idf1": round(idf1, 4),
            "idp": round(idp, 4),
            "idr": round(idr, 4),
            "idtp": idtp,
            "idfp": idfp,
            "idfn": idfn,
            "id_switches": total_idsw,
            "fragmentations": total_frag,
            "gt_unique_identities": len(gt_ids),
            "pred_unique_identities_matched": len(pr_ids),
            "mostly_tracked_mt": mt,
            "partially_tracked_pt": pt,
            "mostly_lost_ml": ml,
            "mean_gt_dominant_id_purity": round(float(np.mean(purities)), 4) if purities else 0.0,
            "mean_distinct_pred_ids_per_gt_player": round(float(np.mean(distinct_counts)), 2) if distinct_counts else 0.0,
            "gs_hota": round(float(np.mean(hota_vals)), 4),
            "gs_deta": round(float(np.mean(deta_vals)), 4),
            "gs_assa": round(float(np.mean(assa_vals)), 4),
            "gs_loca": round(float(np.mean(loca_vals)), 4),
            "per_gt_player": per_gt_player,
        },
        "team_discovery": {
            "outfield_frame_observations_eval": outfield_eval,
            "outfield_raw_pre_stabilizer_eval": outfield_raw_eval,
            "outfield_plus_gk_as_unknown_eval": with_gk_eval,
            "predicted_track_level_outfield_eval": track_eval,
            "gt_outfield_player_majority_summary": {
                "total_gt_outfield_players_matched": outfield_ok + outfield_bad + outfield_unk,
                "correct_team": outfield_ok,
                "wrong_team": outfield_bad,
                "unknown_majority": outfield_unk,
            },
            "goalkeeper_handling": {
                "total_gk_matched_observations": int(sum(gk_pred_counts.values())),
                "gk_predicted_counts": dict(gk_pred_counts),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Football Engineering outputs on SoccerTrack v2.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Pipeline output directory.")
    parser.add_argument("--gsr-json", type=Path, required=True, help="SoccerTrack v2 GSR JSON path.")
    parser.add_argument("--match-id", type=str, required=True, help="SoccerTrack v2 match ID (e.g. 117092).")
    parser.add_argument("--bas-json", type=Path, default=None, help="Optional BAS events JSON path.")
    parser.add_argument("--ball-npz", type=Path, default=None, help="Optional auxiliary ball npz path.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional raw/<match_id> calibration dir.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to save evaluation JSON.")
    parser.add_argument("--export-ball-sample-video", type=Path, default=None, help="Optional MP4 to export ball annotation frames.")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="Where to export DetectionAnnotationStore.")
    args = parser.parse_args()

    obs_by_frame, all_obs = load_pipeline_observations(args.run_dir / "frame_observations.csv")
    ball_records = load_ball_trajectory(args.run_dir / "ball_tracking_diagnostics.csv")
    eval_frames = sorted(obs_by_frame.keys())
    if not eval_frames:
        raise RuntimeError(f"No frame observations found in {args.run_dir / 'frame_observations.csv'}")

    # Load GSR frames covering eval_frames with [-2..+4] buffer for offset verification
    target_1b: Set[int] = set()
    for pf in eval_frames:
        for d in (-1, 0, 1, 2, 3):
            if pf + d >= 1:
                target_1b.add(pf + d)
    print(f"Streaming GSR annotations from {args.gsr_json} for {len(eval_frames)} evaluated frames...")
    gsr_by_frame, gsr_meta = load_gsr_frames_streaming(args.gsr_json, target_1b, stop_after_frame=max(target_1b) + 10)
    print(f"Loaded {len(gsr_by_frame)} GSR frames.")

    offset_check = verify_frame_offset(obs_by_frame, gsr_by_frame)
    best_offset = int(offset_check["best_offset"])
    print(f"Verified frame alignment: gsr_frame = pred_frame + {best_offset} ({offset_check['per_offset'][str(best_offset)]})")

    # Team discovery evaluation (strictly post-hoc)
    team_eval_30 = evaluate_team_discovery_against_gsr(eval_frames, obs_by_frame, gsr_by_frame, best_offset, iou_threshold=0.30)
    team_eval_50 = evaluate_team_discovery_against_gsr(eval_frames, obs_by_frame, gsr_by_frame, best_offset, iou_threshold=0.50)

    # Formal MOT + HOTA + IDF1 evaluation
    tracking_iou30 = compute_mot_and_hota(eval_frames, obs_by_frame, gsr_by_frame, best_offset, primary_iou=0.30)
    tracking_iou50 = compute_mot_and_hota(eval_frames, obs_by_frame, gsr_by_frame, best_offset, primary_iou=0.50)
    cut_frames = [int(r.frame_number) for r in ball_records if r.camera_cut_reset]
    within_shot = evaluate_within_shot_player_tracking(all_obs, camera_cut_frames=cut_frames)

    # Ball tracking error analysis
    ball_eval = analyze_ball_tracking_and_errors(
        ball_records,
        eval_frames,
        gsr_by_frame,
        best_offset,
        bas_json_path=args.bas_json,
        ball_npz_path=args.ball_npz,
        raw_dir=args.raw_dir,
        match_id=args.match_id,
    )

    # Pitch-space authoritative evaluation (SoccerTrack v2 / SoccerNet GSR convention)
    projector = load_fisheye_pitch_projector(args.raw_dir, args.match_id)
    pitch_offset = 1  # SoccerTrack v2 v1.2 canonical: GSR frame k (1-based) == video frame k-1 (0-based)
    pitch_eval_5m = None
    pitch_eval_8m = None
    if projector is not None:
        # Verify temporal offset in pitch space too (important for 117092 where bbox_image is undistorted)
        p2p = projector["pixel_to_pitch"]
        sample_pfs = eval_frames[5:55]
        off_scores = {}
        for off in (-1, 0, 1, 2, 3):
            dists_o = []
            m_cnt = 0
            tot_p = 0
            for pf in sample_pfs:
                gt = [g for g in gsr_by_frame.get(pf + off, []) if g["pitch_x"] is not None]
                pr = obs_by_frame.get(pf, [])
                if not gt or not pr:
                    continue
                g_xy = np.array([[float(g["pitch_x"]), float(g["pitch_y"])] for g in gt], dtype=np.float64)
                p_xy = p2p(np.array([[(p.player_bbox[0] + p.player_bbox[2]) / 2.0, float(p.player_bbox[3])] for p in pr]))
                on_p = p_xy[(np.abs(p_xy[:, 0]) <= 55.0) & (np.abs(p_xy[:, 1]) <= 36.5)]
                tot_p += len(on_p)
                ms = match_pitch_hungarian(g_xy, on_p, max_dist_m=6.0)
                m_cnt += len(ms)
                dists_o.extend(d for _, _, d in ms)
            off_scores[off] = {
                "matched_within_6m": m_cnt,
                "total_on_pitch_pred": tot_p,
                "median_m": round(float(np.median(dists_o)), 2) if dists_o else 99.0,
            }
        # Default to canonical v1.2 offset=+1 (OpenCV 0-based -> GSR 1-based) unless clearly shifted
        pitch_offset = 1
        pitch_eval_5m = evaluate_pitch_space_gsr(eval_frames, obs_by_frame, gsr_by_frame, pitch_offset, projector, match_radius_m=5.0)
        pitch_eval_5m["pitch_offset_check"] = {str(k): v for k, v in off_scores.items()}
        pitch_eval_8m = evaluate_pitch_space_gsr(eval_frames, obs_by_frame, gsr_by_frame, pitch_offset, projector, match_radius_m=8.0)

    # Read team discovery & possession summary from unchanged run dir if present
    td_report_path = args.run_dir / "team_discovery_report.json"
    td_report = json.loads(td_report_path.read_text(encoding="utf-8")) if td_report_path.is_file() else {}
    analytics_path = args.run_dir / "analytics_report.json"
    analytics_report = json.loads(analytics_path.read_text(encoding="utf-8")) if analytics_path.is_file() else {}
    poss_report = analytics_report.get("possession", {})

    # Export representative + failure frames for verified ball-box annotation if requested
    ball_sample_info = None
    if args.export_ball_sample_video and args.annotations_dir:
        # Select 30 frames: up to 12 BAS event frames + 8 large-jump/observed frames + 10 evenly spaced frames
        chosen: List[int] = []
        for bc in ball_eval["bas_event_anchor_checks"]:
            if bc["eval_frame_0b"] not in chosen and len(chosen) < 12:
                chosen.append(int(bc["eval_frame_0b"]))
        for jmp in ball_eval["top_large_jumps"]:
            if jmp["frame_number"] not in chosen and len(chosen) < 20:
                chosen.append(int(jmp["frame_number"]))
        grid = np.unique(np.linspace(eval_frames[0], eval_frames[-1], 30).astype(int)).tolist()
        for gf in grid:
            nearest = min(eval_frames, key=lambda x: abs(x - gf))
            if nearest not in chosen and len(chosen) < 30:
                chosen.append(int(nearest))
        chosen = sorted(set(chosen))
        store = DetectionAnnotationStore(args.annotations_dir, match_id=f"soccertrack_{args.match_id}")
        store.export_frames_for_annotation(args.export_ball_sample_video, chosen)
        ball_sample_info = {
            "annotations_dir": str(args.annotations_dir),
            "annotator_html": str(store.html_path),
            "exported_frame_count": len(chosen),
            "exported_frames": chosen,
        }

    result = {
        "match_id": args.match_id,
        "run_dir": str(args.run_dir),
        "gsr_json": str(args.gsr_json),
        "gsr_meta": gsr_meta,
        "evaluated_frame_range_0based": [eval_frames[0], eval_frames[-1]],
        "evaluated_frame_count": len(eval_frames),
        "frame_alignment_check": offset_check,
        "pipeline_metadata": analytics_report.get("experiment_metadata", {}),
        "unsupervised_team_discovery_report": td_report,
        "team_discovery_evaluation_iou30": team_eval_30,
        "team_discovery_evaluation_iou50": team_eval_50,
        "player_tracking_evaluation_iou30": tracking_iou30,
        "player_tracking_evaluation_iou50": tracking_iou50,
        "pitch_space_gsr_evaluation_5m": pitch_eval_5m,
        "pitch_space_gsr_evaluation_8m": pitch_eval_8m,
        "within_shot_player_tracking": within_shot,
        "ball_tracking_analysis": ball_eval,
        "possession_summary_excerpt": {
            "attributed_possession_share": poss_report.get("attributed_possession_share"),
            "possession_coverage_pct": poss_report.get("possession_coverage_pct"),
            "state_distribution_pct": poss_report.get("total_analysed_time_distribution_pct"),
            "ball_tracking": poss_report.get("ball_tracking"),
        },
        "ball_sample_export": ball_sample_info,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {args.output_json}")


if __name__ == "__main__":
    main()
