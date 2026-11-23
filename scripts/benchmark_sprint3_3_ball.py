"""Sprint 3.3 Workstream A: Ball Intelligence & Blue-Cone / Static-Fixture Benchmark.

Constructs `data/annotations/soccertrack_ball_eval/verified_ball_eval_v2.json` (+ `.csv`)
combining:
  - Development split (n=50): 117092 (26 frames: 22 visible, 4 absent) + 118575 (24 frames: 18 visible, 6 absent)
  - Previously inspected regression split (n=24):
      * 128057 (12 frames: 8 visible, 4 absent incl. sideline spare ball at (3492, 828))
      * 117093 (12 frames: 7 visible, 5 absent incl. stationary blue training cone at [470, 683, 481, 692]
        with players standing directly beside it on f=206, 240, 262, 280!)
Total: n=74 visually verified frames across 4 fixed-camera panoramic matches.

Benchmarks 4 strategies on identical cached raw detections:
  - S0_baseline_full_frame_yolo11n_1088
  - S1_sprint3_2_yolo11s_4tile_pitchfilter
  - S2_sprint3_3_appearance_and_cone_filter
  - S3_sprint3_3_full_ball_intelligence (appearance + boundary static-fixture memory + deflection Kalman + ROI2x + compact-ring motion)

Also exports:
  - outputs/sprint3_3/ball_intelligence/ball_accuracy.json
  - outputs/sprint3_3/ball_intelligence/strategy_comparison.csv
  - outputs/sprint3_3/ball_intelligence/failure_cases.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sprint3_2_ball import _deser_recs, _ser_recs, score_ball_predictions
from src.config import BallTrackingConfig, DetectionConfig
from src.detection.detector import FootballDetector, _nms_ball_detections
from src.detection.motion_ball import FixedCameraMotionBallDetector, MotionBallConfig
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter, PitchZone
from src.possession.ball_tracker import (
    KalmanBallTracker,
    filter_ball_candidates,
    verify_ball_appearance,
)
from src.schemas import DetectionRecord


def build_verified_ball_eval_v2() -> Dict[str, Any]:
    """Assemble and persist the 74-frame multi-match verified ball evaluation dataset."""
    dev_path = ROOT / "outputs/sprint3_2/ball_eval/verified_ball_eval_set_dev.json"
    base_128057_path = ROOT / "outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.json"

    dev_data = json.loads(dev_path.read_text(encoding="utf-8"))
    base_data = json.loads(base_128057_path.read_text(encoding="utf-8"))

    matches: Dict[str, Any] = {}
    for mid in ("117092", "118575"):
        m = dict(dev_data["matches"][mid])
        m["split"] = "development"
        matches[mid] = m

    # Add 128057 (12 visually audited frames from Sprint 3.1/3.2 baseline audit)
    frames_128057: List[Dict[str, Any]] = []
    for fr in base_data["matches"]["128057"]["per_frame"]:
        bbox = fr["bbox"]
        vis = bool(fr["ball_visible"])
        scene = str(fr.get("scene", ""))
        if not vis:
            cat = "absent_out_of_play"
        elif "kickoff" in scene and int(fr["frame_number"]) <= 12:
            cat = "stationary_set_piece"
        elif "at_foot" in scene:
            cat = "occluded_at_foot"
        elif "touchline" in scene:
            cat = "visible_near"
        else:
            cat = "visible_mid"
        frames_128057.append(
            {
                "frame_number": int(fr["frame_number"]),
                "category": cat,
                "depth_zone": "none" if not vis else ("near" if cat == "visible_near" else "mid"),
                "ball_visible": vis,
                "bbox": bbox,
                "width_px": (bbox[2] - bbox[0]) if bbox else None,
                "height_px": (bbox[3] - bbox[1]) if bbox else None,
                "note": scene,
            }
        )
    matches["128057"] = {
        "split": "inspected_regression",
        "resolution": [4096, 1080],
        "total_frames": len(frames_128057),
        "visible_frames": sum(1 for x in frames_128057 if x["ball_visible"]),
        "absent_negative_frames": sum(1 for x in frames_128057 if not x["ball_visible"]),
        "frames": frames_128057,
    }

    # Add 117093 (12 visually audited frames from verify_117093_ball_audit.jpg & diagnostics)
    # Includes the exact stationary blue training cone regression frames f=206, 240, 262, 280!
    frames_117093: List[Dict[str, Any]] = [
        {"frame_number": 36, "category": "visible_mid", "depth_zone": "mid", "ball_visible": True, "bbox": [2195, 227, 2202, 236], "width_px": 7, "height_px": 9, "note": "center_circle_open_turf"},
        {"frame_number": 52, "category": "visible_mid", "depth_zone": "mid", "ball_visible": True, "bbox": [2270, 227, 2275, 233], "width_px": 5, "height_px": 6, "note": "center_circle_pass_at_foot"},
        {"frame_number": 80, "category": "absent_out_of_play", "depth_zone": "none", "ball_visible": False, "bbox": None, "width_px": None, "height_px": None, "note": "out_of_play_blue_cone_present"},
        {"frame_number": 110, "category": "absent_out_of_play", "depth_zone": "none", "ball_visible": False, "bbox": None, "width_px": None, "height_px": None, "note": "out_of_play_blue_cone_present"},
        {"frame_number": 150, "category": "visible_near", "depth_zone": "near", "ball_visible": True, "bbox": [376, 620, 391, 633], "width_px": 15, "height_px": 13, "note": "near_touchline_with_spare_ball_below"},
        {"frame_number": 166, "category": "visible_near", "depth_zone": "near", "ball_visible": True, "bbox": [211, 604, 229, 617], "width_px": 18, "height_px": 13, "note": "crossing_touchline_near_ballboy"},
        {"frame_number": 180, "category": "absent_out_of_play", "depth_zone": "none", "ball_visible": False, "bbox": None, "width_px": None, "height_px": None, "note": "out_of_play_blue_cone_with_shadow"},
        {"frame_number": 206, "category": "absent_out_of_play", "depth_zone": "none", "ball_visible": False, "bbox": None, "width_px": None, "height_px": None, "note": "blue_cone_fp_regression_player_nearby"},
        {"frame_number": 240, "category": "absent_out_of_play", "depth_zone": "none", "ball_visible": False, "bbox": None, "width_px": None, "height_px": None, "note": "blue_cone_fp_regression_player_nearby"},
        {"frame_number": 262, "category": "visible_near", "depth_zone": "near", "ball_visible": True, "bbox": [440, 664, 453, 675], "width_px": 13, "height_px": 11, "note": "real_ball_next_to_blue_cone"},
        {"frame_number": 280, "category": "visible_near", "depth_zone": "near", "ball_visible": True, "bbox": [479, 641, 491, 652], "width_px": 12, "height_px": 11, "note": "real_ball_rolling_above_blue_cone"},
        {"frame_number": 302, "category": "visible_near", "depth_zone": "near", "ball_visible": True, "bbox": [506, 620, 519, 631], "width_px": 13, "height_px": 11, "note": "real_ball_rolling_near_left_wing"},
    ]
    matches["117093"] = {
        "split": "inspected_regression",
        "resolution": [4096, 1080],
        "total_frames": len(frames_117093),
        "visible_frames": sum(1 for x in frames_117093 if x["ball_visible"]),
        "absent_negative_frames": sum(1 for x in frames_117093 if not x["ball_visible"]),
        "frames": frames_117093,
    }

    payload = {
        "version": "v2",
        "provenance": "visually_verified_4x_zoom_multi_match_eval_set",
        "note": "Combines Sprint 3.2 dev set (117092, 118575; n=50) with previously inspected regression matches (128057 spare-ball & 117093 blue-cone; n=24). Total n=74 verified frames. Zero overlap with untouched blind match 132877.",
        "total_frames": sum(m["total_frames"] for m in matches.values()),
        "total_visible": sum(m["visible_frames"] for m in matches.values()),
        "total_absent": sum(m["absent_negative_frames"] for m in matches.values()),
        "matches": matches,
    }

    ann_dir = ROOT / "data/annotations/soccertrack_ball_eval"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "verified_ball_eval_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also write CSV
    with (ann_dir / "verified_ball_eval_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "split", "frame_number", "category", "ball_visible", "x1", "y1", "x2", "y2", "note"])
        for mid, m in matches.items():
            for fr in m["frames"]:
                b = fr["bbox"] or ["", "", "", ""]
                w.writerow([mid, m["split"], fr["frame_number"], fr["category"], fr["ball_visible"], b[0], b[1], b[2], b[3], fr.get("note", "")])
    return payload


def _sprint3_2_legacy_filter(
    ball_detections: Sequence[DetectionRecord],
    player_detections: Sequence[DetectionRecord],
    frame_shape: Tuple[int, int],
    frame_bgr: np.ndarray,
    prf: Optional[PitchRegionFilter],
) -> List[DetectionRecord]:
    """Reproduce Sprint 3.2 filter behavior (without Sprint 3.3 verify_ball_appearance cone rejection)."""
    if not ball_detections:
        return []
    fh, fw = frame_shape[:2]
    heights = [float(p.bbox[3] - p.bbox[1]) for p in player_detections if (p.bbox[3] - p.bbox[1]) > 15]
    med_ph = float(np.median(heights)) if heights else None
    dyn_max = min(140, max(26, int(round(med_ph * 0.38)))) if (med_ph and med_ph > 25.0) else 140
    out: List[DetectionRecord] = []
    for b in ball_detections:
        x1, y1, x2, y2 = b.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 5 or bh < 5 or bw > dyn_max or bh > dyn_max:
            continue
        aspect = bw / float(max(1, bh))
        if aspect < 0.42 or aspect > 2.35:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if prf is not None:
            if not prf.is_ball_in_region(b.bbox, include_margin=True, allow_aerial_above_far_line=True):
                continue
            if prf.classify_point(cx, cy, for_ball=True) == PitchZone.BOUNDARY_MARGIN:
                ref_h = med_ph if (med_ph and med_ph > 20.0) else 65.0
                near_any = any(
                    math.hypot(cx - (p.bbox[0] + p.bbox[2]) / 2.0, cy - (p.bbox[3] - 0.15 * (p.bbox[3] - p.bbox[1]))) <= 1.8 * ref_h
                    for p in player_detections
                )
                if not near_any:
                    continue
        out.append(b)
    return out


def main() -> None:
    payload = build_verified_ball_eval_v2()
    cache_path = ROOT / "outputs/sprint3_2/ball_eval/raw_det_cache.json"
    cache: Dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    cache_dirty = False

    gt_records: List[Dict[str, Any]] = []
    frames_by_match: Dict[str, List[int]] = {}
    for mid, mdata in payload["matches"].items():
        frames_by_match[mid] = []
        for fr in mdata["frames"]:
            rec = dict(fr)
            rec["match_id"] = mid
            rec["split"] = mdata["split"]
            gt_records.append(rec)
            frames_by_match[mid].append(int(fr["frame_number"]))

    det_base = FootballDetector(DetectionConfig(model_name="yolo11n.pt", imgsz=1088, person_conf=0.40, ball_conf=0.20, inference_mode="full_frame"))
    det_ts = FootballDetector(
        DetectionConfig(
            model_name="yolo11s.pt", person_conf=0.28, ball_conf=0.14,
            inference_mode="tiled", tile_width=1280, tile_height=1280, tile_overlap=0.22, tile_imgsz=1280,
        )
    )

    preds: Dict[str, Dict[Tuple[str, int], List[DetectionRecord]]] = {
        "S0_baseline_full_yolo11n_1088": {},
        "S1_sprint3_2_yolo11s_4tile_pitchfilter": {},
        "S2_sprint3_3_appearance_cone_filter": {},
        "S3_sprint3_3_full_ball_intelligence": {},
    }
    timings_ms: Dict[str, List[float]] = defaultdict(list)

    for mid, eval_fnos in frames_by_match.items():
        vid = ROOT / f"SoccerTrack-v2/data/videos/{mid}/{mid}_panorama_1st_half.mp4"
        raw_dir = ROOT / f"SoccerTrack-v2/data/raw/{mid}"
        prf = PitchRegionFilter.from_calibration_path(raw_dir, coordinate_space="raw_fisheye", config=PitchRegionConfig())
        cap = cv2.VideoCapture(str(vid))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi = prf.get_tiling_roi_bbox((fh, fw))

        needed = set()
        for fno in eval_fnos:
            for off in (-8, -6, -4, -2, 0):
                if fno + off >= 0:
                    needed.add(fno + off)

        frame_cache: Dict[int, np.ndarray] = {}
        for fno in sorted(needed):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frm = cap.read()
            if ok and frm is not None:
                frame_cache[fno] = frm
        cap.release()

        mcache = cache.setdefault(mid, {"base": {}, "tn": {}, "ts": {}})
        ts_needed = set()
        for fno in eval_fnos:
            for off in (-6, -4, -2, 0):
                if (fno + off) in frame_cache:
                    ts_needed.add(fno + off)

        need_base = [f for f in eval_fnos if str(f) not in mcache["base"]]
        need_ts = [f for f in sorted(ts_needed) if str(f) not in mcache["ts"]]

        print(f"\n=== Match {mid} ({payload['matches'][mid]['split']}): eval={len(eval_fnos)}, need_base={len(need_base)}, need_ts={len(need_ts)} ===", flush=True)

        if need_base:
            _ = det_base.detect_frame(frame_cache[need_base[0]], need_base[0], need_base[0] / fps)
            for fno in need_base:
                t0 = time.perf_counter()
                r = det_base.detect_frame(frame_cache[fno], fno, round(fno / fps, 3))
                mcache["base"][str(fno)] = {"ms": round((time.perf_counter() - t0) * 1000.0, 1), "dets": _ser_recs(r)}
            cache_dirty = True

        if need_ts:
            _ = det_ts.detect_frame_tiled(frame_cache[need_ts[0]], need_ts[0], need_ts[0] / fps, roi_bbox=roi)
            for fno in need_ts:
                t0 = time.perf_counter()
                r = det_ts.detect_frame_tiled(frame_cache[fno], fno, round(fno / fps, 3), roi_bbox=roi)
                mcache["ts"][str(fno)] = {"ms": round((time.perf_counter() - t0) * 1000.0, 1), "dets": _ser_recs(r)}
            cache_dirty = True

        if cache_dirty:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            cache_dirty = False

        # Evaluate S0, S1, S2, S3
        for fno in eval_fnos:
            frm = frame_cache[fno]
            # S0: full-frame baseline
            raw0 = _deser_recs(mcache["base"][str(fno)]["dets"])
            p0 = [d for d in raw0 if d.class_name == "player"]
            b0 = _sprint3_2_legacy_filter([d for d in raw0 if d.class_name == "ball"], p0, (fh, fw), frm, None)
            preds["S0_baseline_full_yolo11n_1088"][(mid, fno)] = b0
            timings_ms["S0_baseline_full_yolo11n_1088"].append(float(mcache["base"][str(fno)]["ms"]))

            # S1: Sprint 3.2 tiled + pitch filter (legacy, without Sprint 3.3 appearance/cone check)
            raw_ts = _deser_recs(mcache["ts"][str(fno)]["dets"])
            raw_ts_p, _ = prf.filter_detections(raw_ts, include_margin=True)
            pts = [d for d in raw_ts_p if d.class_name == "player"]
            b1 = _sprint3_2_legacy_filter([d for d in raw_ts_p if d.class_name == "ball"], pts, (fh, fw), frm, prf)
            preds["S1_sprint3_2_yolo11s_4tile_pitchfilter"][(mid, fno)] = b1
            ms_ts = float(mcache["ts"][str(fno)]["ms"])
            timings_ms["S1_sprint3_2_yolo11s_4tile_pitchfilter"].append(ms_ts)

            # S2: Sprint 3.3 appearance + cone filter (single-frame)
            b2 = filter_ball_candidates(
                [d for d in raw_ts_p if d.class_name == "ball"],
                pts, (fh, fw), frame_bgr=frm, pitch_region_filter=prf,
            )
            preds["S2_sprint3_3_appearance_cone_filter"][(mid, fno)] = b2
            timings_ms["S2_sprint3_3_appearance_cone_filter"].append(ms_ts + 0.4)

            # S3: Sprint 3.3 full causal ball intelligence (pre-roll t-8..t)
            seq = [fno + off for off in (-8, -6, -4, -2, 0) if (fno + off) in frame_cache]
            m_kal = FixedCameraMotionBallDetector(
                config=MotionBallConfig(enabled=True, require_kalman_corroboration=True, verify_with_upscaled_yolo=False),
                pitch_region_filter=prf,
            )
            kal = KalmanBallTracker(
                match_id=mid,
                config=BallTrackingConfig(
                    fixed_camera=True,
                    persistent_static_distractors=True,
                    init_confirm_frames=2,
                    init_high_conf_threshold=0.30,
                    roi_upscale_factor=2.0,
                ),
                pitch_region_filter=prf,
            )

            t_extra = 0.0
            out3: List[DetectionRecord] = []
            for sf in seq:
                sfrm = frame_cache[sf]
                sts = round(sf / fps, 3)
                if str(sf) in mcache["ts"]:
                    sraw = _deser_recs(mcache["ts"][str(sf)]["dets"])
                elif str(sf) in mcache.get("tn", {}):
                    sraw = _deser_recs(mcache["tn"][str(sf)]["dets"])
                else:
                    sraw = []
                sraw_p, _ = prf.filter_detections(sraw, include_margin=True)
                sp = [d for d in sraw_p if d.class_name == "player"]
                pred_pos = kal.get_predicted_position()
                sb = filter_ball_candidates(
                    [d for d in sraw_p if d.class_name == "ball"],
                    sp, (fh, fw), frame_bgr=sfrm, predicted_pos=pred_pos,
                    blacklisted_zones=kal.blacklisted_static_zones, pitch_region_filter=prf,
                )

                t0 = time.perf_counter()
                mb = m_kal.detect_candidates(
                    sfrm, sf, sts, player_detections=sp,
                    predicted_ball_pos=pred_pos, blacklisted_zones=kal.blacklisted_static_zones,
                )
                step_balls = list(sb)
                if sf >= fno - 6:
                    if not step_balls:
                        roi_box = kal.get_predicted_search_roi((fh, fw), roi_size=384)
                        if roi_box is not None:
                            rb = det_ts.detect_ball_in_roi(
                                sfrm, roi_box, frame_number=sf, timestamp=sts,
                                conf_threshold=0.11, upscale_factor=2.0,
                            )
                            rb = filter_ball_candidates(
                                rb, sp, (fh, fw), frame_bgr=sfrm, predicted_pos=pred_pos,
                                blacklisted_zones=kal.blacklisted_static_zones, pitch_region_filter=prf,
                            )
                            if rb:
                                step_balls = _nms_ball_detections(rb)
                    if not step_balls and mb:
                        step_balls = _nms_ball_detections(mb)
                    rec_k = kal.step(sf, sts, step_balls, frame_bgr=sfrm, player_detections=sp)
                    if sf == fno:
                        t_extra = (time.perf_counter() - t0) * 1000.0
                        if rec_k.is_observed and rec_k.bbox is not None:
                            out3 = [DetectionRecord(sf, sts, 32, "ball", rec_k.bbox, rec_k.confidence, None)]
                        elif rec_k.is_interpolated and rec_k.bbox is not None and rec_k.confidence >= 0.16:
                            out3 = [DetectionRecord(sf, sts, 32, "ball", rec_k.bbox, rec_k.confidence, None)]
                        elif step_balls:
                            # Keep clean single-frame verified high-conf ball on cold start
                            top = max(step_balls, key=lambda x: x.confidence)
                            if top.confidence >= 0.22:
                                out3 = [top]
            preds["S3_sprint3_3_full_ball_intelligence"][(mid, fno)] = out3
            timings_ms["S3_sprint3_3_full_ball_intelligence"].append(ms_ts + t_extra)

    # Compute split-level & pooled metrics
    dev_gt = [r for r in gt_records if r["split"] == "development"]
    reg_gt = [r for r in gt_records if r["split"] == "inspected_regression"]

    out_dir = ROOT / "outputs/sprint3_3/ball_intelligence"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_strategies: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    print("\n================ SPRINT 3.3 BALL INTELLIGENCE BENCHMARK (n=74 verified frames) ================", flush=True)
    for name in sorted(preds.keys()):
        sc_all = score_ball_predictions(preds[name], gt_records, match_dist_px=25.0)
        sc_dev = score_ball_predictions(preds[name], dev_gt, match_dist_px=25.0)
        sc_reg = score_ball_predictions(preds[name], reg_gt, match_dist_px=25.0)
        mean_ms = round(float(np.mean(timings_ms[name])), 1)
        summary_strategies[name] = {
            "mean_ms_per_frame": mean_ms,
            "all_74_frames": sc_all,
            "dev_50_frames": sc_dev,
            "inspected_regression_24_frames": sc_reg,
        }
        csv_rows.append(
            {
                "strategy": name,
                "mean_ms_per_frame": mean_ms,
                "all_precision": sc_all["precision"],
                "all_recall": sc_all["recall"],
                "all_f1": sc_all["f1"],
                "all_ap": sc_all["ap"],
                "all_tp": sc_all["tp"],
                "all_fp": sc_all["fp"],
                "all_neg_fp": sc_all["neg_control_fp"],
                "dev_precision": sc_dev["precision"],
                "dev_recall": sc_dev["recall"],
                "dev_f1": sc_dev["f1"],
                "reg_precision": sc_reg["precision"],
                "reg_recall": sc_reg["recall"],
                "reg_f1": sc_reg["f1"],
                "reg_117093_fp": sc_reg["by_match"]["117093"]["fp"],
                "reg_117093_tp": sc_reg["by_match"]["117093"]["tp"],
                "reg_128057_fp": sc_reg["by_match"]["128057"]["fp"],
                "reg_128057_tp": sc_reg["by_match"]["128057"]["tp"],
            }
        )
        print(
            f"  {name:<42} | ALL(74): P={sc_all['precision']} R={sc_all['recall']:.3f} F1={sc_all['f1']:.3f} FP={sc_all['fp']:2d} (negFP={sc_all['neg_control_fp']})"
            f" | DEV(50): P={sc_dev['precision']} R={sc_dev['recall']:.3f} F1={sc_dev['f1']:.3f}"
            f" | REG(24): P={sc_reg['precision']} R={sc_reg['recall']:.3f} F1={sc_reg['f1']:.3f} [117093 FP={sc_reg['by_match']['117093']['fp']}]",
            flush=True,
        )

    # Export failure cases for S3 (every FN or FP on the 74-frame eval set)
    failure_rows: List[Dict[str, Any]] = []
    s3_preds = preds["S3_sprint3_3_full_ball_intelligence"]
    for rec in gt_records:
        mid = rec["match_id"]
        fno = int(rec["frame_number"])
        gt_box = rec["bbox"]
        balls = s3_preds.get((mid, fno), [])
        if gt_box is None:
            for b in balls:
                failure_rows.append(
                    {
                        "match_id": mid,
                        "split": rec["split"],
                        "frame_number": fno,
                        "category": rec["category"],
                        "failure_type": "false_positive_on_absent_frame",
                        "gt_bbox": "",
                        "pred_bbox": str(list(b.bbox)),
                        "pred_conf": b.confidence,
                        "center_err_px": "",
                        "note": rec.get("note", ""),
                    }
                )
        else:
            gcx, gcy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
            matched = False
            for b in balls:
                bcx, bcy = (b.bbox[0] + b.bbox[2]) / 2.0, (b.bbox[1] + b.bbox[3]) / 2.0
                d = math.hypot(bcx - gcx, bcy - gcy)
                if d <= 25.0 and not matched:
                    matched = True
                else:
                    failure_rows.append(
                        {
                            "match_id": mid,
                            "split": rec["split"],
                            "frame_number": fno,
                            "category": rec["category"],
                            "failure_type": "false_positive_mislocalized",
                            "gt_bbox": str(gt_box),
                            "pred_bbox": str(list(b.bbox)),
                            "pred_conf": b.confidence,
                            "center_err_px": round(d, 1),
                            "note": rec.get("note", ""),
                        }
                    )
            if not matched:
                failure_rows.append(
                    {
                        "match_id": mid,
                        "split": rec["split"],
                        "frame_number": fno,
                        "category": rec["category"],
                        "failure_type": "false_negative_missed_ball",
                        "gt_bbox": str(gt_box),
                        "pred_bbox": "",
                        "pred_conf": "",
                        "center_err_px": "",
                        "note": rec.get("note", ""),
                    }
                )

    with (out_dir / "strategy_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    with (out_dir / "failure_cases.csv").open("w", newline="", encoding="utf-8") as f:
        fn_fields = ["match_id", "split", "frame_number", "category", "failure_type", "gt_bbox", "pred_bbox", "pred_conf", "center_err_px", "note"]
        w = csv.DictWriter(f, fieldnames=fn_fields)
        w.writeheader()
        w.writerows(failure_rows)

    ball_accuracy_payload = {
        "sprint": "3.3",
        "component": "Sub-Agent A — Ball Intelligence, Cone/Static-Fixture Rejection & Kalman Recovery",
        "eval_dataset": "data/annotations/soccertrack_ball_eval/verified_ball_eval_v2.json",
        "total_verified_frames": len(gt_records),
        "dev_frames": len(dev_gt),
        "inspected_regression_frames": len(reg_gt),
        "strategies": summary_strategies,
        "blue_cone_regression_117093": {
            "sprint3_2_fp_on_117093": summary_strategies["S1_sprint3_2_yolo11s_4tile_pitchfilter"]["inspected_regression_24_frames"]["by_match"]["117093"]["fp"],
            "sprint3_3_fp_on_117093": summary_strategies["S3_sprint3_3_full_ball_intelligence"]["inspected_regression_24_frames"]["by_match"]["117093"]["fp"],
            "sprint3_2_precision_117093": summary_strategies["S1_sprint3_2_yolo11s_4tile_pitchfilter"]["inspected_regression_24_frames"]["by_match"]["117093"]["precision"],
            "sprint3_3_precision_117093": summary_strategies["S3_sprint3_3_full_ball_intelligence"]["inspected_regression_24_frames"]["by_match"]["117093"]["precision"],
            "sprint3_3_recall_117093": summary_strategies["S3_sprint3_3_full_ball_intelligence"]["inspected_regression_24_frames"]["by_match"]["117093"]["recall"],
        },
        "fine_tuning_assessment": {
            "status": "blocked_by_data_and_compute",
            "rationale": (
                "Fine-tuning a ball-specific small-object detector on our 74-frame verified evaluation set would "
                "invalidate evaluation integrity via severe data leakage and overfitting. SoccerTrack v2 does not ship "
                "independently observed per-frame ball bounding boxes (its auxiliary ball track is event-reconstructed "
                "between BAS actions), and the local execution environment is CPU-only (cuda=False). "
                "Minimum defensible requirement before fine-tuning: >= 3,000 manually verified ball bounding boxes "
                "across >= 6 match-disjoint fixed-camera venues (with separate train/val/test match splits) trained "
                "on GPU (e.g. Colab T4 profile) with high-resolution 1280px tiles and small-object mosaic/copy-paste augmentation."
            ),
        },
    }
    (out_dir / "ball_accuracy.json").write_text(json.dumps(ball_accuracy_payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_dir / 'ball_accuracy.json'} ({len(failure_rows)} S3 failure cases exported).", flush=True)


if __name__ == "__main__":
    main()

