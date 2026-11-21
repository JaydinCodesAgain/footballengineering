"""Sprint 3.2 Phase D Benchmark: Small-Football Detection Ablation Study (v2 with caching & real per-frame masks).

Evaluates 6 football detection strategies on the 50-frame visually verified
development benchmark (`outputs/sprint3_2/ball_eval/verified_ball_eval_set_dev.json`,
spanning 117092 night 3840x1906 and 118575 daytime 4096x1080 across near/mid/far,
at-foot/occluded, aerial/motion-blur, stationary, and absent out-of-play negative controls):

1. baseline_full_yolo11n_1088: Unchanged full-frame generic COCO detector (no pitch filter)
2. tiled_yolo11n_1280_pitch:   4-tile native-res YOLO11n + PitchRegionFilter + FP filter
3. tiled_yolo11s_1280_pitch:   4-tile native-res YOLO11s + PitchRegionFilter + FP filter
4. tiled_yolo11s_plus_motion3f: #3 + FixedCameraMotionBallDetector (signed positive diff + 3-frame collinear kinematics)
5. tiled_yolo11s_plus_motion3f_yolo2x: #4 + 2x-upscaled local YOLO11s verification on motion trajectories
6. tiled_yolo11s_motion3f_kalman_roi2x: #3 + Motion3f + Kalman trajectory prediction & 2x-upscaled ROI recovery

Caches raw per-frame detections on disk so post-filter/motion/Kalman ablations are fast and reproducible.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scripts.benchmark_sprint3_2_tiling import compute_ap_from_matches
from src.config import BallTrackingConfig, DetectionConfig
from src.detection.detector import FootballDetector, _nms_ball_detections
from src.detection.motion_ball import FixedCameraMotionBallDetector, MotionBallConfig
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter
from src.possession.ball_tracker import KalmanBallTracker, filter_ball_candidates
from src.schemas import DetectionRecord


def _ser_recs(recs: Sequence[DetectionRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "f": r.frame_number,
            "t": r.timestamp,
            "cid": r.class_id,
            "cname": r.class_name,
            "bbox": list(r.bbox),
            "conf": r.confidence,
        }
        for r in recs
    ]


def _deser_recs(rows: Sequence[Dict[str, Any]]) -> List[DetectionRecord]:
    return [
        DetectionRecord(
            frame_number=int(x["f"]),
            timestamp=float(x["t"]),
            class_id=int(x["cid"]),
            class_name=str(x["cname"]),
            bbox=tuple(int(v) for v in x["bbox"]),  # type: ignore[arg-type]
            confidence=float(x["conf"]),
            tracking_id=None,
        )
        for x in rows
    ]


def score_ball_predictions(
    preds_by_key: Dict[Tuple[str, int], List[DetectionRecord]],
    gt_records: List[Dict[str, Any]],
    match_dist_px: float = 25.0,
) -> Dict[str, Any]:
    tp = fp = fn = neg_fp = neg_total = vis_total = 0
    scored: List[Tuple[float, bool]] = []
    center_errs: List[float] = []
    by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "tp": 0, "fp": 0, "fn": 0})
    by_match: Dict[str, Dict[str, int]] = defaultdict(lambda: {"vis": 0, "neg": 0, "tp": 0, "fp": 0, "fn": 0, "neg_fp": 0})

    for rec in gt_records:
        mid = rec["match_id"]
        fno = int(rec["frame_number"])
        cat = rec["category"]
        gt_box = rec["bbox"]
        balls = [d for d in preds_by_key.get((mid, fno), []) if d.class_name == "ball"]
        by_cat[cat]["total"] += 1

        if gt_box is None:
            neg_total += 1
            by_match[mid]["neg"] += 1
            for b in balls:
                fp += 1
                neg_fp += 1
                by_cat[cat]["fp"] += 1
                by_match[mid]["fp"] += 1
                by_match[mid]["neg_fp"] += 1
                scored.append((float(b.confidence), False))
            continue

        vis_total += 1
        by_match[mid]["vis"] += 1
        gcx, gcy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
        matched = False
        for b in sorted(balls, key=lambda x: x.confidence, reverse=True):
            bcx, bcy = (b.bbox[0] + b.bbox[2]) / 2.0, (b.bbox[1] + b.bbox[3]) / 2.0
            dist = math.hypot(bcx - gcx, bcy - gcy)
            if not matched and dist <= match_dist_px:
                matched = True
                tp += 1
                by_cat[cat]["tp"] += 1
                by_match[mid]["tp"] += 1
                center_errs.append(dist)
                scored.append((float(b.confidence), True))
            else:
                fp += 1
                by_cat[cat]["fp"] += 1
                by_match[mid]["fp"] += 1
                scored.append((float(b.confidence), False))
        if not matched:
            fn += 1
            by_cat[cat]["fn"] += 1
            by_match[mid]["fn"] += 1

    prec = tp / float(tp + fp) if (tp + fp) > 0 else None
    rec_v = tp / float(max(1, vis_total))
    f1 = (2.0 * prec * rec_v / (prec + rec_v)) if (prec is not None and (prec + rec_v) > 0) else 0.0
    ap = compute_ap_from_matches(scored, vis_total)

    cat_summary = {}
    for cat, st in sorted(by_cat.items()):
        if cat == "absent_out_of_play":
            cat_summary[cat] = {"total": st["total"], "fp": st["fp"], "fp_per_frame": round(st["fp"] / max(1, st["total"]), 3)}
        else:
            cat_summary[cat] = {
                "total": st["total"],
                "tp": st["tp"],
                "fp": st["fp"],
                "recall": round(st["tp"] / max(1, st["total"]), 4),
            }

    match_summary = {}
    for mid, st in sorted(by_match.items()):
        mp = st["tp"] / float(st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else None
        mr = st["tp"] / float(max(1, st["vis"]))
        match_summary[mid] = {
            "visible": st["vis"],
            "absent": st["neg"],
            "tp": st["tp"],
            "fp": st["fp"],
            "neg_fp": st["neg_fp"],
            "precision": round(mp, 4) if mp is not None else None,
            "recall": round(mr, 4),
        }

    return {
        "visible_total": vis_total,
        "absent_total": neg_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "neg_control_fp": neg_fp,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec_v, 4),
        "f1": round(f1, 4),
        "ap": round(ap, 4),
        "median_center_err_px": round(float(np.median(center_errs)), 2) if center_errs else None,
        "by_match": match_summary,
        "by_category": cat_summary,
    }


def main() -> None:
    eval_path = Path("outputs/sprint3_2/ball_eval/verified_ball_eval_set_dev.json")
    cache_path = Path("outputs/sprint3_2/ball_eval/raw_det_cache.json")
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    gt_records: List[Dict[str, Any]] = []
    frames_by_match: Dict[str, List[int]] = {}
    for mid, mdata in data["matches"].items():
        frames_by_match[mid] = []
        for fr in mdata["frames"]:
            rec = dict(fr)
            rec["match_id"] = mid
            gt_records.append(rec)
            frames_by_match[mid].append(int(fr["frame_number"]))

    det_base = FootballDetector(
        DetectionConfig(model_name="yolo11n.pt", imgsz=1088, person_conf=0.40, ball_conf=0.20, inference_mode="full_frame")
    )
    det_tn = FootballDetector(
        DetectionConfig(
            model_name="yolo11n.pt", person_conf=0.28, ball_conf=0.14,
            inference_mode="tiled", tile_width=1280, tile_height=1280, tile_overlap=0.22, tile_imgsz=1280,
        )
    )
    det_ts = FootballDetector(
        DetectionConfig(
            model_name="yolo11s.pt", person_conf=0.28, ball_conf=0.14,
            inference_mode="tiled", tile_width=1280, tile_height=1280, tile_overlap=0.22, tile_imgsz=1280,
        )
    )

    cache: Dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    cache_dirty = False

    preds: Dict[str, Dict[Tuple[str, int], List[DetectionRecord]]] = {
        "1_baseline_full_yolo11n_1088": {},
        "2_tiled_yolo11n_1280_pitch": {},
        "3_tiled_yolo11s_1280_pitch": {},
        "4_tiled_yolo11s_plus_motion3f_unverified": {},
        "5_tiled_yolo11s_plus_motion3f_yolo2x": {},
        "6_tiled_yolo11s_kalman_roi2x_gated_motion": {},
    }
    timings_ms: Dict[str, List[float]] = defaultdict(list)

    for mid, eval_fnos in frames_by_match.items():
        vid = Path(f"SoccerTrack-v2/data/videos/{mid}/{mid}_panorama_1st_half.mp4")
        raw_dir = Path(f"SoccerTrack-v2/data/raw/{mid}")
        prf = PitchRegionFilter.from_calibration_path(raw_dir, coordinate_space="raw_fisheye", config=PitchRegionConfig())
        cap = cv2.VideoCapture(str(vid))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi = prf.get_tiling_roi_bbox((fh, fw))

        # Causal pre-roll: t-8 (bg init), t-6, t-4, t-2, t (gives 3 full motion steps t-6->t-4->t-2->t!)
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
        print(f"\n=== Match {mid}: loaded {len(frame_cache)} frames ({len(eval_fnos)} eval), roi={roi} ===", flush=True)

        # Populate raw detection cache (once per frame)
        # Note: tn is run on ALL needed frames (including t-6, t-4, t-2) so motion masking ALWAYS has real per-frame players!
        # ts is run on eval + t-4, t-2 (for Kalman lock + tiled_s baseline).
        ts_needed = set()
        for fno in eval_fnos:
            for off in (-4, -2, 0):
                if (fno + off) in frame_cache:
                    ts_needed.add(fno + off)

        need_base = [f for f in eval_fnos if str(f) not in mcache["base"]]
        need_tn = [f for f in sorted(needed) if str(f) not in mcache["tn"]]
        need_ts = [f for f in sorted(ts_needed) if str(f) not in mcache["ts"]]

        if need_base:
            _ = det_base.detect_frame(frame_cache[need_base[0]], need_base[0], need_base[0] / fps)
            for fno in need_base:
                t0 = time.perf_counter()
                r = det_base.detect_frame(frame_cache[fno], fno, round(fno / fps, 3))
                mcache["base"][str(fno)] = {"ms": round((time.perf_counter() - t0) * 1000.0, 1), "dets": _ser_recs(r)}
            cache_dirty = True

        if need_tn:
            _ = det_tn.detect_frame_tiled(frame_cache[need_tn[0]], need_tn[0], need_tn[0] / fps, roi_bbox=roi)
            for fno in need_tn:
                t0 = time.perf_counter()
                r = det_tn.detect_frame_tiled(frame_cache[fno], fno, round(fno / fps, 3), roi_bbox=roi)
                mcache["tn"][str(fno)] = {"ms": round((time.perf_counter() - t0) * 1000.0, 1), "dets": _ser_recs(r)}
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

        # Helper to filter cached detections cleanly with current Sprint 3.2 pitch + FP rules
        def get_filtered(model_key: str, fno: int, use_pitch: bool = True) -> Tuple[List[DetectionRecord], List[DetectionRecord], float]:
            entry = mcache[model_key][str(fno)]
            raw = _deser_recs(entry["dets"])
            frm = frame_cache[fno]
            if use_pitch:
                raw, _ = prf.filter_detections(raw, include_margin=True)
            pl = [d for d in raw if d.class_name == "player"]
            bl = filter_ball_candidates(
                [d for d in raw if d.class_name == "ball"],
                pl, (fh, fw), frame_bgr=frm, pitch_region_filter=prf if use_pitch else None,
            )
            return pl, bl, float(entry["ms"])

        # Strategies 1, 2, 3
        for fno in eval_fnos:
            _, b1, ms1 = get_filtered("base", fno, use_pitch=False)
            _, b2, ms2 = get_filtered("tn", fno, use_pitch=True)
            _, b3, ms3 = get_filtered("ts", fno, use_pitch=True)
            preds["1_baseline_full_yolo11n_1088"][(mid, fno)] = b1
            preds["2_tiled_yolo11n_1280_pitch"][(mid, fno)] = b2
            preds["3_tiled_yolo11s_1280_pitch"][(mid, fno)] = b3
            timings_ms["1_baseline_full_yolo11n_1088"].append(ms1)
            timings_ms["2_tiled_yolo11n_1280_pitch"].append(ms2)
            timings_ms["3_tiled_yolo11s_1280_pitch"].append(ms3)

        # Strategies 4, 5, 6 (causal short-window streaming with real per-frame players)
        for fno in eval_fnos:
            seq = [fno + off for off in (-8, -6, -4, -2, 0) if (fno + off) in frame_cache]
            m_raw = FixedCameraMotionBallDetector(
                config=MotionBallConfig(enabled=True, require_kalman_corroboration=False, verify_with_upscaled_yolo=False),
                pitch_region_filter=prf,
            )
            m_ver = FixedCameraMotionBallDetector(
                config=MotionBallConfig(enabled=True, require_kalman_corroboration=False, verify_with_upscaled_yolo=True, verify_upscale_factor=2.0, verify_conf_threshold=0.10),
                pitch_region_filter=prf,
            )
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

            out4 = out5 = out6 = []
            t4_ex = t5_ex = t6_ex = 0.0
            for sf in seq:
                sfrm = frame_cache[sf]
                sts = round(sf / fps, 3)
                if str(sf) in mcache["ts"]:
                    sp, sb, _ = get_filtered("ts", sf, use_pitch=True)
                else:
                    sp, sb, _ = get_filtered("tn", sf, use_pitch=True)

                t0 = time.perf_counter()
                mb4 = m_raw.detect_candidates(sfrm, sf, sts, player_detections=sp)
                if sf == fno:
                    t4_ex = (time.perf_counter() - t0) * 1000.0
                    out4 = _nms_ball_detections(sb + mb4) if not sb else sb

                t0 = time.perf_counter()
                mb5 = m_ver.detect_candidates(
                    sfrm, sf, sts, player_detections=sp,
                    yolo_detector=det_ts if (sf == fno and not sb) else None,
                )
                if sf == fno:
                    t5_ex = (time.perf_counter() - t0) * 1000.0
                    out5 = _nms_ball_detections(sb + mb5) if not sb else sb

                t0 = time.perf_counter()
                pred_pos = kal.get_predicted_position()
                mb6 = m_kal.detect_candidates(
                    sfrm, sf, sts, player_detections=sp,
                    predicted_ball_pos=pred_pos,
                    blacklisted_zones=kal.blacklisted_static_zones,
                )
                step_balls = list(sb)
                if sf >= fno - 4:
                    if not step_balls:
                        roi_box = kal.get_predicted_search_roi((fh, fw), roi_size=384)
                        if roi_box is not None:
                            rb = det_ts.detect_ball_in_roi(
                                sfrm, roi_box, frame_number=sf, timestamp=sts,
                                conf_threshold=0.10, upscale_factor=2.0,
                            )
                            rb = filter_ball_candidates(
                                rb, sp, (fh, fw), frame_bgr=sfrm, predicted_pos=pred_pos,
                                blacklisted_zones=kal.blacklisted_static_zones, pitch_region_filter=prf,
                            )
                            if rb:
                                step_balls = _nms_ball_detections(rb)
                    if not step_balls and mb6:
                        step_balls = _nms_ball_detections(mb6)
                    rec_k = kal.step(sf, sts, step_balls, frame_bgr=sfrm, player_detections=sp)
                    if sf == fno:
                        t6_ex = (time.perf_counter() - t0) * 1000.0
                        if step_balls:
                            out6 = step_balls
                        elif rec_k.is_interpolated and rec_k.bbox is not None:
                            out6 = [DetectionRecord(sf, sts, 32, "ball", rec_k.bbox, rec_k.confidence, None)]
                        else:
                            out6 = []

            base_ms = float(mcache["ts"][str(fno)]["ms"])
            preds["4_tiled_yolo11s_plus_motion3f_unverified"][(mid, fno)] = out4
            preds["5_tiled_yolo11s_plus_motion3f_yolo2x"][(mid, fno)] = out5
            preds["6_tiled_yolo11s_kalman_roi2x_gated_motion"][(mid, fno)] = out6
            timings_ms["4_tiled_yolo11s_plus_motion3f_unverified"].append(base_ms + t4_ex)
            timings_ms["5_tiled_yolo11s_plus_motion3f_yolo2x"].append(base_ms + t5_ex)
            timings_ms["6_tiled_yolo11s_kalman_roi2x_gated_motion"].append(base_ms + t6_ex)

    summary = {"eval_set_path": str(eval_path), "strategies": {}}
    print("\n================ SMALL-FOOTBALL ABLATION RESULTS v2 (n=50 verified dev frames: 40 vis, 10 neg) ================", flush=True)
    for name in sorted(preds.keys()):
        sc = score_ball_predictions(preds[name], gt_records, match_dist_px=25.0)
        sc["mean_ms_per_frame"] = round(float(np.mean(timings_ms[name])), 1)
        summary["strategies"][name] = sc
        p_str = f"{sc['precision']:.3f}" if sc["precision"] is not None else "N/A  "
        print(
            f"  {name:<40} | P={p_str} R={sc['recall']:.3f} F1={sc['f1']:.3f} AP={sc['ap']:.3f} "
            f"| TP={sc['tp']:2d}/40 FP={sc['fp']:2d} (negFP={sc['neg_control_fp']:2d}/10) | err={sc['median_center_err_px']}px | {sc['mean_ms_per_frame']:6.0f}ms/f",
            flush=True,
        )
        print(f"      by_match: {sc['by_match']}", flush=True)
        print(f"      by_cat  : { {k: (v.get('recall', v.get('fp'))) for k, v in sc['by_category'].items()} }", flush=True)

    out_json = Path("outputs/sprint3_2/ball_eval/ball_ablation_results.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json}", flush=True)


if __name__ == "__main__":
    main()
