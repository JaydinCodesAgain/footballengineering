"""Reproducible Sprint 3.1 Stage C benchmark: Ball failure analysis, filter verification,
and multi-model/multi-strategy ball detection comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.config import DetectionConfig
from src.detection.detector import (
    FootballDetector,
    benchmark_ball_detection_strategies,
)
from src.evaluation.detection_eval import evaluate_legacy_manual_counts
from src.evaluation.report import write_csv_rows
from src.pipeline.legacy_migration import LEGACY_NORWAY_WALES_DETECTION_COUNTS
from src.possession.ball_tracker import (
    estimate_pitch_horizon_y,
    filter_ball_candidates,
)
from src.schemas import DetectionRecord


def run_failure_case_audit(out_dir: Path) -> List[Dict[str, Any]]:
    """Audit visually verified real failure frames from Sprint 3 baseline and verify
    that Sprint 3.1 filters reject false positives while preserving true football detections.
    """
    frames_dir = out_dir / "failure_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Visually verified real cases from Stage A inspection (Man Utd vs Villa + Chelsea + Norway)
    cases = [
        # (match_id, video_path, frame_no, bbox, conf, category, is_true_ball, notes)
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 162, (1069, 115, 1087, 138), 0.2398, "crowd_high_vis_steward", False, "Neon-yellow steward vest in stands above pitch"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 296, (619, 109, 635, 129), 0.1780, "crowd_above_horizon", False, "Crowd artifact above pitch horizon (381px jump)"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 322, (883, 122, 897, 134), 0.7361, "crowd_high_vis_steward", False, "High-confidence (0.74) yellow steward vest in stands"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 446, (772, 65, 786, 81), 0.6534, "crowd_high_vis_steward", False, "High-confidence (0.65) yellow steward vest in upper stands"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 98, (944, 374, 953, 379), 0.2078, "pure_turf_phantom", False, "Uniform green grass phantom while real ball is at (675,405)"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 166, (506, 322, 514, 331), 0.2013, "white_sock_or_boot_fp", False, "601px teleport onto player white sock while ball in net"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 14, (881, 319, 894, 332), 0.4631, "white_sock_jump_outlier", False, "341px jump to player boot; rejected by Kalman motion gate (not static box)"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 18, (431, 316, 440, 324), 0.5535, "true_ball_on_pitch", True, "Real match football in open play on pitch (must be preserved)"),
        ("man_utd_villa", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", 650, (494, 480, 508, 494), 0.6200, "true_ball_at_feet", True, "Real match football at Man Utd #10 feet (must be preserved)"),
        ("norway_wales", "Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4", 1640, (582, 636, 665, 718), 0.4100, "closeup_shorts_oversized", False, "Sprint 2 failure: 82x82 box on player shorts in close-up"),
    ]

    detector = FootballDetector(DetectionConfig(model_name="yolo11n.pt", imgsz=1088))
    rows: List[Dict[str, Any]] = []

    caps: Dict[str, cv2.VideoCapture] = {}
    try:
        for mid, vpath, fno, bbox, conf, category, is_true_ball, notes in cases:
            if vpath not in caps:
                caps[vpath] = cv2.VideoCapture(vpath)
            cap = caps[vpath]
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frm = cap.read()
            if not ok or frm is None:
                continue
            fh, fw = frm.shape[:2]
            ts = round(fno / fps, 3)

            dets = detector.detect_frame(frm, fno, ts, person_conf_override=0.40, ball_conf_override=0.99)
            players = [d for d in dets if d.class_name == "player"]
            cand = DetectionRecord(fno, ts, 32, "ball", bbox, conf, None)

            horizon_y = estimate_pitch_horizon_y(frm)
            kept = filter_ball_candidates(
                [cand],
                players,
                (fh, fw),
                frame_bgr=frm,
                predicted_pos=None,
                filter_crowd_horizon=True,
            )
            kept_flag = len(kept) == 1

            # Save compact annotated diagnostic thumbnail
            vis = frm.copy()
            if horizon_y is not None:
                cv2.line(vis, (0, horizon_y), (fw, horizon_y), (255, 220, 0), 1)
            col = (0, 220, 0) if is_true_ball else (0, 0, 255)
            cv2.rectangle(vis, (bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2), col, 2)
            status = "KEPT" if kept_flag else "REJECTED"
            cv2.putText(
                vis,
                f"{category} | f={fno} ({ts:.2f}s) c={conf:.2f} -> S3.1:{status}",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                col,
                2,
            )
            thumb = cv2.resize(vis, (640, 360), interpolation=cv2.INTER_AREA)
            img_name = f"{mid}_f{fno:04d}_{category}.jpg"
            cv2.imwrite(str(frames_dir / img_name), thumb)

            rows.append(
                {
                    "match_id": mid,
                    "frame_number": fno,
                    "timestamp_s": ts,
                    "bbox": str(list(bbox)),
                    "confidence": round(conf, 4),
                    "failure_category": category,
                    "is_true_ball": is_true_ball,
                    "estimated_pitch_horizon_y": horizon_y,
                    "sprint3_baseline_filter_kept": True,
                    "sprint3_1_filter_kept": kept_flag,
                    "correct_filter_decision": (kept_flag == is_true_ball),
                    "diagnostic_image": f"failure_frames/{img_name}",
                    "notes": notes,
                }
            )
    finally:
        for cap in caps.values():
            cap.release()

    write_csv_rows(out_dir / "false_positive_examples.csv", rows)
    return rows


def extract_frames(video_path: str, frame_numbers: List[int]) -> List[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    out: List[Tuple[int, float, np.ndarray]] = []
    try:
        for fno in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frm = cap.read()
            if ok and frm is not None:
                out.append((int(fno), round(fno / fps, 3), frm))
    finally:
        cap.release()
    return out


def main() -> None:
    import sys

    out_dir = Path("outputs/sprint3_1/ball_tracking")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Running real failure-frame audit & Sprint 3.1 filter verification...")
    fp_rows = run_failure_case_audit(out_dir)
    for r in fp_rows:
        print(
            f"  f={r['frame_number']:4d} | {r['failure_category']:24s} | true={str(r['is_true_ball']):5s} "
            f"| horizon={str(r['estimated_pitch_horizon_y']):4s} | S3.1_kept={str(r['sprint3_1_filter_kept']):5s}"
        )
    if "--audit-only" in sys.argv:
        det_json = out_dir / "detection_metrics.json"
        if det_json.is_file():
            d = json.loads(det_json.read_text(encoding="utf-8"))
            d["failure_case_filter_audit"]["false_positives_rejected_by_static_filter"] = sum(
                1 for r in fp_rows if (not r["is_true_ball"]) and (not r["sprint3_1_filter_kept"])
            )
            d["failure_case_filter_audit"]["true_balls_preserved"] = sum(
                1 for r in fp_rows if r["is_true_ball"] and r["sprint3_1_filter_kept"]
            )
            det_json.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return

    print("\n2. Benchmarking YOLO11n vs YOLO11s detection strategies on real match frames...")
    nw_fnos = sorted(LEGACY_NORWAY_WALES_DETECTION_COUNTS.keys())
    cs_fnos = list(range(840, 1840, 80))  # 13 wide-angle tactical frames from Chelsea vs Sunderland
    nw_frames = extract_frames("Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4", nw_fnos)
    cs_frames = extract_frames("27.mp4", cs_fnos)

    # Benchmark YOLO11n on both sets
    nw_bench_n = benchmark_ball_detection_strategies(nw_frames, verified_annotations=None, model_name="yolo11n.pt")
    cs_bench_n = benchmark_ball_detection_strategies(cs_frames, verified_annotations=None, model_name="yolo11n.pt")

    # Benchmark YOLO11s on the exact same frames (if downloadable/available)
    nw_bench_s: List[Dict[str, Any]] = []
    cs_bench_s: List[Dict[str, Any]] = []
    yolo11s_status = "evaluated"
    try:
        det_s = FootballDetector(DetectionConfig(model_name="yolo11s.pt", imgsz=1088))
        _ = det_s.model  # triggers load/download
        nw_bench_s = benchmark_ball_detection_strategies(nw_frames, verified_annotations=None, model_name="yolo11s.pt")
        cs_bench_s = benchmark_ball_detection_strategies(cs_frames, verified_annotations=None, model_name="yolo11s.pt")
    except Exception as exc:
        yolo11s_status = f"unavailable: {exc}"

    # Also compute legacy 11-frame visual count reference on Norway vs Wales (explicitly flagged iou_verified=False)
    legacy_counts_eval: Dict[str, Any] = {}
    for m_name in ("yolo11n.pt", "yolo11s.pt"):
        if m_name == "yolo11s.pt" and yolo11s_status != "evaluated":
            continue
        det = FootballDetector(DetectionConfig(model_name=m_name, imgsz=1088, ball_conf=0.20, person_conf=0.40))
        pred_counts = {}
        t0 = time.perf_counter()
        for fno, ts, frm in nw_frames:
            raw = det.detect_frame(frm, fno, ts)
            players = [d for d in raw if d.class_name == "player"]
            balls = filter_ball_candidates(
                [d for d in raw if d.class_name == "ball"],
                players,
                frm.shape[:2],
                frame_bgr=frm,
            )
            pred_counts[fno] = {"people": len(players), "balls": len(balls)}
        ms = round((time.perf_counter() - t0) / max(1, len(nw_frames)) * 1000.0, 1)
        ev = evaluate_legacy_manual_counts(pred_counts, LEGACY_NORWAY_WALES_DETECTION_COUNTS)
        ev["inference_ms_per_frame"] = ms
        legacy_counts_eval[m_name] = ev

    payload = {
        "metrics_available_iou_ground_truth": False,
        "validation_blocker": (
            "0 manually verified ground-truth ball bounding boxes (`verified=true, source=manual`) "
            "exist in data/annotations/* yet. Strategy benchmarks report ball detection coverage and "
            "inference latency (strictly distinguished from IoU recall/AP@50), plus legacy 11-frame "
            "count-only reference on Norway vs Wales (iou_verified=False)."
        ),
        "yolo11s_status": yolo11s_status,
        "failure_case_filter_audit": {
            "total_cases": len(fp_rows),
            "false_positives_tested": sum(1 for r in fp_rows if not r["is_true_ball"]),
            "false_positives_rejected_by_static_filter": sum(
                1 for r in fp_rows if (not r["is_true_ball"]) and (not r["sprint3_1_filter_kept"])
            ),
            "true_balls_tested": sum(1 for r in fp_rows if r["is_true_ball"]),
            "true_balls_preserved": sum(
                1 for r in fp_rows if r["is_true_ball"] and r["sprint3_1_filter_kept"]
            ),
            "note": (
                "Case f=14 (conf=0.46 white sock 341px jump) is intentionally handled by Kalman motion "
                "gating rather than static box filtering so real high-confidence balls near players' feet "
                "are never suppressed."
            ),
        },
        "norway_wales_11_frames": {
            "yolo11n": nw_bench_n,
            "yolo11s": nw_bench_s,
            "legacy_manual_count_reference_not_iou_verified": legacy_counts_eval,
        },
        "chelsea_sunderland_13_tactical_frames": {
            "yolo11n": cs_bench_n,
            "yolo11s": cs_bench_s,
        },
    }
    (out_dir / "detection_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved Stage C detection metrics to:", out_dir / "detection_metrics.json")
    print("\n--- Chelsea vs Sunderland 13 wide tactical frames summary ---")
    for row in cs_bench_n + cs_bench_s:
        print(
            f"  {row['strategy']:42s} | cov={row['ball_detection_coverage_pct']:5.1f}% "
            f"| dets={row['total_ball_detections']:2d} | {row['inference_ms_per_frame']:6.1f} ms/f"
        )
    print("\n--- Norway vs Wales 11 legacy-counted frames summary ---")
    for m_name, ev in legacy_counts_eval.items():
        b = ev["classes"]["ball"]
        p = ev["classes"]["player"]
        print(
            f"  {m_name:10s} ({ev['inference_ms_per_frame']} ms/f) | "
            f"Ball(count-only): TP={b['tp']} FP={b['fp']} FN={b['fn']} P={b['precision']} R={b['recall']} F1={b['f1']} | "
            f"Player: P={p['precision']} R={p['recall']}"
        )


if __name__ == "__main__":
    main()
