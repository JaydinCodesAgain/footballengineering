"""Sprint 3.2 Priority 1 Benchmark: Full-Frame vs Native-Resolution Multi-Tile Detection.

Evaluates full-frame and overlapping multi-tile configurations on identical frames of
SoccerTrack v2 development matches (117092 night 3840x1906 + 118575 daytime 4096x1080;
NEVER touches held-out test matches) measuring:
- Ground-truth Player Precision, Recall, F1, AP (pitch space <= 5.0m & <= 8.0m)
- Duplicate-detection rate near tile seams
- Verified Ball Recall / Precision on 117092's 10 visually verified ball frames
- Inference latency (ms/frame) and peak memory delta (tracemalloc MB)
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scripts.evaluate_soccertrack_v2 import (
    load_fisheye_pitch_projector,
    load_gsr_frames_streaming,
    match_pitch_hungarian,
)
from src.config import DetectionConfig
from src.detection.detector import FootballDetector
from src.detection.tiling import estimate_active_content_bbox, generate_tile_grid
from src.schemas import DetectionRecord


def compute_ap_from_matches(scored_flags: List[Tuple[float, bool]], total_gt: int) -> float:
    """Compute VOC/COCO-style all-point interpolated Average Precision."""
    if total_gt <= 0 or not scored_flags:
        return 0.0
    ordered = sorted(scored_flags, key=lambda x: x[0], reverse=True)
    tp = np.cumsum([1.0 if ok else 0.0 for _, ok in ordered])
    fp = np.cumsum([0.0 if ok else 1.0 for _, ok in ordered])
    rec = tp / float(total_gt)
    prec = tp / np.maximum(1.0, tp + fp)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_player_detections_pitch(
    preds_by_frame: Dict[int, List[DetectionRecord]],
    gsr_by_frame: Dict[int, List[Dict[str, Any]]],
    projector: Dict[str, Any],
    match_radius_m: float = 5.0,
) -> Dict[str, Any]:
    p2p = projector["pixel_to_pitch"]
    total_gt = total_tp = total_fp_all = total_fp_on = total_fn = 0
    scored_all: List[Tuple[float, bool]] = []
    scored_on: List[Tuple[float, bool]] = []
    duplicate_near_pairs = 0
    dists: List[float] = []

    for fno in sorted(preds_by_frame.keys()):
        gt_list = [g for g in gsr_by_frame.get(fno + 1, []) if g["pitch_x"] is not None]
        if not gt_list:
            continue
        players = [d for d in preds_by_frame[fno] if d.class_name == "player"]
        total_gt += len(gt_list)

        # Count close duplicate predicted pairs (centre dist < 18px & IoU > 0.25)
        for i in range(len(players)):
            ax1, ay1, ax2, ay2 = players[i].bbox
            acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
            for j in range(i + 1, len(players)):
                bx1, by1, bx2, by2 = players[j].bbox
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if math.hypot(acx - bcx, acy - bcy) < 18.0:
                    duplicate_near_pairs += 1

        if not players:
            total_fn += len(gt_list)
            continue

        gt_xy = np.array([[float(g["pitch_x"]), float(g["pitch_y"])] for g in gt_list], dtype=np.float64)
        feet_uv = np.array([[(p.bbox[0] + p.bbox[2]) / 2.0, float(p.bbox[3])] for p in players], dtype=np.float64)
        pr_xy = p2p(feet_uv)
        on_pitch = (np.abs(pr_xy[:, 0]) <= 55.0) & (np.abs(pr_xy[:, 1]) <= 36.5)

        matches = match_pitch_hungarian(gt_xy, pr_xy, max_dist_m=match_radius_m)
        matched_pi = {pi for _, pi, _ in matches}
        dists.extend(d for _, _, d in matches)

        total_tp += len(matches)
        total_fn += len(gt_list) - len(matches)
        total_fp_all += len(players) - len(matches)
        total_fp_on += sum(1 for pi in range(len(players)) if on_pitch[pi] and pi not in matched_pi)

        for pi, p in enumerate(players):
            is_tp = pi in matched_pi
            scored_all.append((float(p.confidence), is_tp))
            if on_pitch[pi] or is_tp:
                scored_on.append((float(p.confidence), is_tp))

    prec_all = total_tp / float(max(1, total_tp + total_fp_all))
    prec_on = total_tp / float(max(1, total_tp + total_fp_on))
    rec = total_tp / float(max(1, total_gt))
    f1_all = (2.0 * prec_all * rec / (prec_all + rec)) if (prec_all + rec) > 0 else 0.0
    f1_on = (2.0 * prec_on * rec / (prec_on + rec)) if (prec_on + rec) > 0 else 0.0

    return {
        "match_radius_m": match_radius_m,
        "total_gt": total_gt,
        "tp": total_tp,
        "fp_all": total_fp_all,
        "fp_on_pitch": total_fp_on,
        "fn": total_fn,
        "precision_all": round(prec_all, 4),
        "precision_on_pitch": round(prec_on, 4),
        "recall": round(rec, 4),
        "f1_all": round(f1_all, 4),
        "f1_on_pitch": round(f1_on, 4),
        "ap_all": round(compute_ap_from_matches(scored_all, total_gt), 4),
        "ap_on_pitch": round(compute_ap_from_matches(scored_on, total_gt), 4),
        "median_dist_m": round(float(np.median(dists)), 2) if dists else None,
        "close_duplicate_pairs_lt18px": duplicate_near_pairs,
    }


def evaluate_ball_detections_sample(
    preds_by_frame: Dict[int, List[DetectionRecord]],
    gt_ball_boxes: Dict[int, Optional[List[int]]],
    match_dist_px: float = 25.0,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    scored: List[Tuple[float, bool]] = []
    n_vis = 0
    for fno, gt_box in sorted(gt_ball_boxes.items()):
        if fno not in preds_by_frame:
            continue
        balls = [d for d in preds_by_frame[fno] if d.class_name == "ball"]
        if gt_box is None:
            for b in balls:
                fp += 1
                scored.append((float(b.confidence), False))
            continue
        n_vis += 1
        gcx, gcy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
        matched = False
        for b in sorted(balls, key=lambda x: x.confidence, reverse=True):
            bcx, bcy = (b.bbox[0] + b.bbox[2]) / 2.0, (b.bbox[1] + b.bbox[3]) / 2.0
            if not matched and math.hypot(bcx - gcx, bcy - gcy) <= match_dist_px:
                matched = True
                tp += 1
                scored.append((float(b.confidence), True))
            else:
                fp += 1
                scored.append((float(b.confidence), False))
        if not matched:
            fn += 1
    prec = tp / float(tp + fp) if (tp + fp) > 0 else None
    rec = tp / float(max(1, n_vis))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec is not None and (prec + rec) > 0) else 0.0
    return {
        "gt_visible": n_vis,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "ap": round(compute_ap_from_matches(scored, n_vis), 4),
    }


def load_frames(video_path: Path, frame_numbers: Sequence[int]) -> List[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    out = []
    for fno in sorted(set(frame_numbers)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
        ok, frm = cap.read()
        if ok and frm is not None:
            out.append((int(fno), round(fno / fps, 3), frm))
    cap.release()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3.2 Multi-Tile Detection Benchmark on Dev Matches")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/sprint3_2/tiling_benchmark.json"))
    args = parser.parse_args()

    ball_sample = json.loads(
        Path("outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.json").read_text(encoding="utf-8")
    )
    gt_ball_117092 = {
        int(x["frame_number"]): x["bbox"] for x in ball_sample["matches"]["117092"]["per_frame"]
    }

    dev_datasets = [
        {
            "match_id": "117092",
            "condition": "night_floodlit_3840x1906",
            "video": Path("SoccerTrack-v2/data/videos/117092/117092_panorama_1st_half.mp4"),
            "gsr": Path("SoccerTrack-v2/data/gsr/117092/117092_1st.json"),
            "raw_dir": Path("SoccerTrack-v2/data/raw/117092"),
            # 12 frames including all 10 verified ball frames + 2 extra frames
            "frames": [20, 32, 40, 50, 60, 70, 86, 136, 204, 240, 258, 320],
            "ball_gt": gt_ball_117092,
        },
        {
            "match_id": "118575",
            "condition": "daylight_4096x1080",
            "video": Path("SoccerTrack-v2/data/videos/118575/118575_panorama_1st_half.mp4"),
            "gsr": Path("SoccerTrack-v2/data/gsr/118575/118575_1st.json"),
            "raw_dir": Path("SoccerTrack-v2/data/raw/118575"),
            "frames": [20, 50, 80, 110, 140, 170, 200, 230, 260, 290, 320, 350],
            "ball_gt": None,
        },
    ]

    configs: List[Tuple[str, DetectionConfig]] = [
        (
            "baseline_full_yolo11n_1088_p040",
            DetectionConfig(model_name="yolo11n.pt", imgsz=1088, person_conf=0.40, ball_conf=0.20, inference_mode="full_frame"),
        ),
        (
            "full_yolo11n_1280_p025",
            DetectionConfig(model_name="yolo11n.pt", imgsz=1280, person_conf=0.25, ball_conf=0.14, inference_mode="full_frame"),
        ),
        (
            "full_yolo11s_1280_p025",
            DetectionConfig(model_name="yolo11s.pt", imgsz=1280, person_conf=0.25, ball_conf=0.14, inference_mode="full_frame"),
        ),
        (
            "tiled_yolo11n_1600x1080_img960_p025",
            DetectionConfig(
                model_name="yolo11n.pt", person_conf=0.25, ball_conf=0.14,
                inference_mode="tiled", tile_width=1600, tile_height=1080, tile_overlap=0.20, tile_imgsz=960,
            ),
        ),
        (
            "tiled_yolo11n_1280x1080_img1280_p025",
            DetectionConfig(
                model_name="yolo11n.pt", person_conf=0.25, ball_conf=0.14,
                inference_mode="tiled", tile_width=1280, tile_height=1080, tile_overlap=0.22, tile_imgsz=1280,
            ),
        ),
        (
            "tiled_yolo11n_1024x960_img1024_p025",
            DetectionConfig(
                model_name="yolo11n.pt", person_conf=0.25, ball_conf=0.14,
                inference_mode="tiled", tile_width=1024, tile_height=960, tile_overlap=0.22, tile_imgsz=1024,
            ),
        ),
        (
            "tiled_yolo11s_1280x1080_img1280_p025",
            DetectionConfig(
                model_name="yolo11s.pt", person_conf=0.25, ball_conf=0.14,
                inference_mode="tiled", tile_width=1280, tile_height=1080, tile_overlap=0.22, tile_imgsz=1280,
            ),
        ),
        (
            "tiled_yolo11s_1024x960_img1024_p025",
            DetectionConfig(
                model_name="yolo11s.pt", person_conf=0.25, ball_conf=0.14,
                inference_mode="tiled", tile_width=1024, tile_height=960, tile_overlap=0.22, tile_imgsz=1024,
            ),
        ),
    ]

    # Preload models once per weight file so model-load time doesn't pollute per-frame latency
    all_results: Dict[str, Any] = {"matches": {}}

    for ds in dev_datasets:
        mid = ds["match_id"]
        print(f"\n=== Benchmarking Dev Match {mid} ({ds['condition']}) ===")
        frames = load_frames(ds["video"], ds["frames"])
        gsr, _ = load_gsr_frames_streaming(ds["gsr"], {f + 1 for f in ds["frames"]}, stop_after_frame=max(ds["frames"]) + 5)
        proj = load_fisheye_pitch_projector(ds["raw_dir"], mid)
        fh, fw = frames[0][2].shape[:2]
        content_roi = estimate_active_content_bbox(frames[0][2])
        print(f"Loaded {len(frames)} frames ({fw}x{fh}), auto content ROI={content_roi}, cal_err={proj['calibration_keypoint_error']}")

        match_rows = []
        for name, cfg in configs:
            det = FootballDetector(config=cfg)
            # Warmup 1 frame outside timing
            _ = det.detect_frame_auto(frames[0][2], frames[0][0], frames[0][1])

            n_tiles = (
                len(generate_tile_grid((fh, fw), cfg.tile_width, cfg.tile_height, cfg.tile_overlap, roi_bbox=content_roi))
                if cfg.inference_mode == "tiled"
                else 1
            )

            tracemalloc.start()
            t0 = time.perf_counter()
            preds_by_frame: Dict[int, List[DetectionRecord]] = {}
            total_p = total_b = 0
            for fno, ts, frm in frames:
                recs = det.detect_frame_auto(frm, fno, ts)
                preds_by_frame[fno] = recs
                total_p += sum(1 for r in recs if r.class_name == "player")
                total_b += sum(1 for r in recs if r.class_name == "ball")
            wall_s = time.perf_counter() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            p5 = evaluate_player_detections_pitch(preds_by_frame, gsr, proj, match_radius_m=5.0)
            p8 = evaluate_player_detections_pitch(preds_by_frame, gsr, proj, match_radius_m=8.0) if mid == "117092" else None
            b_eval = evaluate_ball_detections_sample(preds_by_frame, ds["ball_gt"]) if ds["ball_gt"] else None

            row = {
                "config_name": name,
                "model": cfg.model_name,
                "mode": cfg.inference_mode,
                "tiles_per_frame": n_tiles,
                "tile_spec": f"{cfg.tile_width}x{cfg.tile_height}@{cfg.tile_imgsz}" if cfg.inference_mode == "tiled" else f"full@{cfg.imgsz}",
                "avg_players_detected": round(total_p / float(len(frames)), 2),
                "avg_balls_detected": round(total_b / float(len(frames)), 2),
                "ms_per_frame": round((wall_s / float(len(frames))) * 1000.0, 1),
                "peak_python_mem_mb": round(peak_bytes / (1024.0 * 1024.0), 2),
                "player_pitch_5m": p5,
                "player_pitch_8m": p8,
                "ball_verified_sample": b_eval,
            }
            match_rows.append(row)
            b_str = f" | ball_R={b_eval['recall']:.0%} P={b_eval['precision']} TP={b_eval['tp']} FP={b_eval['fp']}" if b_eval else f" | raw_balls/f={row['avg_balls_detected']:.2f}"
            print(
                f"  {name:<38} tiles={n_tiles} | {row['ms_per_frame']:6.0f}ms/f | "
                f"P5m={p5['precision_on_pitch']:.3f} R5m={p5['recall']:.3f} F1={p5['f1_on_pitch']:.3f} AP={p5['ap_on_pitch']:.3f} dup={p5['close_duplicate_pairs_lt18px']}{b_str}"
            )

        all_results["matches"][mid] = {
            "condition": ds["condition"],
            "resolution": [fw, fh],
            "auto_content_roi": list(content_roi),
            "frames_evaluated": ds["frames"],
            "calibration_error": proj["calibration_keypoint_error"],
            "benchmarks": match_rows,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved tiling benchmark to {args.out_json}")


if __name__ == "__main__":
    main()

