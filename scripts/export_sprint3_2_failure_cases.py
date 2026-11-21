"""Generate visual before/after comparison panels and failure-case diagnostic images for Sprint 3.2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from scripts.evaluate_soccertrack_v2 import load_ball_trajectory, load_pipeline_observations
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter


def draw_run_overlay(
    frame_bgr: np.ndarray,
    obs_list,
    ball_rec,
    title: str,
    pitch_filter: Optional[PitchRegionFilter] = None,
) -> np.ndarray:
    vis = frame_bgr.copy()
    if pitch_filter is not None:
        cv2.polylines(vis, [np.round(pitch_filter.playable_polygon_px).astype(np.int32)], True, (0, 220, 0), 2)
        cv2.polylines(vis, [np.round(pitch_filter.player_margin_polygon_px).astype(np.int32)], True, (0, 200, 255), 1)
    colours = {
        "Team_A": (0, 70, 255),
        "Team_B": (255, 120, 0),
        "Referee": (0, 255, 255),
        "Unknown": (170, 170, 170),
    }
    for o in obs_list:
        x1, y1, x2, y2 = o.player_bbox
        c = colours.get(o.team_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            vis,
            f"#{o.player_id}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            c,
            2,
            cv2.LINE_AA,
        )
    if ball_rec is not None and ball_rec.bbox is not None:
        bx1, by1, bx2, by2 = ball_rec.bbox
        bc = (0, 255, 0) if ball_rec.is_observed else (0, 200, 255)
        cv2.rectangle(vis, (bx1 - 4, by1 - 4), (bx2 + 4, by2 + 4), bc, 3)
        cv2.putText(
            vis,
            f"BALL ({ball_rec.observation_mode}:{ball_rec.detection_source})",
            (max(10, bx1 - 30), max(24, by1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bc,
            2,
            cv2.LINE_AA,
        )

    fh, fw = vis.shape[:2]
    scale = 1800.0 / float(fw)
    out = cv2.resize(vis, (1800, int(round(fh * scale))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (1800, 38), (0, 0, 0), -1)
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_before_after_image(
    video_path: Path,
    base_dir: Path,
    new_dir: Path,
    raw_dir: Path,
    frame_number: int,
    match_label: str,
    out_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frm = cap.read()
    cap.release()
    if not ok or frm is None:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")

    prf = PitchRegionFilter.from_calibration_path(raw_dir, coordinate_space="raw_fisheye", config=PitchRegionConfig())
    obs_b, _ = load_pipeline_observations(base_dir / "frame_observations.csv")
    obs_n, _ = load_pipeline_observations(new_dir / "frame_observations.csv")
    ball_b = {int(r.frame_number): r for r in load_ball_trajectory(base_dir / "ball_tracking_diagnostics.csv")}
    ball_n = {int(r.frame_number): r for r in load_ball_trajectory(new_dir / "ball_tracking_diagnostics.csv")}

    p_b = obs_b.get(frame_number, [])
    p_n = obs_n.get(frame_number, [])
    br_b = ball_b.get(frame_number)
    br_n = ball_n.get(frame_number)

    top = draw_run_overlay(
        frm,
        p_b,
        br_b,
        f"BEFORE (Baseline full_frame yolo11n@1088) | {match_label} f={frame_number} | players={len(p_b)} | ball={br_b.observation_mode if br_b else 'none'}",
        pitch_filter=None,
    )
    bot = draw_run_overlay(
        frm,
        p_n,
        br_n,
        f"AFTER (Sprint 3.2 4-tile yolo11s@1280 + PitchFilter + FixedCam) | {match_label} f={frame_number} | players={len(p_n)} | ball={br_n.observation_mode if br_n else 'none'}",
        pitch_filter=prf,
    )
    stacked = np.vstack([top, bot])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), stacked, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Wrote before/after comparison: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sprint 3.2 before/after and failure-case images.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    make_before_after_image(args.video, args.base_dir, args.new_dir, args.raw_dir, args.frame, args.label, args.out)


if __name__ == "__main__":
    main()

