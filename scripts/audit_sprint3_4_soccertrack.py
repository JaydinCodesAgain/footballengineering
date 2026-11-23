"""Audit locally downloaded SoccerTrack v2 assets before using them for Sprint 3.4.

This script intentionally distinguishes the release's dense ``ball/*.npz`` trajectories
from independently reviewed image-space football boxes.  It can validate coordinate and
clip alignment, but it must *not* be used to claim detector precision/recall or train a
football detector without a separate visual annotation review.

Example (after downloading the small MOT clip and metadata):
    python scripts/audit_sprint3_4_soccertrack.py --dataset-root data/soccertrack_v2_external/132831 --match-id 132831
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_first_half_paths(root: Path, match_id: str) -> dict[str, Path]:
    paths = {
        "clip": root / "mot" / "clips" / f"{match_id}.mp4",
        "ball": root / "ball" / f"{match_id}_1st_ball.npz",
        "homography": root / "raw" / match_id / f"{match_id}_homography.npy",
        "padding": root / "raw" / match_id / f"{match_id}_padding_info.csv",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required SoccerTrack assets: {detail}")
    return paths


def _load_first_half_padding_frame(padding_csv: Path, fps: float) -> int:
    with padding_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    first_half = next((row for row in rows if row.get("Event Period") == "FIRST_HALF"), None)
    if first_half is None:
        raise ValueError(f"No FIRST_HALF row in {padding_csv}")
    return int(round(float(first_half["Padding"]) / 1000.0 * fps))


def audit_dataset(dataset_root: Path, match_id: str) -> dict[str, Any]:
    paths = _find_first_half_paths(dataset_root, match_id)
    capture = cv2.VideoCapture(str(paths["clip"]))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open clip {paths['clip']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("Clip metadata is invalid; cannot establish frame alignment.")

    ball = np.load(paths["ball"], allow_pickle=False)
    required = {"frame", "x", "y", "status"}
    missing = required.difference(ball.files)
    if missing:
        raise ValueError(f"Ball NPZ is missing arrays: {sorted(missing)}")
    frames = ball["frame"].astype(np.int64)
    xs = ball["x"].astype(np.float64)
    ys = ball["y"].astype(np.float64)
    statuses = ball["status"].astype(np.int64)
    if not (len(frames) == len(xs) == len(ys) == len(statuses)):
        raise ValueError("Ball NPZ arrays have inconsistent lengths.")

    clip_start_frame = _load_first_half_padding_frame(paths["padding"], fps)
    clip_end_frame_exclusive = clip_start_frame + frame_count
    in_clip = (frames >= clip_start_frame) & (frames < clip_end_frame_exclusive)
    if not np.any(in_clip):
        raise ValueError(
            "No ball trajectory samples overlap the MOT clip after applying the documented padding offset."
        )

    # SoccerTrack raw ball coordinates are centred on the pitch.  The released homography is
    # defined on the project convention [0..105] x [0..68], hence the explicit translation.
    pitch_points = np.column_stack([xs[in_clip] + 52.5, ys[in_clip] + 34.0]).astype(np.float32)
    homography = np.load(paths["homography"]).astype(np.float64)
    projected = cv2.perspectiveTransform(pitch_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    within_frame = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )

    status_counts = {str(key): int(value) for key, value in sorted(Counter(statuses.tolist()).items())}
    report = {
        "match_id": match_id,
        "asset_type": "SoccerTrack_v2_MOT_clip_plus_reconstructed_ball_trajectory",
        "clip": {
            "path": str(paths["clip"]),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(frame_count / fps, 3),
            "first_half_source_frame_start": clip_start_frame,
            "first_half_source_frame_end_exclusive": clip_end_frame_exclusive,
        },
        "ball_npz": {
            "path": str(paths["ball"]),
            "arrays": sorted(ball.files),
            "frame_count": int(len(frames)),
            "frame_min": int(frames.min()),
            "frame_max": int(frames.max()),
            "status_counts": status_counts,
            "clip_overlap_samples": int(np.count_nonzero(in_clip)),
            "pitch_coordinate_convention": "source coordinates centred at (0, 0); translated by (+52.5, +34.0) before projection",
            "projected_points_inside_clip_fraction": round(float(np.mean(within_frame)), 6),
        },
        "fitness": {
            "mot_tracking_evaluation": "eligible: clip has per-frame player boxes; keep its 22-player scope explicit",
            "ball_detection_evaluation": "proxy-only: not eligible for independent precision/recall because NPZ positions are released trajectory coordinates, not visually verified image boxes",
            "ball_detector_training": "blocked until ball boxes are independently visually verified and split match-disjointly",
            "final_untouched_evaluation": "eligible only if this match remains excluded from model selection, threshold tuning, and visual review",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sprint3_4/soccertrack_asset_audit.json"))
    args = parser.parse_args()
    result = audit_dataset(args.dataset_root, args.match_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
