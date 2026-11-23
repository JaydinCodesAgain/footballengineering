"""Compare full-player masking against opt-in foot-zone motion recovery on a SoccerTrack MOT clip.

The score is deliberately named a *reconstructed-coordinate proxy*: the SoccerTrack ball
NPZ contains pitch-plane trajectory samples, not independently verified image-space boxes.
Use it to reject harmful recovery configurations, never as a published ball-detection claim.
Player boxes come from the released MOT ground truth solely to isolate the motion-recovery
ablation from person-detector variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.motion_ball import FixedCameraMotionBallDetector, MotionBallConfig
from src.schemas import DetectionRecord


def _load_mot(path: Path) -> Dict[int, List[Tuple[int, int, int, int]]]:
    rows: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) < 6:
                continue
            frame, _, x, y, width, height = row[:6]
            x1, y1 = int(round(float(x))), int(round(float(y)))
            rows[int(frame)].append((x1, y1, x1 + int(round(float(width))), y1 + int(round(float(height)))))
    return rows


def _load_proxy_points(ball_path: Path, homography_path: Path, clip_start_frame: int) -> Dict[int, Tuple[float, float]]:
    ball = np.load(ball_path, allow_pickle=False)
    homography = np.load(homography_path).astype(np.float64)
    result: Dict[int, Tuple[float, float]] = {}
    for source_frame, x, y in zip(ball["frame"], ball["x"], ball["y"]):
        clip_frame = int(source_frame) - clip_start_frame
        if clip_frame < 0:
            continue
        pitch = np.asarray([[[float(x) + 52.5, float(y) + 34.0]]], dtype=np.float32)
        uv = cv2.perspectiveTransform(pitch, homography)[0, 0]
        result[clip_frame] = (float(uv[0]), float(uv[1]))
    return result


def _padding_start_frame(padding_path: Path, fps: float) -> int:
    with padding_path.open("r", newline="", encoding="utf-8") as file:
        first_half = next(row for row in csv.DictReader(file) if row["Event Period"] == "FIRST_HALF")
    return int(round(float(first_half["Padding"]) / 1000.0 * fps))


def _evaluate_variant(
    clip_path: Path,
    mot_boxes: Dict[int, List[Tuple[int, int, int, int]]],
    proxy_points: Dict[int, Tuple[float, float]],
    frame_count: int,
    stride: int,
    preserve_player_foot_region: bool,
    tolerance_px: float,
) -> dict:
    detector = FixedCameraMotionBallDetector(
        MotionBallConfig(
            preserve_player_foot_region=preserve_player_foot_region,
            player_foot_region_height_fraction=0.18,
            require_kalman_corroboration=False,
            require_three_frame_or_kalman=True,
        )
    )
    capture = cv2.VideoCapture(str(clip_path))
    frames_with_proxy = 0
    proposal_frames = 0
    matched_frames = 0
    proposal_count = 0
    false_proxy_proposals = 0
    try:
        # Decode sequentially. Seeking every frame in a long-GOP MP4 is both much slower and
        # can produce inconsistent timing on OpenCV backends.
        for frame_number in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % stride:
                continue
            players = [
                DetectionRecord(frame_number, frame_number / 25.0, 0, "player", box, 1.0)
                for box in mot_boxes.get(frame_number, [])
            ]
            proposals = detector.detect_candidates(frame, frame_number, frame_number / 25.0, players)
            proxy = proxy_points.get(frame_number)
            if proxy is None:
                continue
            if not (0 <= proxy[0] < frame.shape[1] and 0 <= proxy[1] < frame.shape[0]):
                continue
            frames_with_proxy += 1
            distances = [
                math.hypot((p.bbox[0] + p.bbox[2]) / 2.0 - proxy[0], (p.bbox[1] + p.bbox[3]) / 2.0 - proxy[1])
                for p in proposals
            ]
            if proposals:
                proposal_frames += 1
                proposal_count += len(proposals)
            if distances and min(distances) <= tolerance_px:
                matched_frames += 1
            false_proxy_proposals += sum(distance > tolerance_px for distance in distances)
    finally:
        capture.release()
    return {
        "frames_with_in_bounds_proxy": frames_with_proxy,
        "frames_with_motion_proposal": proposal_frames,
        "matched_proxy_frames": matched_frames,
        "motion_proposal_count": proposal_count,
        "unmatched_proxy_proposals": false_proxy_proposals,
        "proxy_recall": round(matched_frames / max(1, frames_with_proxy), 6),
        "proposals_per_frame": round(proposal_count / max(1, frames_with_proxy), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--match-id", default="132831")
    parser.add_argument("--frames", type=int, default=150, help="Number of dense clip frames to inspect.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tolerance-px", type=float, default=35.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/sprint3_4/foot_zone_motion_proxy.json"))
    args = parser.parse_args()
    root = args.dataset_root
    mid = args.match_id
    clip = root / "mot" / "clips" / f"{mid}.mp4"
    mot = root / "mot" / f"{mid}.txt"
    ball = root / "ball" / f"{mid}_1st_ball.npz"
    raw = root / "raw" / mid
    homography = raw / f"{mid}_homography.npy"
    padding = raw / f"{mid}_padding_info.csv"
    required = (clip, mot, ball, homography, padding)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Required clip, MOT, ball, homography, or padding asset is missing.")
    probe = cv2.VideoCapture(str(clip))
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    available_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    clip_start = _padding_start_frame(padding, fps)
    proxy = _load_proxy_points(ball, homography, clip_start)
    mot_boxes = _load_mot(mot)
    frames = min(max(3, int(args.frames)), available_frames)
    report = {
        "match_id": mid,
        "protocol": {
            "label_type": "reconstructed_ball_coordinate_proxy_not_independent_ground_truth",
            "player_boxes": "released_MOT_ground_truth_to_isolate_motion_ablation",
            "clip_start_source_frame": clip_start,
            "evaluated_clip_frames": frames,
            "stride": int(args.stride),
            "tolerance_px": float(args.tolerance_px),
        },
        "full_player_mask_baseline": _evaluate_variant(
            clip, mot_boxes, proxy, frames, max(1, args.stride), False, args.tolerance_px
        ),
        "foot_zone_motion_recovery": _evaluate_variant(
            clip, mot_boxes, proxy, frames, max(1, args.stride), True, args.tolerance_px
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
