"""Video frame annotation, detection diagnostic overlays, and MP4 encoding."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Sequence
import cv2
import numpy as np

from src.config import MatchConfig
from src.schemas import DetectionRecord, FrameObservationRecord


def annotate_match_frame(
    frame_bgr: np.ndarray,
    player_observations: Sequence[FrameObservationRecord],
    ball_detections: Sequence[DetectionRecord],
    match_config: MatchConfig,
) -> np.ndarray:
    """
    Draw team-colored bounding boxes, track IDs, ball detections, and a summary HUD on a frame.
    """
    canvas = frame_bgr.copy()

    # Draw football detections
    for ball in ball_detections:
        x1, y1, x2, y2 = ball.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            canvas,
            f"Ball ({ball.confidence:.2f})",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # Draw player observations
    for obs in player_observations:
        x1, y1, x2, y2 = obs.player_bbox
        colour = match_config.get_colour_bgr(obs.team_id)
        disp_id = obs.player_id if obs.player_id is not None else "?"
        label = f"{obs.team_display_name} | ID: {disp_id}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
        )

    # Top-left HUD banner
    cv2.rectangle(canvas, (8, 8), (410, 58), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"People: {len(player_observations)} | Balls: {len(ball_detections)}",
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return canvas


def render_detection_diagnostic_frame(
    frame_bgr: np.ndarray,
    class_diagnostic: Dict[str, Any],
    title_prefix: str = "Player",
) -> np.ndarray:
    """
    Render diagnostic overlay for a single frame:
    - Green: True Positives (matched prediction)
    - Red: False Positives (unmatched prediction)
    - Yellow: False Negatives (missed ground truth)
    """
    canvas = frame_bgr.copy()

    for m in class_diagnostic.get("matched_pairs", []):
        x1, y1, x2, y2 = [int(v) for v in m["pred_bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            canvas,
            f"TP IoU={m['iou']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 0),
            2,
        )

    for fp in class_diagnostic.get("false_positive_boxes", []):
        x1, y1, x2, y2 = [int(v) for v in fp["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"FP ({fp['confidence']:.2f})",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            2,
        )

    for fn_box in class_diagnostic.get("missed_ground_truth_boxes", []):
        x1, y1, x2, y2 = [int(v) for v in fn_box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            canvas,
            "FN (Missed GT)",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
        )

    tp = class_diagnostic.get("tp", 0)
    fp = class_diagnostic.get("fp", 0)
    fn = class_diagnostic.get("fn", 0)
    fno = class_diagnostic.get("frame_number", 0)

    cv2.rectangle(canvas, (8, 8), (520, 52), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"Frame {fno} [{title_prefix}] TP:{tp} FP:{fp} FN:{fn}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    return canvas


def finalize_mp4_video(raw_mp4_path: str | Path, final_mp4_path: str | Path) -> Path:
    """
    Convert an OpenCV `mp4v` video to browser-compatible H.264 using `ffmpeg` if available,
    or fall back cleanly to copying the `mp4v` file so execution never crashes on machines
    without an external `ffmpeg` binary.
    """
    raw_p = Path(raw_mp4_path)
    final_p = Path(final_mp4_path)
    final_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_p),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    str(final_p),
                ],
                check=True,
            )
            if raw_p != final_p and raw_p.is_file():
                raw_p.unlink()
            return final_p
        except Exception:
            pass

    if raw_p != final_p:
        shutil.copy2(raw_p, final_p)
        if raw_p.is_file():
            raw_p.unlink()
    return final_p

