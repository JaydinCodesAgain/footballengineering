"""Video frame annotation, live possession scoreboard HUD, detection diagnostic overlays, and MP4 encoding."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Sequence
import cv2
import numpy as np

from src.config import MatchConfig
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    DetectionRecord,
    FrameObservationRecord,
    InternalLabel,
    PossessionFrameRecord,
    PossessionState,
)


def _draw_possession_scoreboard(
    canvas: np.ndarray,
    possession_rec: PossessionFrameRecord,
    ball_rec: Optional[BallTrajectoryRecord],
    match_config: MatchConfig,
) -> None:
    """
    Draw a compact, high-contrast broadcast possession scoreboard in the top-right corner.
    """
    fh, fw = canvas.shape[:2]
    panel_w = 360
    panel_h = 168
    x0 = max(10, fw - panel_w - 14)
    y0 = 10
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    # Semi-transparent dark background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 100), 1)

    team_a_name = match_config.to_display_label(InternalLabel.TEAM_A.value).upper()
    team_b_name = match_config.to_display_label(InternalLabel.TEAM_B.value).upper()
    col_a = match_config.get_colour_bgr(InternalLabel.TEAM_A.value)
    col_b = match_config.get_colour_bgr(InternalLabel.TEAM_B.value)

    pct_a = float(possession_rec.running_team_a_share_pct)
    pct_b = float(possession_rec.running_team_b_share_pct)

    # Team headers & percentages
    cv2.putText(
        canvas,
        team_a_name[:12],
        (x0 + 14, y0 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        col_a,
        2,
    )
    cv2.putText(
        canvas,
        team_b_name[:12],
        (x1 - 14 - len(team_b_name[:12]) * 11, y0 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        col_b,
        2,
    )

    cv2.putText(
        canvas,
        f"{pct_a:4.1f}%",
        (x0 + 14, y0 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"{pct_b:4.1f}%",
        (x1 - 86, y0 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
    )

    # Horizontal possession split bar
    bar_x0 = x0 + 14
    bar_x1 = x1 - 14
    bar_y0 = y0 + 60
    bar_y1 = y0 + 68
    bar_w = bar_x1 - bar_x0
    split_x = bar_x0 + int(round(bar_w * (pct_a / 100.0)))
    cv2.rectangle(canvas, (bar_x0, bar_y0), (split_x, bar_y1), col_a, -1)
    cv2.rectangle(canvas, (split_x, bar_y0), (bar_x1, bar_y1), col_b, -1)

    # Status lines
    poss_label = possession_rec.controlling_team_display.upper()
    if possession_rec.controlling_team_id is not None:
        state_col = match_config.get_colour_bgr(possession_rec.controlling_team_id)
    elif possession_rec.possession_state == PossessionState.CONTESTED.value:
        state_col = (0, 215, 255)
    else:
        state_col = (200, 200, 200)

    mode = ball_rec.observation_mode if ball_rec else possession_rec.ball_observation_mode
    if mode == BallObservationMode.OBSERVED.value:
        ball_status_str = "TRACKED (OBSERVED)"
        ball_col = (80, 240, 80)
    elif mode == BallObservationMode.INTERPOLATED.value:
        ball_status_str = "COASTING (INTERP)"
        ball_col = (0, 190, 255)
    else:
        ball_status_str = "LOST"
        ball_col = (140, 140, 140)

    ctrl_str = (
        f"PLAYER #{possession_rec.controlling_player_id}"
        if possession_rec.controlling_player_id is not None
        and possession_rec.controlling_team_id is not None
        else "NONE"
    )

    cv2.putText(
        canvas,
        f"POSSESSION: {poss_label}",
        (x0 + 14, y0 + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        state_col,
        2,
    )
    cv2.putText(
        canvas,
        f"BALL STATUS: {ball_status_str}",
        (x0 + 14, y0 + 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        ball_col,
        1,
    )
    cv2.putText(
        canvas,
        f"CONTROL: {ctrl_str}",
        (x0 + 14, y0 + 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        canvas,
        f"CONFIDENCE: {possession_rec.possession_confidence:.2f}",
        (x0 + 14, y0 + 157),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
    )


def annotate_match_frame(
    frame_bgr: np.ndarray,
    player_observations: Sequence[FrameObservationRecord],
    ball_detections: Sequence[DetectionRecord],
    match_config: MatchConfig,
    ball_trajectory_rec: Optional[BallTrajectoryRecord] = None,
    possession_rec: Optional[PossessionFrameRecord] = None,
) -> np.ndarray:
    """
    Draw team-colored bounding boxes, track IDs, ball tracking state (observed vs interpolated),
    controlling player foot highlight, and live possession scoreboard HUD on a frame.
    """
    canvas = frame_bgr.copy()
    ctrl_pid = possession_rec.controlling_player_id if possession_rec else None
    ctrl_team = possession_rec.controlling_team_id if possession_rec else None

    # Draw player observations
    for obs in player_observations:
        x1, y1, x2, y2 = obs.player_bbox
        colour = match_config.get_colour_bgr(obs.team_id)
        disp_id = obs.player_id if obs.player_id is not None else "?"
        label = f"{obs.team_display_name} | ID: {disp_id}"

        thickness = 3 if (ctrl_pid is not None and obs.player_id == ctrl_pid and ctrl_team is not None) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
        )

        # Highlight controlling player's foot control zone
        if ctrl_pid is not None and obs.player_id == ctrl_pid and ctrl_team is not None:
            foot_cx = int((x1 + x2) / 2)
            foot_cy = int(y2 - 2)
            axis_w = max(14, int((x2 - x1) * 0.55))
            cv2.ellipse(canvas, (foot_cx, foot_cy), (axis_w, 7), 0, 0, 360, colour, 2)

    # Draw ball observation or Kalman interpolation
    if ball_trajectory_rec is not None:
        if (
            ball_trajectory_rec.observation_mode == BallObservationMode.OBSERVED.value
            and ball_trajectory_rec.bbox is not None
        ):
            bx1, by1, bx2, by2 = ball_trajectory_rec.bbox
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv2.putText(
                canvas,
                f"Ball ({ball_trajectory_rec.confidence:.2f})",
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )
        elif (
            ball_trajectory_rec.observation_mode == BallObservationMode.INTERPOLATED.value
            and ball_trajectory_rec.center_x is not None
            and ball_trajectory_rec.center_y is not None
        ):
            cx = int(round(ball_trajectory_rec.center_x))
            cy = int(round(ball_trajectory_rec.center_y))
            rad = max(8, min(28, int(round(ball_trajectory_rec.uncertainty_radius_px * 0.5))))
            cv2.circle(canvas, (cx, cy), rad, (0, 190, 255), 2)
            cv2.putText(
                canvas,
                "Ball (Interp)",
                (cx - 20, max(20, cy - rad - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 190, 255),
                2,
            )
    else:
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

    # Top-left detection counter HUD banner
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

    # Top-right possession scoreboard HUD when enabled
    if possession_rec is not None:
        _draw_possession_scoreboard(canvas, possession_rec, ball_trajectory_rec, match_config)

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
