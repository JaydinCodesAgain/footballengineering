"""End-to-end possession inference engine combining Kalman ball tracking, proximity ownership, and state aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.config import MatchConfig
from src.possession.ball_tracker import KalmanBallTracker, filter_ball_candidates
from src.possession.metrics import compute_possession_summary
from src.possession.ownership import evaluate_frame_ownership_candidates
from src.possession.state_machine import PossessionStateMachine
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    DetectionRecord,
    FrameObservationRecord,
    InternalLabel,
    PlayerProximityRecord,
    PossessionFrameRecord,
    PossessionState,
)


class PossessionEngine:
    """
    Coordinates per-frame ball tracking, player-ball ownership scoring, temporal state machine
    transitions, and cumulative possession share computation.
    """

    def __init__(
        self,
        match_config: MatchConfig,
        fps: float = 25.0,
        frame_stride: int = 2,
        total_video_duration_seconds: Optional[float] = None,
    ) -> None:
        self.match_config = match_config
        self.fps = max(1.0, float(fps))
        self.frame_stride = max(1, int(frame_stride))
        self.default_dt = float(self.frame_stride) / self.fps
        self.total_video_duration_seconds = total_video_duration_seconds

        self.ball_tracker = KalmanBallTracker(
            match_id=match_config.match_id,
            config=match_config.ball_tracking,
        )
        self.state_machine = PossessionStateMachine(config=match_config.possession)

        self.ball_trajectory: List[BallTrajectoryRecord] = []
        self.player_proximities: List[PlayerProximityRecord] = []
        self.timeline: List[PossessionFrameRecord] = []

        self._cum_team_a_sec: float = 0.0
        self._cum_team_b_sec: float = 0.0
        self._last_timestamp: Optional[float] = None

    def get_predicted_ball_roi(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Return the predicted ball search ROI `(x1, y1, x2, y2)` when ROI search is enabled."""
        if not self.match_config.detection.enable_roi_ball_search:
            return None
        return self.ball_tracker.get_predicted_search_roi(
            frame_shape, roi_size=self.match_config.detection.roi_size_px
        )

    def step_frame(
        self,
        frame_number: int,
        timestamp: float,
        player_observations: Sequence[FrameObservationRecord],
        raw_ball_detections: Sequence[DetectionRecord],
        player_detections: Optional[Sequence[DetectionRecord]] = None,
        frame_bgr: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        detection_source: str = "full_frame",
    ) -> Tuple[BallTrajectoryRecord, PossessionFrameRecord, List[PlayerProximityRecord]]:
        """
        Process a single video frame through the ball tracker, proximity evaluator, and possession
        state machine.
        """
        if frame_shape is None and frame_bgr is not None:
            frame_shape = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))

        p_dets = (
            list(player_detections)
            if player_detections is not None
            else [
                DetectionRecord(
                    frame_number=o.frame_number,
                    timestamp=o.timestamp,
                    class_id=0,
                    class_name="player",
                    bbox=o.player_bbox,
                    confidence=o.player_detection_confidence,
                    tracking_id=o.player_id,
                )
                for o in player_observations
            ]
        )

        if self.match_config.detection.filter_false_positive_balls and frame_shape is not None:
            filtered_balls = filter_ball_candidates(
                raw_ball_detections,
                p_dets,
                frame_shape=frame_shape,
                min_size=self.match_config.detection.min_ball_box_size_px,
                max_size=self.match_config.detection.max_ball_box_size_px,
                blacklisted_zones=self.ball_tracker.blacklisted_static_zones,
                frame_bgr=frame_bgr,
                predicted_pos=self.ball_tracker.get_predicted_position(),
                filter_crowd_horizon=self.match_config.ball_tracking.filter_crowd_horizon,
            )
        else:
            filtered_balls = list(raw_ball_detections)

        ball_rec = self.ball_tracker.step(
            frame_number=frame_number,
            timestamp=timestamp,
            ball_candidates=filtered_balls,
            frame_bgr=frame_bgr,
            detection_source=detection_source if filtered_balls else "none",
            player_detections=p_dets,
        )
        self.ball_trajectory.append(ball_rec)

        prev_pid = None if ball_rec.camera_cut_reset else self.state_machine.controlling_player_id
        prev_tid = None if ball_rec.camera_cut_reset else self.state_machine.controlling_team_id
        proximities, raw_interaction = evaluate_frame_ownership_candidates(
            frame_observations=player_observations,
            ball_record=ball_rec,
            config=self.match_config.possession,
            frame_shape=frame_shape,
            prev_controlling_player_id=prev_pid,
            prev_controlling_team_id=prev_tid,
        )
        self.player_proximities.extend(proximities)

        poss_state, ctrl_team, ctrl_pid, poss_conf = self.state_machine.step(
            raw_interaction, camera_cut=ball_rec.camera_cut_reset
        )

        if self._last_timestamp is not None and timestamp > self._last_timestamp:
            dt = float(timestamp - self._last_timestamp)
            # Clamp dt across large frame jumps so skipped segments don't inflate a single frame
            if dt > self.default_dt * 3.0:
                dt = self.default_dt
        else:
            dt = self.default_dt
        self._last_timestamp = float(timestamp)

        if poss_state == PossessionState.TEAM_A_CONTROL.value:
            self._cum_team_a_sec += dt
        elif poss_state == PossessionState.TEAM_B_CONTROL.value:
            self._cum_team_b_sec += dt

        tot_attr = self._cum_team_a_sec + self._cum_team_b_sec
        if tot_attr > 0:
            run_a_pct = round((self._cum_team_a_sec / tot_attr) * 100.0, 1)
            run_b_pct = round((self._cum_team_b_sec / tot_attr) * 100.0, 1)
        else:
            run_a_pct = 50.0
            run_b_pct = 50.0

        if ball_rec.observation_mode == BallObservationMode.OBSERVED.value:
            visibility = "visible"
        elif ball_rec.observation_mode == BallObservationMode.INTERPOLATED.value:
            visibility = "interpolated"
        else:
            visibility = "missing"

        if ctrl_team is not None:
            ctrl_display = self.match_config.to_display_label(ctrl_team)
        elif poss_state == PossessionState.CONTESTED.value:
            ctrl_display = "CONTESTED"
        elif poss_state == PossessionState.LOOSE_BALL.value:
            ctrl_display = "LOOSE BALL"
        elif poss_state == PossessionState.OUT_OF_PLAY.value:
            ctrl_display = "OUT OF PLAY"
        else:
            ctrl_display = "UNKNOWN"

        frame_rec = PossessionFrameRecord(
            match_id=self.match_config.match_id,
            frame_number=int(frame_number),
            timestamp=round(float(timestamp), 3),
            dt_seconds=round(dt, 4),
            ball_observation_mode=ball_rec.observation_mode,
            ball_visibility=visibility,
            possession_state=poss_state,
            raw_interaction_state=str(raw_interaction["raw_state"]),
            controlling_player_id=ctrl_pid,
            controlling_team_id=ctrl_team,
            controlling_team_display=ctrl_display,
            possession_confidence=poss_conf,
            closest_player_id=raw_interaction.get("closest_player_id"),
            closest_team_id=raw_interaction.get("closest_team_id"),
            closest_normalized_distance=raw_interaction.get("closest_norm_dist"),
            second_closest_team_id=raw_interaction.get("second_closest_team_id"),
            second_closest_normalized_distance=raw_interaction.get("second_closest_norm_dist"),
            running_team_a_share_pct=run_a_pct,
            running_team_b_share_pct=run_b_pct,
            camera_cut=ball_rec.camera_cut_reset,
        )
        self.timeline.append(frame_rec)
        return ball_rec, frame_rec, proximities

    def summarize(self) -> Dict[str, Any]:
        """Compute full possession summary and ball tracker summary."""
        poss_summary = compute_possession_summary(
            self.timeline,
            match_config=self.match_config,
            total_video_duration_seconds=self.total_video_duration_seconds,
            fps=self.fps,
            frame_stride=self.frame_stride,
        )
        poss_summary["ball_tracking"] = self.ball_tracker.summarize()
        return poss_summary

