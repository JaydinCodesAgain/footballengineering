"""Persistent football tracking using a 2D constant-velocity Kalman filter with static-clutter suppression, outlier rejection, and camera-cut resets."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.config import BallTrackingConfig
from src.schemas import BallObservationMode, BallTrajectoryRecord, DetectionRecord


def detect_camera_cut(
    prev_frame_bgr: Optional[np.ndarray],
    curr_frame_bgr: Optional[np.ndarray],
    corr_threshold: float = 0.58,
    l1_diff_threshold: float = 30.0,
    prev_median_player_h: Optional[float] = None,
    curr_median_player_h: Optional[float] = None,
) -> bool:
    """
    Detect a broadcast camera cut, replay wipe, or wide-to-closeup focal-length switch using:
    1. Global HSV colour histogram correlation (`corr < corr_threshold`),
    2. Downsampled spatial grayscale L1 difference (`mean_l1 > l1_diff_threshold`),
    3. Abrupt player bounding-box scale jump (`ratio > 2.2` or `< 0.45`) when cutting between
       a wide tactical camera and a player close-up on the same pitch.
    """
    if (
        prev_median_player_h is not None
        and curr_median_player_h is not None
        and prev_median_player_h > 20.0
        and curr_median_player_h > 20.0
    ):
        scale_ratio = curr_median_player_h / prev_median_player_h
        if scale_ratio > 2.25 or scale_ratio < 0.44:
            return True

    if prev_frame_bgr is None or curr_frame_bgr is None:
        return False
    if prev_frame_bgr.size == 0 or curr_frame_bgr.size == 0:
        return False

    small_prev = cv2.resize(prev_frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)

    hsv_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2HSV)
    hsv_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2HSV)

    hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [18, 8], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [18, 8], [0, 180, 0, 256])

    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)

    corr = float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL))
    if corr < corr_threshold:
        return True

    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l1_diff = float(np.mean(np.abs(gray_curr - gray_prev)))
    return bool(corr < 0.72 and l1_diff > max(38.0, l1_diff_threshold))


def _compute_median_player_height(player_detections: Sequence[DetectionRecord]) -> Optional[float]:
    heights = [float(p.bbox[3] - p.bbox[1]) for p in player_detections if (p.bbox[3] - p.bbox[1]) > 15]
    if not heights:
        return None
    return float(np.median(heights))


def filter_ball_candidates(
    ball_detections: Sequence[DetectionRecord],
    player_detections: Sequence[DetectionRecord],
    frame_shape: Tuple[int, int],
    min_size: int = 5,
    max_size: int = 140,
    blacklisted_zones: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> List[DetectionRecord]:
    """
    Filter false-positive ball detections caused by:
    - Broadcast scoreboards / banners (`cy < 0.06 * fh`),
    - Extreme aspect ratios (`< 0.42` or `> 2.35`),
    - Boxes disproportionate to player scale (`size > 0.38 * median_player_height`, e.g. white shorts),
    - Low-confidence triggers on white jerseys/badges inside upper player torsos,
    - Previously identified static background clutter zones (`blacklisted_zones`).
    """
    if not ball_detections:
        return []

    fh, fw = frame_shape[:2]
    med_ph = _compute_median_player_height(player_detections)
    dynamic_max_size = max_size
    if med_ph is not None and med_ph > 25.0:
        # A football diameter (~22cm) is ~12%-25% of player height; allow up to 38% for motion blur
        dynamic_max_size = min(max_size, max(26, int(round(med_ph * 0.38))))

    filtered: List[DetectionRecord] = []

    for b in ball_detections:
        x1, y1, x2, y2 = b.bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_size or bh < min_size or bw > dynamic_max_size or bh > dynamic_max_size:
            continue

        aspect = bw / float(max(1, bh))
        if aspect < 0.42 or aspect > 2.35:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Exclude static top broadcast scoreboard overlays
        if cy < fh * 0.06 and b.confidence < 0.55:
            continue

        # Exclude known static background clutter hot-spots
        if blacklisted_zones:
            if any(math.hypot(cx - zx, cy - zy) <= zr for zx, zy, zr in blacklisted_zones):
                continue

        # Check if low-confidence ball sits inside a player's upper chest/head (white badge/collar)
        is_torso_logo_fp = False
        if b.confidence < 0.26:
            for p in player_detections:
                px1, py1, px2, py2 = p.bbox
                ph = py2 - py1
                if px1 <= cx <= px2 and (py1 + 0.10 * ph) <= cy <= (py1 + 0.62 * ph):
                    is_torso_logo_fp = True
                    break
        if is_torso_logo_fp:
            continue

        filtered.append(b)

    return filtered


class KalmanBallTracker:
    """
    Dedicated single-ball tracker using a 4D constant-velocity Kalman filter `[x, y, vx, vy]`.

    Reliability features:
    - Multi-candidate gating and player-proximity-aware confidence scoring.
    - Static background clutter detection & blacklisting: if a tracked target remains stationary
      (`< 3.5 px` movement) for `>= 4` consecutive observations while far from all players
      (`> 2.2 * median_player_height`), it is flagged as a `false_track_association`, blacklisted,
      and reset so the tracker does not stay locked onto a touchline cone, spare ball, or stadium logo.
    - Outlier jump rejection with multi-frame re-acquisition confirmation for fast long passes.
    - Short-gap coasting interpolation (`observation_mode='interpolated'`) strictly separated
      from genuine detector observations (`observation_mode='observed'`).
    - Automatic reset across broadcast camera cuts (`camera_cut_reset=True`).
    """

    def __init__(
        self,
        match_id: str = "match",
        config: Optional[BallTrackingConfig] = None,
    ) -> None:
        self.match_id = match_id
        self.config = config or BallTrackingConfig()
        self.blacklisted_static_zones: List[Tuple[float, float, float]] = []
        self.false_track_associations: int = 0
        self.reacquisition_events: int = 0
        self.history: List[BallTrajectoryRecord] = []
        self._prev_frame_small: Optional[np.ndarray] = None
        self._prev_median_player_h: Optional[float] = None
        self._reset_kalman_state()

    def _reset_kalman_state(self) -> None:
        self.is_initialized: bool = False
        self.state: np.ndarray = np.zeros((4, 1), dtype=np.float64)
        self.cov: np.ndarray = np.eye(4, dtype=np.float64) * 500.0
        self.last_frame_number: Optional[int] = None
        self.last_timestamp: Optional[float] = None
        self.last_observed_bbox_size: Tuple[int, int] = (16, 16)
        self.last_confidence: float = 0.0
        self.frames_since_observed: int = 0
        self._reacquire_candidate_pos: Optional[Tuple[float, float]] = None
        self._reacquire_streak: int = 0
        self._static_anchor_pos: Optional[Tuple[float, float]] = None
        self._static_isolated_streak: int = 0

    def reset(self) -> None:
        """Full reset of tracker state and history."""
        self.blacklisted_static_zones.clear()
        self.false_track_associations = 0
        self.reacquisition_events = 0
        self.history.clear()
        self._prev_frame_small = None
        self._prev_median_player_h = None
        self._reset_kalman_state()

    @property
    def is_tracking_active(self) -> bool:
        return self.is_initialized and self.frames_since_observed <= self.config.max_coast_frames

    def get_predicted_search_roi(
        self,
        frame_shape: Tuple[int, int],
        roi_size: int = 640,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Return a square `(x1, y1, x2, y2)` region of interest around the current predicted ball
        position for high-resolution local ROI detection when full-frame detection misses the ball.
        """
        if not self.is_tracking_active:
            return None

        fh, fw = frame_shape[:2]
        pred_x = float(self.state[0, 0])
        pred_y = float(self.state[1, 0])
        half = roi_size // 2

        x1 = max(0, min(fw - roi_size, int(round(pred_x)) - half))
        y1 = max(0, min(fh - roi_size, int(round(pred_y)) - half))
        x2 = min(fw, x1 + roi_size)
        y2 = min(fh, y1 + roi_size)
        if x2 - x1 < 64 or y2 - y1 < 64:
            return None
        return (x1, y1, x2, y2)

    def _predict_kalman(self, dt: float) -> Tuple[float, float]:
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, self.config.velocity_damping, 0.0],
                [0.0, 0.0, 0.0, self.config.velocity_damping],
            ],
            dtype=np.float64,
        )
        q_pos = float(self.config.process_noise_scale) ** 2
        q_vel = (float(self.config.process_noise_scale) * 25.0) ** 2
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def _update_kalman(self, meas_x: float, meas_y: float) -> None:
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        r = max(4.0, float(self.config.measurement_noise_scale))
        R = np.diag([r, r])
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        y = z - (H @ self.state)
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        I = np.eye(4, dtype=np.float64)
        self.cov = (I - K @ H) @ self.cov

    @staticmethod
    def _min_distance_to_players(
        cx: float,
        cy: float,
        player_detections: Optional[Sequence[DetectionRecord]],
    ) -> Optional[float]:
        if not player_detections:
            return None
        min_d = 999999.0
        for p in player_detections:
            px1, py1, px2, py2 = p.bbox
            fx = (px1 + px2) / 2.0
            fy = py2 - 0.05 * (py2 - py1)
            d = math.hypot(cx - fx, cy - fy)
            if d < min_d:
                min_d = d
        return min_d

    def _score_candidate(
        self,
        cand: DetectionRecord,
        pred_pos: Optional[Tuple[float, float]],
        max_gate: float,
        player_detections: Optional[Sequence[DetectionRecord]],
        med_ph: Optional[float],
    ) -> float:
        bx1, by1, bx2, by2 = cand.bbox
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        score = float(cand.confidence)

        if pred_pos is not None and max_gate > 0:
            dist_pred = math.hypot(cx - pred_pos[0], cy - pred_pos[1])
            score -= 0.35 * (dist_pred / max_gate)

        # Penalize candidates that are very far from all players when confidence is weak
        if player_detections and med_ph and med_ph > 20.0:
            min_p_dist = self._min_distance_to_players(cx, cy, player_detections)
            if min_p_dist is not None and min_p_dist > 3.2 * med_ph:
                score -= 0.18
        return score

    def step(
        self,
        frame_number: int,
        timestamp: float,
        ball_candidates: Sequence[DetectionRecord],
        frame_bgr: Optional[np.ndarray] = None,
        detection_source: str = "full_frame",
        player_detections: Optional[Sequence[DetectionRecord]] = None,
    ) -> BallTrajectoryRecord:
        """
        Process a single video frame and return a `BallTrajectoryRecord` with explicit provenance.
        """
        med_ph = _compute_median_player_height(player_detections or [])
        camera_cut = False

        if frame_bgr is not None and frame_bgr.size > 0:
            small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
            if self._prev_frame_small is not None and detect_camera_cut(
                self._prev_frame_small,
                small_curr,
                corr_threshold=self.config.camera_cut_hist_threshold,
                prev_median_player_h=self._prev_median_player_h,
                curr_median_player_h=med_ph,
            ):
                camera_cut = True
                self._reset_kalman_state()
            self._prev_frame_small = small_curr
        elif (
            self._prev_median_player_h is not None
            and med_ph is not None
            and detect_camera_cut(
                None,
                None,
                prev_median_player_h=self._prev_median_player_h,
                curr_median_player_h=med_ph,
            )
        ):
            camera_cut = True
            self._reset_kalman_state()

        if med_ph is not None:
            self._prev_median_player_h = med_ph

        # Filter out any candidates sitting inside blacklisted static background zones
        active_candidates = [
            c
            for c in ball_candidates
            if not any(
                math.hypot((c.bbox[0] + c.bbox[2]) / 2.0 - zx, (c.bbox[1] + c.bbox[3]) / 2.0 - zy) <= zr
                for zx, zy, zr in self.blacklisted_static_zones
            )
        ]

        dt = (
            max(1.0 / 120.0, float(timestamp - self.last_timestamp))
            if self.last_timestamp is not None and timestamp > self.last_timestamp
            else (1.0 / 25.0)
        )
        frame_delta = (
            max(1, int(frame_number - self.last_frame_number))
            if self.last_frame_number is not None and frame_number > self.last_frame_number
            else 1
        )

        rejected_outliers = 0
        chosen_det: Optional[DetectionRecord] = None
        was_lost = not self.is_tracking_active

        if not self.is_initialized:
            if active_candidates:
                chosen_det = max(
                    active_candidates,
                    key=lambda d: self._score_candidate(d, None, 0.0, player_detections, med_ph),
                )
                bx1, by1, bx2, by2 = chosen_det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                self.state = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
                self.cov = np.diag([36.0, 36.0, 25000.0, 25000.0])
                self.is_initialized = True
                self.frames_since_observed = 0
                self.last_observed_bbox_size = (max(6, bx2 - bx1), max(6, by2 - by1))
                self.last_confidence = float(chosen_det.confidence)
                self._static_anchor_pos = (cx, cy)
                self._static_isolated_streak = 1
        else:
            pred_x, pred_y = self._predict_kalman(dt)
            gate_steps = max(1, min(1 + self.frames_since_observed // 2, 4))
            max_gate = float(self.config.max_jump_px_per_frame) * gate_steps

            in_gate: List[Tuple[float, DetectionRecord]] = []
            out_of_gate: List[Tuple[float, DetectionRecord]] = []

            for cand in active_candidates:
                bx1, by1, bx2, by2 = cand.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist <= max_gate or not self.is_tracking_active:
                    score = self._score_candidate(
                        cand, (pred_x, pred_y), max_gate, player_detections, med_ph
                    )
                    in_gate.append((score, cand))
                else:
                    out_of_gate.append((dist, cand))

            if in_gate:
                in_gate.sort(key=lambda item: item[0], reverse=True)
                chosen_det = in_gate[0][1]
                rejected_outliers = len(out_of_gate)
                self._reacquire_candidate_pos = None
                self._reacquire_streak = 0
                if was_lost:
                    self.reacquisition_events += 1
            elif out_of_gate:
                best_out = max(
                    out_of_gate,
                    key=lambda item: self._score_candidate(
                        item[1], None, 0.0, player_detections, med_ph
                    ),
                )[1]
                bx1, by1, bx2, by2 = best_out.bbox
                ox, oy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                prev_reacq = self._reacquire_candidate_pos
                if prev_reacq is not None:
                    if math.hypot(ox - prev_reacq[0], oy - prev_reacq[1]) <= self.config.max_jump_px_per_frame * 1.5:
                        self._reacquire_streak += 1
                    else:
                        self._reacquire_streak = 1
                else:
                    self._reacquire_streak = 1
                self._reacquire_candidate_pos = (ox, oy)

                if self._reacquire_streak >= self.config.reacquire_confirm_frames:
                    chosen_det = best_out
                    vx_est = (ox - prev_reacq[0]) / max(1e-3, dt) if prev_reacq is not None else 0.0
                    vy_est = (oy - prev_reacq[1]) / max(1e-3, dt) if prev_reacq is not None else 0.0
                    self.state = np.array([[ox], [oy], [vx_est], [vy_est]], dtype=np.float64)
                    self.cov = np.diag([36.0, 36.0, 25000.0, 25000.0])
                    self._reacquire_candidate_pos = None
                    self._reacquire_streak = 0
                    self.reacquisition_events += 1
                else:
                    rejected_outliers = len(out_of_gate)

            if chosen_det is not None:
                bx1, by1, bx2, by2 = chosen_det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

                # Check for static isolated clutter lock (motionless object far from all players)
                if player_detections and len(player_detections) > 0:
                    min_p_d = self._min_distance_to_players(cx, cy, player_detections)
                    ref_h = med_ph if (med_ph and med_ph > 25.0) else 75.0
                    is_isolated = (min_p_d is not None) and (min_p_d > 2.2 * ref_h)
                    if self._static_anchor_pos is not None:
                        move_d = math.hypot(cx - self._static_anchor_pos[0], cy - self._static_anchor_pos[1])
                        if move_d <= 3.5 and is_isolated:
                            self._static_isolated_streak += 1
                        else:
                            self._static_anchor_pos = (cx, cy)
                            self._static_isolated_streak = 1 if is_isolated else 0
                    else:
                        self._static_anchor_pos = (cx, cy)
                        self._static_isolated_streak = 1 if is_isolated else 0

                    if self._static_isolated_streak >= 4:
                        # Static background false positive confirmed -> blacklist zone and reset track
                        self.blacklisted_static_zones.append((cx, cy, 24.0))
                        self.false_track_associations += 1
                        chosen_det = None
                        self._reset_kalman_state()

            if chosen_det is not None and self.is_initialized:
                bx1, by1, bx2, by2 = chosen_det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                self._update_kalman(cx, cy)
                self.frames_since_observed = 0
                self.last_observed_bbox_size = (max(6, bx2 - bx1), max(6, by2 - by1))
                self.last_confidence = float(chosen_det.confidence)
            elif self.is_initialized:
                self.frames_since_observed += frame_delta

        self.last_frame_number = int(frame_number)
        self.last_timestamp = float(timestamp)

        # Build output record based on observation mode
        if chosen_det is not None and self.is_initialized:
            bx1, by1, bx2, by2 = chosen_det.bbox
            cx_f = float(self.state[0, 0])
            cy_f = float(self.state[1, 0])
            vx_f = float(self.state[2, 0])
            vy_f = float(self.state[3, 0])
            speed = math.hypot(vx_f, vy_f)
            unc = float(math.sqrt(max(1.0, self.cov[0, 0] + self.cov[1, 1])))
            rec = BallTrajectoryRecord(
                match_id=self.match_id,
                frame_number=int(frame_number),
                timestamp=round(float(timestamp), 3),
                observation_mode=BallObservationMode.OBSERVED.value,
                is_observed=True,
                is_interpolated=False,
                bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                center_x=round(cx_f, 2),
                center_y=round(cy_f, 2),
                velocity_x=round(vx_f, 2),
                velocity_y=round(vy_f, 2),
                speed_px_per_sec=round(speed, 2),
                confidence=round(float(chosen_det.confidence), 4),
                uncertainty_radius_px=round(unc, 2),
                frames_since_observed=0,
                camera_cut_reset=camera_cut,
                rejected_outliers_count=rejected_outliers,
                detection_source=detection_source,
            )
        elif self.is_initialized and self.frames_since_observed <= self.config.max_coast_frames:
            cx_f = float(self.state[0, 0])
            cy_f = float(self.state[1, 0])
            vx_f = float(self.state[2, 0])
            vy_f = float(self.state[3, 0])
            speed = math.hypot(vx_f, vy_f)
            bw, bh = self.last_observed_bbox_size
            interp_bbox = (
                int(round(cx_f - bw / 2.0)),
                int(round(cy_f - bh / 2.0)),
                int(round(cx_f + bw / 2.0)),
                int(round(cy_f + bh / 2.0)),
            )
            decayed_conf = float(
                self.last_confidence * (self.config.interpolation_conf_decay ** self.frames_since_observed)
            )
            unc = float(math.sqrt(max(4.0, self.cov[0, 0] + self.cov[1, 1])))
            rec = BallTrajectoryRecord(
                match_id=self.match_id,
                frame_number=int(frame_number),
                timestamp=round(float(timestamp), 3),
                observation_mode=BallObservationMode.INTERPOLATED.value,
                is_observed=False,
                is_interpolated=True,
                bbox=interp_bbox,
                center_x=round(cx_f, 2),
                center_y=round(cy_f, 2),
                velocity_x=round(vx_f, 2),
                velocity_y=round(vy_f, 2),
                speed_px_per_sec=round(speed, 2),
                confidence=round(decayed_conf, 4),
                uncertainty_radius_px=round(unc, 2),
                frames_since_observed=int(self.frames_since_observed),
                camera_cut_reset=camera_cut,
                rejected_outliers_count=rejected_outliers,
                detection_source="kalman_interpolation",
            )
        else:
            rec = BallTrajectoryRecord(
                match_id=self.match_id,
                frame_number=int(frame_number),
                timestamp=round(float(timestamp), 3),
                observation_mode=BallObservationMode.LOST.value,
                is_observed=False,
                is_interpolated=False,
                bbox=None,
                center_x=None,
                center_y=None,
                velocity_x=0.0,
                velocity_y=0.0,
                speed_px_per_sec=0.0,
                confidence=0.0,
                uncertainty_radius_px=999.0,
                frames_since_observed=int(self.frames_since_observed),
                camera_cut_reset=camera_cut,
                rejected_outliers_count=rejected_outliers,
                detection_source="none",
            )

        self.history.append(rec)
        return rec

    def summarize(self) -> Dict[str, object]:
        """
        Summarize ball trajectory reliability metrics with strict separation of observed vs
        interpolated frames, continuity statistics, reacquisition events, and false track rejections.
        """
        total = len(self.history)
        observed = sum(1 for r in self.history if r.is_observed)
        interpolated = sum(1 for r in self.history if r.is_interpolated)
        lost = total - observed - interpolated
        cuts = sum(1 for r in self.history if r.camera_cut_reset)
        outliers = sum(r.rejected_outliers_count for r in self.history)

        # Compute continuous active track segment lengths
        segment_lengths: List[int] = []
        cur_len = 0
        for r in self.history:
            if r.camera_cut_reset and cur_len > 0:
                segment_lengths.append(cur_len)
                cur_len = 0
            if r.is_observed or r.is_interpolated:
                cur_len += 1
            elif cur_len > 0:
                segment_lengths.append(cur_len)
                cur_len = 0
        if cur_len > 0:
            segment_lengths.append(cur_len)

        mean_seg = float(np.mean(segment_lengths)) if segment_lengths else 0.0
        max_seg = int(max(segment_lengths)) if segment_lengths else 0

        return {
            "total_frames": total,
            "observed_frames": observed,
            "interpolated_frames": interpolated,
            "lost_frames": lost,
            "observed_coverage_pct": round(observed / float(max(1, total)) * 100.0, 2),
            "interpolated_coverage_pct": round(interpolated / float(max(1, total)) * 100.0, 2),
            "lost_pct": round(lost / float(max(1, total)) * 100.0, 2),
            "tracked_or_interpolated_pct": round(
                (observed + interpolated) / float(max(1, total)) * 100.0, 2
            ),
            "camera_cut_resets": cuts,
            "rejected_outlier_detections": outliers,
            "false_track_associations": int(self.false_track_associations),
            "blacklisted_static_zones_count": len(self.blacklisted_static_zones),
            "reacquisition_events": int(self.reacquisition_events),
            "active_track_segments_count": len(segment_lengths),
            "mean_continuous_track_frames": round(mean_seg, 2),
            "max_continuous_track_frames": max_seg,
        }
