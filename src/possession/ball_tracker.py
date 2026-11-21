"""Persistent football tracking using a 2D constant-velocity Kalman filter with
pitch-horizon crowd suppression, motion-aware gating, set-piece-safe static-clutter
suppression, outlier rejection, and camera-cut resets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.config import BallTrackingConfig
from src.geometry.pitch_region import PitchRegionFilter, PitchZone
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
    3. Abrupt player bounding-box scale jump (`ratio > 2.25` or `< 0.44`) when cutting between
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


def estimate_pitch_horizon_y(frame_bgr: Optional[np.ndarray]) -> Optional[int]:
    """Estimate the upper pixel row (`y`) where the green football pitch begins.

    Returns `None` if the frame is not a wide elevated pitch view (e.g., player close-up
    or low-angle goalmouth view), ensuring legitimate low-angle balls are never suppressed.
    """
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return None
    fh, fw = frame_bgr.shape[:2]
    if fh < 60 or fw < 80:
        return None

    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array([30, 35, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    row_green_frac = np.mean(green > 0, axis=1)
    overall_green = float(np.mean(row_green_frac))
    green_rows_ratio = float(np.mean(row_green_frac >= 0.35))
    if overall_green < 0.25 or green_rows_ratio < 0.25:
        return None

    # Find first sustained pitch band (3 consecutive rows >= 22% green)
    for r in range(len(row_green_frac) - 2):
        if (
            row_green_frac[r] >= 0.22
            and row_green_frac[r + 1] >= 0.25
            and row_green_frac[r + 2] >= 0.25
        ):
            return int(round((r / 90.0) * fh))
    return None


def _compute_median_player_height(player_detections: Sequence[DetectionRecord]) -> Optional[float]:
    heights = [float(p.bbox[3] - p.bbox[1]) for p in player_detections if (p.bbox[3] - p.bbox[1]) > 15]
    if not heights:
        return None
    return float(np.median(heights))


def _is_pure_turf_patch(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """Detect pure uniform green-grass phantom detections (e.g. compression artifacts on turf)."""
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, min(fw - 1, int(bbox[0])))
    y1 = max(0, min(fh - 1, int(bbox[1])))
    x2 = max(x1 + 1, min(fw, int(bbox[2])))
    y2 = max(y1 + 1, min(fh, int(bbox[3])))
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array([30, 35, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    green_ratio = float(np.mean(green > 0))
    gray_std = float(np.std(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))
    return bool(green_ratio > 0.78 and gray_std < 18.0)


def _local_surrounding_green_ratio(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """Measure pitch-green pixel fraction in a ring strictly surrounding (excluding) a candidate box."""
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(10, int(max(x2 - x1, y2 - y1) * 1.4))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(fw, x2 + pad), min(fh, y2 + pad)
    ring = frame_bgr[ry1:ry2, rx1:rx2]
    if ring.size == 0:
        return 0.0
    hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
    # Strict pitch-green hue [36..92] excludes neon-yellow/gold [22..35]
    green = cv2.inRange(
        hsv,
        np.array([36, 35, 25], dtype=np.uint8),
        np.array([92, 255, 255], dtype=np.uint8),
    )
    # Exclude the candidate box interior so a greenish/yellowish object never counts as surrounding turf
    ix1, iy1 = max(0, x1 - rx1), max(0, y1 - ry1)
    ix2, iy2 = min(ring.shape[1], x2 - rx1), min(ring.shape[0], y2 - ry1)
    valid = np.ones(ring.shape[:2], dtype=bool)
    if ix2 > ix1 and iy2 > iy1:
        valid[iy1:iy2, ix1:ix2] = False
    n_valid = int(np.count_nonzero(valid))
    if n_valid <= 0:
        return 0.0
    return float(np.count_nonzero((green > 0) & valid)) / float(n_valid)


def _is_high_vis_steward_patch(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """Detect neon-yellow/lime high-vis steward vests in the stands often mistaken for a ball."""
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, min(fw - 1, int(bbox[0])))
    y1 = max(0, min(fh - 1, int(bbox[1])))
    x2 = max(x1 + 1, min(fw, int(bbox[2])))
    y2 = max(y1 + 1, min(fh, int(bbox[3])))
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # High-saturation neon yellow-green (H in 22..44, S >= 130, V >= 120)
    hivis = cv2.inRange(
        hsv,
        np.array([22, 130, 120], dtype=np.uint8),
        np.array([44, 255, 255], dtype=np.uint8),
    )
    return float(np.mean(hivis > 0)) >= 0.30


def filter_ball_candidates(
    ball_detections: Sequence[DetectionRecord],
    player_detections: Sequence[DetectionRecord],
    frame_shape: Tuple[int, int],
    min_size: int = 5,
    max_size: int = 140,
    blacklisted_zones: Optional[Sequence[Tuple[float, ...]]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    predicted_pos: Optional[Tuple[float, float]] = None,
    filter_crowd_horizon: bool = True,
    pitch_region_filter: Optional[PitchRegionFilter] = None,
) -> List[DetectionRecord]:
    """
    Filter false-positive ball detections caused by:
    - Broadcast scoreboards / corner channel watermarks,
    - Extreme aspect ratios (`< 0.42` or `> 2.35`),
    - Boxes disproportionate to player scale (`size > 0.38 * median_player_height`, e.g. white shorts),
    - Low-confidence triggers on white jerseys/badges inside upper player torsos or centered white socks,
    - Pure uniform pitch-green grass phantoms (`conf < 0.35`),
    - Stadium crowd/steward detections sitting above the green pitch horizon without trajectory continuity,
    - Active static background clutter zones (`blacklisted_zones`).
    """
    if not ball_detections:
        return []

    fh, fw = frame_shape[:2]
    med_ph = _compute_median_player_height(player_detections)
    dynamic_max_size = max_size
    if med_ph is not None and med_ph > 25.0:
        # A football diameter (~22cm) is ~12%-25% of player height; allow up to 38% for motion blur
        dynamic_max_size = min(max_size, max(26, int(round(med_ph * 0.38))))

    pitch_top_y = (
        estimate_pitch_horizon_y(frame_bgr)
        if (filter_crowd_horizon and frame_bgr is not None)
        else None
    )

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

        # Exclude top-left / top-right corner scoreboard & circular team watermark zones
        in_corner_overlay = (
            cy < fh * 0.18 and (cx < fw * 0.27 or cx > fw * 0.85)
        )
        if in_corner_overlay and (pitch_top_y is None or cy < pitch_top_y) and b.confidence < 0.80:
            continue

        # Exclude known static background clutter hot-spots
        if blacklisted_zones:
            if any(math.hypot(cx - z[0], cy - z[1]) <= z[2] for z in blacklisted_zones):
                continue

        near_predicted = (
            predicted_pos is not None
            and math.hypot(cx - predicted_pos[0], cy - predicted_pos[1]) <= 90.0
        )

        # Geometric fixed-camera pitch-region filter (eliminates sideline spare balls, benches, ads)
        if pitch_region_filter is not None:
            if not pitch_region_filter.is_ball_in_region(
                b.bbox,
                include_margin=True,
                allow_aerial_above_far_line=True,
                near_active_trajectory=near_predicted,
            ):
                continue
            # Reject isolated stationary spare balls / cones sitting OUTSIDE the playable touchlines
            # (in BOUNDARY_MARGIN) when there is no active ball trajectory and no nearby throw-in/corner player.
            b_zone = pitch_region_filter.classify_point(cx, cy, for_ball=True)
            if b_zone == PitchZone.BOUNDARY_MARGIN and not near_predicted:
                ref_h = med_ph if (med_ph is not None and med_ph > 20.0) else 65.0
                near_any_player = False
                for p in player_detections:
                    px1, py1, px2, py2 = p.bbox
                    pfx = (px1 + px2) / 2.0
                    pfy = py2 - 0.15 * (py2 - py1)
                    if math.hypot(cx - pfx, cy - pfy) <= 1.8 * ref_h:
                        near_any_player = True
                        break
                if not near_any_player:
                    continue

        # Reject stadium crowd / high-vis steward detections above the green pitch horizon
        if pitch_top_y is not None and frame_bgr is not None and cy < pitch_top_y - 0.025 * fh:
            surrounding_green = _local_surrounding_green_ratio(frame_bgr, b.bbox)
            if surrounding_green < 0.05:
                if _is_high_vis_steward_patch(frame_bgr, b.bbox):
                    continue
                # Allow legitimate lofted aerial balls above the stands ONLY when connected
                # to an active trajectory rising from the pitch
                if not near_predicted:
                    continue

        # Reject pure green turf phantom detections at low confidence
        if frame_bgr is not None and b.confidence < 0.35 and _is_pure_turf_patch(frame_bgr, b.bbox):
            continue

        # Reject oversized close-up player clothing patches (e.g. white shorts in full-screen close-ups)
        if med_ph is not None and med_ph > 0.44 * fh and max(bw, bh) > 60:
            if any(
                p.bbox[0] <= cx <= p.bbox[2] and p.bbox[1] <= cy <= p.bbox[3]
                for p in player_detections
            ):
                continue

        # Check if low-confidence ball sits inside a player's upper chest/head (badge/collar)
        # or centered on a player's lower-leg/white-sock without trajectory support
        is_player_body_fp = False
        if b.confidence < 0.26:
            for p in player_detections:
                px1, py1, px2, py2 = p.bbox
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)
                if px1 <= cx <= px2 and (py1 + 0.10 * ph) <= cy <= (py1 + 0.62 * ph):
                    is_player_body_fp = True
                    break
                if (
                    b.confidence < 0.24
                    and not near_predicted
                    and (px1 + 0.18 * pw) <= cx <= (px2 - 0.18 * pw)
                    and (py1 + 0.65 * ph) <= cy <= (py1 + 0.88 * ph)
                ):
                    is_player_body_fp = True
                    break
        if is_player_body_fp:
            continue

        filtered.append(b)

    return filtered


class KalmanBallTracker:
    """
    Dedicated single-ball tracker using a 4D constant-velocity Kalman filter `[x, y, vx, vy]`.

    Reliability features:
    - Multi-candidate Mahalanobis/motion-aware gating and player-proximity + velocity-direction scoring.
    - Two-frame motion-consistent initialization confirmation for low-confidence detections
      (`< init_high_conf_threshold`), preventing single-frame grass/sock/clutter phantoms from
      hijacking a lost track while preserving immediate lock on high-confidence balls.
    - Set-piece-safe static background clutter detection & time-bounded blacklisting:
      zones expire after `static_zone_ttl_frames`, clear on broadcast camera cuts, and are
      automatically released if a player approaches within `1.5 * median_player_height`
      (protecting stationary free kicks, penalties, corners, and kickoffs).
    - Outlier jump rejection with multi-frame re-acquisition confirmation for fast long passes.
    - Short-gap coasting interpolation (`observation_mode='interpolated'`) strictly separated
      from genuine detector observations (`observation_mode='observed'`).
    - Automatic reset across broadcast camera cuts with wipe-cooldown protection so multi-frame
      replay wipes do not spawn artificial 1-frame shots.
    """

    def __init__(
        self,
        match_id: str = "match",
        config: Optional[BallTrackingConfig] = None,
        pitch_region_filter: Optional[PitchRegionFilter] = None,
    ) -> None:
        self.match_id = match_id
        self.config = config or BallTrackingConfig()
        self.pitch_region_filter = pitch_region_filter
        # Entries: (cx, cy, radius, created_frame, is_permanent)
        self._blacklisted_zone_entries: List[Tuple[float, float, float, int, bool]] = []
        # Tracks recurring stationary isolated candidate spots before they can repeatedly re-trigger
        self._static_spot_counts: List[List[float]] = []  # [cx, cy, hits, last_frame]
        self.false_track_associations: int = 0
        self.reacquisition_events: int = 0
        self.history: List[BallTrajectoryRecord] = []
        self._prev_frame_small: Optional[np.ndarray] = None
        self._prev_median_player_h: Optional[float] = None
        self._last_cut_frame: Optional[int] = None
        self._reset_kalman_state()

    @property
    def blacklisted_static_zones(self) -> List[Tuple[float, float, float]]:
        """Expose active static clutter zones as `(cx, cy, radius)` tuples."""
        return [(zx, zy, zr) for zx, zy, zr, _, _ in self._blacklisted_zone_entries]

    @property
    def permanent_static_zones(self) -> List[Tuple[float, float, float]]:
        """Expose persistent fixed-camera static distractor zones that do not expire by TTL."""
        return [(zx, zy, zr) for zx, zy, zr, _, perm in self._blacklisted_zone_entries if perm]

    def _register_static_distractor(
        self,
        cx: float,
        cy: float,
        frame_number: int,
        radius: Optional[float] = None,
    ) -> None:
        """Add a confirmed static distractor zone; permanent on fixed cameras (especially off-pitch)."""
        rad = float(radius if radius is not None else self.config.static_distractor_radius_px)
        # Avoid duplicates near an existing zone
        for idx, (zx, zy, zr, created_f, perm) in enumerate(self._blacklisted_zone_entries):
            if math.hypot(cx - zx, cy - zy) <= max(rad, zr):
                # Escalate repeated triggers at the exact same spot to permanent when persistent mode is on
                new_perm = perm or bool(self.config.fixed_camera or self.config.persistent_static_distractors)
                self._blacklisted_zone_entries[idx] = (zx, zy, max(rad, zr), int(frame_number), new_perm)
                return

        is_permanent = False
        if self.config.fixed_camera or self.config.persistent_static_distractors:
            if self.pitch_region_filter is not None:
                zone = self.pitch_region_filter.classify_point(cx, cy, for_ball=True)
                # Outside core playable pitch (margin/curb/sideline): always permanent
                is_permanent = zone != PitchZone.PLAYABLE or bool(self.config.persistent_static_distractors)
            else:
                is_permanent = True

        self._blacklisted_zone_entries.append((float(cx), float(cy), rad, int(frame_number), is_permanent))

    def _update_static_spot_detector(
        self,
        ball_candidates: Sequence[DetectionRecord],
        frame_number: int,
        player_detections: Optional[Sequence[DetectionRecord]],
        med_ph: Optional[float],
    ) -> None:
        """On fixed cameras, detect motionless isolated candidate hotspots across frames so a
        stationary sideline spare ball or marker is masked persistently even without a track lock.
        """
        if not (self.config.fixed_camera or self.config.persistent_static_distractors):
            return
        if not ball_candidates:
            return
        ref_h = med_ph if (med_ph and med_ph > 20.0) else 65.0
        min_hits = max(3, int(self.config.static_distractor_min_hits))
        for cand in ball_candidates:
            cx = (cand.bbox[0] + cand.bbox[2]) / 2.0
            cy = (cand.bbox[1] + cand.bbox[3]) / 2.0
            min_p_d = self._min_distance_to_players(cx, cy, player_detections)
            is_isolated = (min_p_d is None) or (min_p_d > 2.0 * ref_h)
            if not is_isolated:
                continue
            matched = False
            for spot in self._static_spot_counts:
                if math.hypot(cx - spot[0], cy - spot[1]) <= 6.0:
                    if frame_number > int(spot[3]):
                        spot[2] += 1.0
                        spot[3] = float(frame_number)
                    if int(spot[2]) >= min_hits:
                        self._register_static_distractor(spot[0], spot[1], frame_number)
                    matched = True
                    break
            if not matched:
                self._static_spot_counts.append([cx, cy, 1.0, float(frame_number)])
        # Bound bookkeeping list to recent/active spots
        if len(self._static_spot_counts) > 64:
            self._static_spot_counts = [s for s in self._static_spot_counts if frame_number - int(s[3]) <= 300]

    def _prune_blacklisted_zones(
        self,
        frame_number: int,
        player_detections: Optional[Sequence[DetectionRecord]],
        med_ph: Optional[float],
    ) -> None:
        """Expire aged broadcast static zones and release playable-pitch zones approached by a player.

        Permanent fixed-camera distractor zones outside the core playable pitch are never expired
        by TTL so stationary sideline spare balls cannot reappear later in the match.
        """
        if not self._blacklisted_zone_entries:
            return
        ttl = max(20, int(self.config.static_zone_ttl_frames))
        ref_h = med_ph if (med_ph and med_ph > 25.0) else 75.0
        kept: List[Tuple[float, float, float, int, bool]] = []
        for zx, zy, zr, created_f, is_permanent in self._blacklisted_zone_entries:
            if not is_permanent and (frame_number - created_f > ttl):
                continue
            # Release a zone if a player approaches ONLY when the spot lies inside the playable pitch
            # (protects free kicks/corners/kickoffs without un-masking off-pitch concrete-curb spare balls).
            zone_in_playable = (
                self.pitch_region_filter is None
                or self.pitch_region_filter.classify_point(zx, zy, for_ball=False) == PitchZone.PLAYABLE
            )
            if player_detections and zone_in_playable:
                min_p_d = self._min_distance_to_players(zx, zy, player_detections)
                if not is_permanent and min_p_d is not None and min_p_d <= 1.5 * ref_h:
                    continue
                if (
                    is_permanent
                    and self.pitch_region_filter is not None
                    and min_p_d is not None
                    and min_p_d <= 1.2 * ref_h
                ):
                    continue
            kept.append((zx, zy, zr, created_f, is_permanent))
        self._blacklisted_zone_entries = kept

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
        self._init_candidate_pos: Optional[Tuple[float, float]] = None
        self._init_streak: int = 0
        self._static_anchor_pos: Optional[Tuple[float, float]] = None
        self._static_isolated_streak: int = 0

    def reset(self) -> None:
        """Full reset of tracker state and history."""
        self._blacklisted_zone_entries.clear()
        self._static_spot_counts.clear()
        self.false_track_associations = 0
        self.reacquisition_events = 0
        self.history.clear()
        self._prev_frame_small = None
        self._prev_median_player_h = None
        self._last_cut_frame = None
        self._reset_kalman_state()

    @property
    def is_tracking_active(self) -> bool:
        return self.is_initialized and self.frames_since_observed <= self.config.max_coast_frames

    def get_predicted_position(self) -> Optional[Tuple[float, float]]:
        """Return current predicted `(x, y)` ball position when tracker is active."""
        if not self.is_tracking_active:
            return None
        return (float(self.state[0, 0]), float(self.state[1, 0]))

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
        prev_pos: Optional[Tuple[float, float]] = None,
        pred_vel: Optional[Tuple[float, float]] = None,
    ) -> float:
        bx1, by1, bx2, by2 = cand.bbox
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        score = float(cand.confidence)

        if pred_pos is not None and max_gate > 0:
            dist_pred = math.hypot(cx - pred_pos[0], cy - pred_pos[1])
            score -= 0.42 * (dist_pred / max_gate)

            # Box size consistency against established ball size
            last_diag = math.hypot(*self.last_observed_bbox_size)
            curr_diag = math.hypot(bw, bh)
            size_ratio = max(curr_diag, last_diag) / max(1.0, min(curr_diag, last_diag))
            if size_ratio > 2.2:
                score -= 0.14

        # Motion-direction consistency during active fast ball flight
        if prev_pos is not None and pred_vel is not None:
            v_mag = math.hypot(pred_vel[0], pred_vel[1])
            dx = cx - prev_pos[0]
            dy = cy - prev_pos[1]
            d_mag = math.hypot(dx, dy)
            if v_mag > 160.0 and d_mag > 12.0:
                cos_align = (dx * pred_vel[0] + dy * pred_vel[1]) / (d_mag * v_mag)
                score += 0.12 * cos_align

        # Penalize isolated weak candidates very far from all players
        if player_detections and med_ph and med_ph > 20.0:
            min_p_dist = self._min_distance_to_players(cx, cy, player_detections)
            if min_p_dist is not None and min_p_dist > 3.2 * med_ph:
                score -= 0.18
        return score

    def _try_initialize_from_candidates(
        self,
        active_candidates: Sequence[DetectionRecord],
        dt: float,
        player_detections: Optional[Sequence[DetectionRecord]],
        med_ph: Optional[float],
    ) -> Optional[DetectionRecord]:
        """Initialize (or re-initialize from LOST) with confirmation for low-confidence detections."""
        if not active_candidates:
            self._init_candidate_pos = None
            self._init_streak = 0
            return None

        best = max(
            active_candidates,
            key=lambda d: self._score_candidate(d, None, 0.0, player_detections, med_ph),
        )
        bx1, by1, bx2, by2 = best.bbox
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        high_conf = float(best.confidence) >= float(self.config.init_high_conf_threshold)
        need_confirm = max(1, int(self.config.init_confirm_frames))

        if high_conf or need_confirm <= 1:
            vx_est, vy_est = 0.0, 0.0
            if self._init_candidate_pos is not None:
                d_prev = math.hypot(cx - self._init_candidate_pos[0], cy - self._init_candidate_pos[1])
                if d_prev <= self.config.max_jump_px_per_frame * 1.4:
                    vx_est = (cx - self._init_candidate_pos[0]) / max(1e-3, dt)
                    vy_est = (cy - self._init_candidate_pos[1]) / max(1e-3, dt)
            self._init_candidate_pos = None
            self._init_streak = 0
            self.state = np.array([[cx], [cy], [vx_est], [vy_est]], dtype=np.float64)
            self.cov = np.diag([36.0, 36.0, 25000.0, 25000.0])
            self.is_initialized = True
            self.frames_since_observed = 0
            self.last_observed_bbox_size = (max(6, bx2 - bx1), max(6, by2 - by1))
            self.last_confidence = float(best.confidence)
            self._static_anchor_pos = (cx, cy)
            self._static_isolated_streak = 1
            return best

        # Low-confidence candidate requires consecutive spatially consistent confirmation
        prev_init = self._init_candidate_pos
        if (
            prev_init is not None
            and math.hypot(cx - prev_init[0], cy - prev_init[1]) <= self.config.max_jump_px_per_frame * 1.4
        ):
            self._init_streak += 1
        else:
            self._init_streak = 1
        self._init_candidate_pos = (cx, cy)

        if self._init_streak >= need_confirm:
            vx_est = (cx - prev_init[0]) / max(1e-3, dt) if prev_init is not None else 0.0
            vy_est = (cy - prev_init[1]) / max(1e-3, dt) if prev_init is not None else 0.0
            self._init_candidate_pos = None
            self._init_streak = 0
            self.state = np.array([[cx], [cy], [vx_est], [vy_est]], dtype=np.float64)
            self.cov = np.diag([36.0, 36.0, 25000.0, 25000.0])
            self.is_initialized = True
            self.frames_since_observed = 0
            self.last_observed_bbox_size = (max(6, bx2 - bx1), max(6, by2 - by1))
            self.last_confidence = float(best.confidence)
            self._static_anchor_pos = (cx, cy)
            self._static_isolated_streak = 1
            return best

        return None

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
        raw_cut = False
        camera_cut = False

        if not self.config.fixed_camera:
            if frame_bgr is not None and frame_bgr.size > 0:
                small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
                if self._prev_frame_small is not None and detect_camera_cut(
                    self._prev_frame_small,
                    small_curr,
                    corr_threshold=self.config.camera_cut_hist_threshold,
                    prev_median_player_h=self._prev_median_player_h,
                    curr_median_player_h=med_ph,
                ):
                    raw_cut = True
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
                raw_cut = True

        if raw_cut:
            # Always reset tracker state and clear non-permanent broadcast static zones across cuts
            self._reset_kalman_state()
            self._blacklisted_zone_entries = [z for z in self._blacklisted_zone_entries if z[4]]
            cooldown = max(0, int(self.config.camera_cut_cooldown_frames))
            if self._last_cut_frame is None or (frame_number - self._last_cut_frame) > cooldown:
                camera_cut = True
                self._last_cut_frame = int(frame_number)

        if med_ph is not None:
            self._prev_median_player_h = med_ph

        # Maintain static clutter blacklist & fixed-camera persistent static spot detector
        self._prune_blacklisted_zones(frame_number, player_detections, med_ph)
        self._update_static_spot_detector(ball_candidates, frame_number, player_detections, med_ph)

        # Filter out any candidates sitting inside active blacklisted static background zones
        active_candidates = [
            c
            for c in ball_candidates
            if not any(
                math.hypot((c.bbox[0] + c.bbox[2]) / 2.0 - zx, (c.bbox[1] + c.bbox[3]) / 2.0 - zy) <= zr
                for zx, zy, zr, _, _ in self._blacklisted_zone_entries
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
        did_fresh_init = False

        if not self.is_tracking_active:
            chosen_det = self._try_initialize_from_candidates(
                active_candidates, dt, player_detections, med_ph
            )
            if chosen_det is not None:
                did_fresh_init = True
                if was_lost and self.history:
                    self.reacquisition_events += 1
        else:
            prev_pos = (float(self.state[0, 0]), float(self.state[1, 0]))
            pred_vel = (float(self.state[2, 0]), float(self.state[3, 0]))
            pred_x, pred_y = self._predict_kalman(dt)
            gate_steps = max(1, min(1 + self.frames_since_observed // 2, 3))
            max_gate = min(
                float(self.config.max_jump_px_per_frame) * gate_steps,
                float(self.config.max_jump_px_per_frame) * 2.4,
            )

            in_gate: List[Tuple[float, DetectionRecord]] = []
            out_of_gate: List[Tuple[float, DetectionRecord]] = []

            for cand in active_candidates:
                bx1, by1, bx2, by2 = cand.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist <= max_gate:
                    score = self._score_candidate(
                        cand,
                        (pred_x, pred_y),
                        max_gate,
                        player_detections,
                        med_ph,
                        prev_pos=prev_pos,
                        pred_vel=pred_vel,
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
                    did_fresh_init = True
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
                        # Static background false positive confirmed -> blacklist (permanent on fixed camera) & reset
                        self._register_static_distractor(cx, cy, int(frame_number))
                        self.false_track_associations += 1
                        chosen_det = None
                        self._reset_kalman_state()

            if chosen_det is not None and self.is_initialized:
                bx1, by1, bx2, by2 = chosen_det.bbox
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if not did_fresh_init:
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
            "blacklisted_static_zones_count": len(self._blacklisted_zone_entries),
            "permanent_static_zones_count": sum(1 for z in self._blacklisted_zone_entries if z[4]),
            "reacquisition_events": int(self.reacquisition_events),
            "active_track_segments_count": len(segment_lengths),
            "mean_continuous_track_frames": round(mean_seg, 2),
            "max_continuous_track_frames": max_seg,
        }
