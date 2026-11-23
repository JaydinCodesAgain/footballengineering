"""Motion-assisted small-football candidate recovery for fixed-camera panoramas.

Designed specifically for static elevated cameras where 6-14 px footballs shrink below
pretrained COCO YOLO's detection floor. Combines:
1. Signed positive foreground motion (`curr > prev` AND `curr > bg`) restricted strictly
   to the geometric playable pitch ROI — automatically eliminating trailing ghost blobs
   and dark moving player shadows on floodlit night turf.
2. Player-body masking so moving torsos, limbs, and boots do not trigger false candidates.
3. Local luminance contrast, compact size/aspect, and green-turf rejection filters.
4. 3-frame constant-velocity collinear trajectory confirmation (`t-2 -> t-1 -> t`) OR
   tight active Kalman trajectory corroboration — rejecting random glare/flicker FPs
   without weakening false-positive rejection.
5. Optional 2x-upscaled local YOLO patch verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.detection.detector import COCO_BALL_CLASS_ID
from src.geometry.pitch_region import PitchRegionFilter
from src.schemas import DetectionRecord


@dataclass
class MotionBallConfig:
    enabled: bool = True
    diff_threshold: int = 16
    bg_diff_threshold: int = 18
    bg_alpha: float = 0.08
    min_blob_size_px: int = 5
    max_blob_size_px: int = 24
    min_blob_area_px: int = 10
    max_blob_area_px: int = 340
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.25
    min_local_contrast: float = 22.0
    max_green_ratio: float = 0.58
    player_mask_pad_px: int = 7
    min_motion_step_px: float = 5.0
    max_motion_step_px: float = 190.0
    max_trajectory_residual_px: float = 22.0
    max_trajectory_residual_frac: float = 0.38
    kalman_corroboration_radius_px: float = 55.0
    max_candidates_per_frame: int = 3
    require_three_frame_or_kalman: bool = True
    require_kalman_corroboration: bool = True
    verify_with_upscaled_yolo: bool = False
    verify_roi_half_px: int = 128
    verify_upscale_factor: float = 2.0
    verify_conf_threshold: float = 0.12


@dataclass
class _RawMotionBlob:
    bbox: Tuple[int, int, int, int]
    cx: float
    cy: float
    area: float
    contrast: float
    score: float


class FixedCameraMotionBallDetector:
    """Causal motion-assisted small-ball proposal generator for fixed panoramic cameras."""

    def __init__(
        self,
        config: Optional[MotionBallConfig] = None,
        pitch_region_filter: Optional[PitchRegionFilter] = None,
    ) -> None:
        self.config = config or MotionBallConfig()
        self.pitch_region_filter = pitch_region_filter
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._pitch_mask_roi: Optional[np.ndarray] = None
        self._prev_gray_roi: Optional[np.ndarray] = None
        self._bg_gray_roi: Optional[np.ndarray] = None
        self._prev_blobs: List[_RawMotionBlob] = []
        self._prev2_blobs: List[_RawMotionBlob] = []
        self._prev_frame_number: Optional[int] = None
        self._prev2_frame_number: Optional[int] = None
        self._frame_count: int = 0

    def reset(self) -> None:
        self._prev_gray_roi = None
        self._bg_gray_roi = None
        self._prev_blobs.clear()
        self._prev2_blobs.clear()
        self._prev_frame_number = None
        self._prev2_frame_number = None
        self._frame_count = 0

    def _ensure_roi(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if self._roi is not None:
            return self._roi
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if self.pitch_region_filter is not None:
            rx1, ry1, rx2, ry2 = self.pitch_region_filter.get_tiling_roi_bbox((fh, fw))
            # Restrict raw motion proposals strictly to the core playable pitch (no sideline clutter)
            full_mask = self.pitch_region_filter.build_binary_mask((fh, fw), use_ball_margin=False)
            self._pitch_mask_roi = full_mask[ry1:ry2, rx1:rx2].copy()
        else:
            rx1, ry1, rx2, ry2 = 0, 0, fw, fh
            self._pitch_mask_roi = None
        self._roi = (rx1, ry1, rx2, ry2)
        return self._roi

    def _score_blob_appearance(
        self,
        roi_bgr: np.ndarray,
        roi_gray: np.ndarray,
        bx1: int,
        by1: int,
        bx2: int,
        by2: int,
    ) -> Optional[Tuple[float, float]]:
        """Check local luminance contrast and turf rejection on the tight foreground core."""
        rh, rw = roi_gray.shape[:2]
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < self.config.min_blob_size_px or bh < self.config.min_blob_size_px:
            return None
        if bw > self.config.max_blob_size_px or bh > self.config.max_blob_size_px:
            return None

        aspect = bw / float(max(1, bh))
        if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
            return None

        patch_gray = roi_gray[by1:by2, bx1:bx2]
        patch_bgr = roi_bgr[by1:by2, bx1:bx2]
        if patch_gray.size == 0 or patch_bgr.size == 0:
            return None

        # Surrounding ring (excluding slightly padded ball interior) for local contrast against turf
        pad = max(8, max(bw, bh))
        ex1, ey1 = max(0, bx1 - 2), max(0, by1 - 2)
        ex2, ey2 = min(rw, bx2 + 2), min(rh, by2 + 2)
        sx1, sy1 = max(0, ex1 - pad), max(0, ey1 - pad)
        sx2, sy2 = min(rw, ex2 + pad), min(rh, ey2 + pad)
        ring_gray = roi_gray[sy1:sy2, sx1:sx2]
        valid = np.ones(ring_gray.shape, dtype=bool)
        valid[ey1 - sy1 : ey2 - sy1, ex1 - sx1 : ex2 - sx1] = False
        ring_vals = ring_gray[valid]
        if ring_vals.size < 16:
            return None

        core_bright = float(np.percentile(patch_gray, 85))
        surround_med = float(np.median(ring_vals))
        contrast = core_bright - surround_med
        if contrast < self.config.min_local_contrast:
            return None

        # Compact 4-cardinal ring contrast check: a real football is brighter than surrounding turf
        # on at least 3 of 4 sides, whereas painted pitch-line flicker is bright along 2 opposite sides.
        ring_h, ring_w = ring_gray.shape[:2]
        icx = (ex1 - sx1 + ex2 - sx1) // 2
        icy = (ey1 - sy1 + ey2 - sy1) // 2
        top_strip = ring_gray[: max(1, ey1 - sy1), :]
        bot_strip = ring_gray[min(ring_h - 1, ey2 - sy1) :, :]
        left_strip = ring_gray[:, : max(1, ex1 - sx1)]
        right_strip = ring_gray[:, min(ring_w - 1, ex2 - sx1) :]
        darker_sides = 0
        for strip in (top_strip, bot_strip, left_strip, right_strip):
            if strip.size >= 4 and (core_bright - float(np.median(strip))) >= self.config.min_local_contrast * 0.55:
                darker_sides += 1
        if darker_sides < 3:
            return None

        # Reject pure green turf patches using the unpadded foreground core (a circle in a square has >= 22% corners)
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array([34, 40, 25], dtype=np.uint8),
            np.array([92, 255, 255], dtype=np.uint8),
        )
        green_ratio = float(np.mean(green > 0))
        if green_ratio > self.config.max_green_ratio:
            return None

        aspect_pen = abs(1.0 - min(aspect, 1.0 / max(aspect, 1e-6)))
        contrast_norm = min(1.0, (contrast - self.config.min_local_contrast) / 70.0)
        non_green_norm = min(1.0, max(0.0, 1.0 - green_ratio))
        quality = 0.50 * contrast_norm + 0.35 * non_green_norm + 0.15 * max(0.0, 1.0 - aspect_pen)
        conf = round(0.16 + 0.22 * float(np.clip(quality, 0.0, 1.0)), 4)
        return contrast, conf

    def _has_three_frame_trajectory(
        self,
        blob: _RawMotionBlob,
        dt1: float,
        dt2: float,
        max_step: float,
    ) -> Optional[float]:
        """Check whether `blob` completes a smooth 3-frame constant-velocity track `(p0 -> p1 -> blob)`.

        Returns the normalized residual error in `[0, 1]` when a collinear track is found, else `None`.
        """
        if not self._prev_blobs or not self._prev2_blobs:
            return None
        ratio = dt2 / max(1e-3, dt1)
        min_step = float(self.config.min_motion_step_px)
        best_res: Optional[float] = None

        for p1 in self._prev_blobs:
            dx2 = blob.cx - p1.cx
            dy2 = blob.cy - p1.cy
            step2 = math.hypot(dx2, dy2)
            if step2 < min_step or step2 > max_step:
                continue
            for p0 in self._prev2_blobs:
                dx1 = p1.cx - p0.cx
                dy1 = p1.cy - p0.cy
                step1 = math.hypot(dx1, dy1)
                if step1 < min_step or step1 > max_step:
                    continue
                # Reject back-and-forth flicker (require forward directional alignment cos >= 0.65)
                cos_ang = (dx1 * dx2 + dy1 * dy2) / max(1e-6, step1 * step2)
                if cos_ang < 0.65:
                    continue
                # Predict current position via constant-velocity extrapolation
                pred_x = p1.cx + dx1 * ratio
                pred_y = p1.cy + dy1 * ratio
                res_px = math.hypot(blob.cx - pred_x, blob.cy - pred_y)
                allowed_res = max(
                    float(self.config.max_trajectory_residual_px),
                    float(self.config.max_trajectory_residual_frac) * max(step1 * ratio, step2),
                )
                if res_px <= allowed_res:
                    norm_res = res_px / max(1.0, allowed_res)
                    if best_res is None or norm_res < best_res:
                        best_res = norm_res
        return best_res

    def detect_candidates(
        self,
        frame_bgr: np.ndarray,
        frame_number: int,
        timestamp: float,
        player_detections: Sequence[DetectionRecord],
        predicted_ball_pos: Optional[Tuple[float, float]] = None,
        blacklisted_zones: Optional[Sequence[Tuple[float, ...]]] = None,
        yolo_detector: Optional[object] = None,
    ) -> List[DetectionRecord]:
        """Extract motion-assisted small-ball candidates on a fixed-camera frame."""
        if not self.config.enabled or frame_bgr is None or frame_bgr.size == 0:
            return []

        rx1, ry1, rx2, ry2 = self._ensure_roi(frame_bgr.shape[:2])
        roi_bgr = frame_bgr[ry1:ry2, rx1:rx2]  # zero-copy view
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self._frame_count += 1

        if self._prev_gray_roi is None or self._bg_gray_roi is None or self._prev_gray_roi.shape != roi_gray.shape:
            self._prev_gray_roi = roi_gray
            self._bg_gray_roi = roi_gray.astype(np.float32)
            self._prev_blobs = []
            self._prev2_blobs = []
            self._prev_frame_number = int(frame_number)
            self._prev2_frame_number = None
            return []

        # Signed positive foreground motion: current pixel must be BRIGHTER than both previous frame
        # and running background. Eliminates trailing ghost blobs AND dark moving player shadows!
        curr_i16 = roi_gray.astype(np.int16)
        diff_prev_pos = curr_i16 - self._prev_gray_roi.astype(np.int16)
        diff_bg_pos = curr_i16 - self._bg_gray_roi.astype(np.int16)
        motion = (
            (diff_prev_pos >= int(self.config.diff_threshold))
            & (diff_bg_pos >= int(self.config.bg_diff_threshold))
        ).astype(np.uint8) * 255

        # Update causal background model AFTER diffing so moving balls aren't absorbed first
        cv2.accumulateWeighted(roi_gray, self._bg_gray_roi, float(self.config.bg_alpha))
        self._prev_gray_roi = roi_gray

        if self._pitch_mask_roi is not None:
            motion = cv2.bitwise_and(motion, self._pitch_mask_roi)

        # Mask out detected player bodies (with generous pad to exclude legs/boots/limbs)
        pad = int(self.config.player_mask_pad_px)
        rh, rw = motion.shape[:2]
        for p in player_detections:
            px1 = max(0, int(p.bbox[0]) - rx1 - pad)
            py1 = max(0, int(p.bbox[1]) - ry1 - pad)
            px2 = min(rw, int(p.bbox[2]) - rx1 + pad)
            py2 = min(rh, int(p.bbox[3]) - ry1 + pad)
            if px2 > px1 and py2 > py1:
                motion[py1:py2, px1:px2] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(motion, connectivity=8)
        raw_blobs: List[_RawMotionBlob] = []

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < self.config.min_blob_area_px or area > self.config.max_blob_area_px:
                continue
            bx = int(stats[lbl, cv2.CC_STAT_LEFT])
            by = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            bx1, by1 = bx, by
            bx2, by2 = min(rw, bx + bw), min(rh, by + bh)

            scored = self._score_blob_appearance(roi_bgr, roi_gray, bx1, by1, bx2, by2)
            if scored is None:
                continue
            contrast, conf = scored

            lx1, ly1 = max(0, bx - 2), max(0, by - 2)
            lx2, ly2 = min(rw, bx + bw + 2), min(rh, by + bh + 2)
            gx1, gy1, gx2, gy2 = lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1
            gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0

            if blacklisted_zones:
                if any(math.hypot(gcx - z[0], gcy - z[1]) <= z[2] for z in blacklisted_zones):
                    continue

            raw_blobs.append(
                _RawMotionBlob(
                    bbox=(gx1, gy1, gx2, gy2),
                    cx=gcx,
                    cy=gcy,
                    area=float(area),
                    contrast=contrast,
                    score=conf,
                )
            )

        # Temporal 3-frame constant-velocity trajectory confirmation OR tight Kalman corroboration
        dt2 = (
            float(max(1, int(frame_number - self._prev_frame_number)))
            if self._prev_frame_number is not None and frame_number > self._prev_frame_number
            else 2.0
        )
        dt1 = (
            float(max(1, int(self._prev_frame_number - self._prev2_frame_number)))
            if (
                self._prev_frame_number is not None
                and self._prev2_frame_number is not None
                and self._prev_frame_number > self._prev2_frame_number
            )
            else dt2
        )
        max_step = float(self.config.max_motion_step_px) * min(2.5, max(1.0, dt2 / 2.0))
        confirmed: List[Tuple[float, _RawMotionBlob]] = []

        for blob in raw_blobs:
            kalman_dist = (
                math.hypot(blob.cx - predicted_ball_pos[0], blob.cy - predicted_ball_pos[1])
                if predicted_ball_pos is not None
                else None
            )
            near_kalman = (
                kalman_dist is not None and kalman_dist <= float(self.config.kalman_corroboration_radius_px)
            )
            traj_res = self._has_three_frame_trajectory(blob, dt1, dt2, max_step)
            has_traj = traj_res is not None

            if self.config.require_kalman_corroboration:
                if not near_kalman:
                    continue
            elif self.config.require_three_frame_or_kalman and not (near_kalman or has_traj):
                continue

            boost = 0.0
            if near_kalman:
                boost += 0.10
            if has_traj:
                boost += 0.08 * (1.0 - float(traj_res or 0.0))
            confirmed.append((min(0.45, blob.score + boost), blob))

        # Shift 2-frame blob history
        self._prev2_blobs = self._prev_blobs
        self._prev2_frame_number = self._prev_frame_number
        self._prev_blobs = raw_blobs
        self._prev_frame_number = int(frame_number)

        confirmed.sort(key=lambda item: item[0], reverse=True)
        confirmed = confirmed[: int(self.config.max_candidates_per_frame)]

        results: List[DetectionRecord] = []
        if self.config.verify_with_upscaled_yolo and yolo_detector is not None and confirmed:
            fh, fw = frame_bgr.shape[:2]
            half = int(self.config.verify_roi_half_px)
            for conf_score, blob in confirmed[:3]:
                vx1 = max(0, int(round(blob.cx)) - half)
                vy1 = max(0, int(round(blob.cy)) - half)
                vx2 = min(fw, vx1 + 2 * half)
                vy2 = min(fh, vy1 + 2 * half)
                roi_dets = yolo_detector.detect_ball_in_roi(  # type: ignore[attr-defined]
                    frame_bgr,
                    (vx1, vy1, vx2, vy2),
                    frame_number=frame_number,
                    timestamp=timestamp,
                    conf_threshold=float(self.config.verify_conf_threshold),
                    upscale_factor=float(self.config.verify_upscale_factor),
                )
                matched_yolo = None
                for yd in roi_dets:
                    ycx = (yd.bbox[0] + yd.bbox[2]) / 2.0
                    ycy = (yd.bbox[1] + yd.bbox[3]) / 2.0
                    if math.hypot(ycx - blob.cx, ycy - blob.cy) <= 24.0:
                        matched_yolo = yd
                        break
                if matched_yolo is not None:
                    results.append(matched_yolo)
                elif (
                    predicted_ball_pos is not None
                    and math.hypot(blob.cx - predicted_ball_pos[0], blob.cy - predicted_ball_pos[1])
                    <= float(self.config.kalman_corroboration_radius_px)
                ):
                    results.append(
                        DetectionRecord(
                            frame_number=int(frame_number),
                            timestamp=round(float(timestamp), 3),
                            class_id=COCO_BALL_CLASS_ID,
                            class_name="ball",
                            bbox=blob.bbox,
                            confidence=round(conf_score, 4),
                            tracking_id=None,
                        )
                    )
            return results

        for conf_score, blob in confirmed:
            results.append(
                DetectionRecord(
                    frame_number=int(frame_number),
                    timestamp=round(float(timestamp), 3),
                    class_id=COCO_BALL_CLASS_ID,
                    class_name="ball",
                    bbox=blob.bbox,
                    confidence=round(conf_score, 4),
                    tracking_id=None,
                )
            )
        return results
