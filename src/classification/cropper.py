"""Torso cropping, quality filtering, background masking, and occlusion handling."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import cv2
import numpy as np

from src.config import CroppingConfig
from src.schemas import BoundingBox, CropQualityMetrics


def compute_blur_score(crop_bgr: np.ndarray) -> float:
    """Compute the Variance of Laplacian as a sharpness metric (higher is sharper)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return float(np.var(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_grass_mask(
    crop_bgr: np.ndarray,
    hsv_low: Tuple[int, int, int] = (33, 35, 25),
    hsv_high: Tuple[int, int, int] = (95, 255, 255),
) -> Tuple[np.ndarray, float]:
    """
    Compute a binary foreground mask (255 = foreground, 0 = pitch grass) and the grass pixel ratio.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), 1.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(
        hsv,
        np.array(hsv_low, dtype=np.uint8),
        np.array(hsv_high, dtype=np.uint8),
    )
    foreground_mask = cv2.bitwise_not(grass_mask)
    total_pixels = max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
    grass_pixels = int(cv2.countNonZero(grass_mask))
    grass_ratio = float(grass_pixels) / float(total_pixels)
    return foreground_mask, grass_ratio


def compute_occlusion_ratio(
    torso_bbox: Tuple[int, int, int, int],
    other_player_bboxes: Optional[Sequence[Sequence[float | int]]] = None,
    target_player_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """
    Estimate how much of a player's torso bounding box is overlapped by other detected players.
    Returns a ratio in [0.0, 1.0].
    """
    if not other_player_bboxes:
        return 0.0

    tx1, ty1, tx2, ty2 = torso_bbox
    torso_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)
    if torso_area <= 0:
        return 1.0

    max_overlap = 0.0
    for other in other_player_bboxes:
        ox1, oy1, ox2, oy2 = [int(round(float(c))) for c in other[:4]]
        if target_player_bbox is not None and (ox1, oy1, ox2, oy2) == target_player_bbox:
            continue

        ix1 = max(tx1, ox1)
        iy1 = max(ty1, oy1)
        ix2 = min(tx2, ox2)
        iy2 = min(ty2, oy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = float((ix2 - ix1) * (iy2 - iy1))
            overlap_ratio = inter_area / float(torso_area)
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio

    return float(min(1.0, max_overlap))


def extract_torso_crop(
    frame: np.ndarray,
    bbox_coords: Sequence[float | int],
    config: Optional[CroppingConfig] = None,
    other_player_bboxes: Optional[Sequence[Sequence[float | int]]] = None,
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int], CropQualityMetrics]:
    """
    Extract a player's torso region and compute quality diagnostics.

    Returns:
        (crop_bgr_or_None, torso_bbox, quality_metrics)
    """
    cfg = config or CroppingConfig()

    if frame is None or frame.ndim != 3 or frame.size == 0:
        empty_metrics = CropQualityMetrics(
            width=0,
            height=0,
            blur_score=0.0,
            grass_ratio=1.0,
            occlusion_ratio=0.0,
            edge_clipped=True,
            is_valid=False,
            rejection_reason="invalid_frame",
        )
        return None, (0, 0, 0, 0), empty_metrics

    frame_h, frame_w = frame.shape[:2]
    try:
        pbox = BoundingBox.from_coords(bbox_coords)
    except ValueError:
        empty_metrics = CropQualityMetrics(
            width=0,
            height=0,
            blur_score=0.0,
            grass_ratio=1.0,
            occlusion_ratio=0.0,
            edge_clipped=True,
            is_valid=False,
            rejection_reason="malformed_bbox",
        )
        return None, (0, 0, 0, 0), empty_metrics

    if pbox.width < cfg.min_player_width or pbox.height < cfg.min_player_height:
        metrics = CropQualityMetrics(
            width=pbox.width,
            height=pbox.height,
            blur_score=0.0,
            grass_ratio=1.0,
            occlusion_ratio=0.0,
            edge_clipped=False,
            is_valid=False,
            rejection_reason="player_bbox_too_small",
        )
        return None, (0, 0, 0, 0), metrics

    edge_clipped = (
        pbox.x1 <= 1
        or pbox.y1 <= 1
        or pbox.x2 >= frame_w - 1
        or pbox.y2 >= frame_h - 1
    )

    left = max(0, int(pbox.x1 + cfg.left_frac * pbox.width))
    right = min(frame_w, int(pbox.x1 + cfg.right_frac * pbox.width))
    top = max(0, int(pbox.y1 + cfg.top_frac * pbox.height))
    bottom = min(frame_h, int(pbox.y1 + cfg.bottom_frac * pbox.height))
    torso_bbox = (left, top, right, bottom)

    crop_w = right - left
    crop_h = bottom - top

    if crop_w < cfg.min_crop_width or crop_h < cfg.min_crop_height:
        metrics = CropQualityMetrics(
            width=max(0, crop_w),
            height=max(0, crop_h),
            blur_score=0.0,
            grass_ratio=1.0,
            occlusion_ratio=0.0,
            edge_clipped=edge_clipped,
            is_valid=False,
            rejection_reason="torso_crop_too_small",
        )
        return None, torso_bbox, metrics

    crop = frame[top:bottom, left:right].copy()
    blur_score = compute_blur_score(crop)
    _, grass_ratio = compute_grass_mask(crop, cfg.grass_hsv_low, cfg.grass_hsv_high)
    occlusion_ratio = compute_occlusion_ratio(
        torso_bbox=torso_bbox,
        other_player_bboxes=other_player_bboxes,
        target_player_bbox=pbox.to_tuple(),
    )

    rejection_reason: Optional[str] = None
    if blur_score < cfg.min_blur_variance:
        rejection_reason = f"too_blurry({blur_score:.1f}<{cfg.min_blur_variance})"
    elif grass_ratio > cfg.max_grass_ratio:
        rejection_reason = f"excessive_grass({grass_ratio:.2f}>{cfg.max_grass_ratio})"
    elif occlusion_ratio > cfg.max_occlusion_ratio:
        rejection_reason = f"heavily_occluded({occlusion_ratio:.2f}>{cfg.max_occlusion_ratio})"

    is_valid = rejection_reason is None
    metrics = CropQualityMetrics(
        width=crop_w,
        height=crop_h,
        blur_score=blur_score,
        grass_ratio=grass_ratio,
        occlusion_ratio=occlusion_ratio,
        edge_clipped=edge_clipped,
        is_valid=is_valid,
        rejection_reason=rejection_reason,
    )
    return (crop if is_valid else None), torso_bbox, metrics


def extract_legacy_jersey_crop(
    frame: np.ndarray,
    coords: Sequence[float | int],
    mode: str = "norway_wales",
) -> Optional[np.ndarray]:
    """
    Exact reproduction of the legacy notebook cropping functions for auditing existing index sets.
    mode='chelsea_sunderland' uses (0.20, 0.80, 0.20, 0.55), min_h=25.
    mode='norway_wales' uses (0.25, 0.75, 0.18, 0.55), min_h=30.
    """
    if frame is None or frame.size == 0:
        return None
    x1, y1, x2, y2 = map(int, coords[:4])
    player_w = x2 - x1
    player_h = y2 - y1
    min_h = 25 if mode == "chelsea_sunderland" else 30
    if player_w <= 0 or player_h < min_h:
        return None

    h, w = frame.shape[:2]
    if mode == "chelsea_sunderland":
        left = max(0, int(x1 + 0.20 * player_w))
        right = min(w, int(x1 + 0.80 * player_w))
        top = max(0, int(y1 + 0.20 * player_h))
        bottom = min(h, int(y1 + 0.55 * player_h))
    else:
        left = max(0, int(x1 + 0.25 * player_w))
        right = min(w, int(x1 + 0.75 * player_w))
        top = max(0, int(y1 + 0.18 * player_h))
        bottom = min(h, int(y1 + 0.55 * player_h))

    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    return crop.copy() if crop.size else None

