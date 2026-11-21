"""Native-resolution overlapping multi-tile inference utilities for panoramic football video.

Provides zero-copy NumPy tile-grid generation, automatic panoramic letterbox cropping,
tile-to-panorama coordinate projection, and seam-aware class-specific NMS + containment
deduplication to prevent duplicated players or balls across tile boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.evaluation.detection_eval import compute_iou
from src.schemas import DetectionRecord


@dataclass(frozen=True)
class TileSpec:
    """Single overlapping tile window inside a full panoramic frame."""

    tile_index: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    has_left_neighbor: bool
    has_right_neighbor: bool
    has_top_neighbor: bool
    has_bottom_neighbor: bool

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def estimate_active_content_bbox(
    frame_bgr: np.ndarray,
    black_threshold: int = 12,
    min_nonblack_frac: float = 0.06,
    pad_frac: float = 0.03,
) -> Tuple[int, int, int, int]:
    """Detect non-black panoramic content bounds (skipping fisheye top/bottom letterbox bars).

    Operates on a lightweight 160x90 downscale (< 0.3 ms) and returns `(x1, y1, x2, y2)`
    in native frame pixel coordinates with a conservative safety margin.
    """
    fh, fw = frame_bgr.shape[:2]
    if fh < 64 or fw < 64:
        return (0, 0, fw, fh)

    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    nonblack = gray > black_threshold

    row_frac = np.mean(nonblack, axis=1)
    col_frac = np.mean(nonblack, axis=0)
    active_rows = np.where(row_frac >= min_nonblack_frac)[0]
    active_cols = np.where(col_frac >= min_nonblack_frac)[0]

    if len(active_rows) == 0 or len(active_cols) == 0:
        return (0, 0, fw, fh)

    ry1 = int(active_rows[0])
    ry2 = int(active_rows[-1]) + 1
    cx1 = int(active_cols[0])
    cx2 = int(active_cols[-1]) + 1

    pad_y = int(round(fh * pad_frac))
    pad_x = int(round(fw * pad_frac))

    y1 = max(0, int(math.floor((ry1 / 90.0) * fh)) - pad_y)
    y2 = min(fh, int(math.ceil((ry2 / 90.0) * fh)) + pad_y)
    x1 = max(0, int(math.floor((cx1 / 160.0) * fw)) - pad_x)
    x2 = min(fw, int(math.ceil((cx2 / 160.0) * fw)) + pad_x)

    if (x2 - x1) < int(fw * 0.25) or (y2 - y1) < int(fh * 0.25):
        return (0, 0, fw, fh)
    return (x1, y1, x2, y2)


def _compute_axis_starts(span_start: int, span_end: int, tile_size: int, overlap: float) -> List[Tuple[int, int]]:
    """Compute evenly spaced overlapping 1D intervals `[a, b)` covering `[span_start, span_end)`."""
    span = span_end - span_start
    if span <= 0:
        return []
    eff_size = min(max(64, int(tile_size)), span)
    if eff_size >= span:
        return [(span_start, span_end)]

    clamped_overlap = max(0.0, min(0.49, float(overlap)))
    max_step = max(1, int(math.floor(eff_size * (1.0 - clamped_overlap))))
    n_tiles = 1 + int(math.ceil(max(0, span - eff_size) / float(max_step)))
    if n_tiles <= 1:
        return [(span_start, span_end)]

    # Distribute tile origins evenly across [span_start, span_end - eff_size] so overlap is uniform
    # and the final tile lands flush on span_end without creating tiny sliver tiles.
    max_origin = span_end - eff_size
    origins = np.linspace(span_start, max_origin, n_tiles)
    intervals: List[Tuple[int, int]] = []
    for o in origins:
        a = int(round(float(o)))
        b = min(span_end, a + eff_size)
        a = max(span_start, b - eff_size)
        if not intervals or a != intervals[-1][0] or b != intervals[-1][1]:
            intervals.append((a, b))
    return intervals


def generate_tile_grid(
    frame_shape: Tuple[int, int],
    tile_width: int = 1280,
    tile_height: int = 1080,
    overlap: float = 0.22,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> List[TileSpec]:
    """Generate an overlapping 2D tile grid covering `roi_bbox` (or the full frame)."""
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    if fh <= 0 or fw <= 0:
        return []

    if roi_bbox is not None:
        rx1 = max(0, min(fw - 1, int(roi_bbox[0])))
        ry1 = max(0, min(fh - 1, int(roi_bbox[1])))
        rx2 = max(rx1 + 1, min(fw, int(roi_bbox[2])))
        ry2 = max(ry1 + 1, min(fh, int(roi_bbox[3])))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, fw, fh

    x_intervals = _compute_axis_starts(rx1, rx2, tile_width, overlap)
    y_intervals = _compute_axis_starts(ry1, ry2, tile_height, overlap)

    tiles: List[TileSpec] = []
    n_rows = len(y_intervals)
    n_cols = len(x_intervals)
    idx = 0
    for r, (ty1, ty2) in enumerate(y_intervals):
        for c, (tx1, tx2) in enumerate(x_intervals):
            tiles.append(
                TileSpec(
                    tile_index=idx,
                    row=r,
                    col=c,
                    x1=tx1,
                    y1=ty1,
                    x2=tx2,
                    y2=ty2,
                    has_left_neighbor=(c > 0),
                    has_right_neighbor=(c < n_cols - 1),
                    has_top_neighbor=(r > 0),
                    has_bottom_neighbor=(r < n_rows - 1),
                )
            )
            idx += 1
    return tiles


def extract_tile_views(
    frame_bgr: np.ndarray,
    tiles: Sequence[TileSpec],
) -> List[np.ndarray]:
    """Return zero-copy NumPy array views for every tile in `tiles`."""
    return [frame_bgr[t.y1 : t.y2, t.x1 : t.x2] for t in tiles]


def compute_iom(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Minimum-area (containment ratio in [0, 1])."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter) / float(min(area_a, area_b))


def _is_clipped_at_internal_seam(
    bbox: Tuple[int, int, int, int],
    tile: TileSpec,
    margin_px: int = 5,
) -> bool:
    """Return True if a full-frame-mapped bbox touches an internal seam shared with an adjacent tile."""
    x1, y1, x2, y2 = bbox
    if tile.has_left_neighbor and (x1 - tile.x1) <= margin_px:
        return True
    if tile.has_right_neighbor and (tile.x2 - x2) <= margin_px:
        return True
    if tile.has_top_neighbor and (y1 - tile.y1) <= margin_px:
        return True
    if tile.has_bottom_neighbor and (tile.y2 - y2) <= margin_px:
        return True
    return False


def merge_tiled_detections(
    tiled_detections: Sequence[Tuple[DetectionRecord, Optional[TileSpec]]],
    frame_shape: Tuple[int, int],
    player_nms_iou: float = 0.45,
    player_containment_iom: float = 0.72,
    ball_nms_iou: float = 0.35,
    seam_margin_px: int = 5,
) -> List[DetectionRecord]:
    """Merge detections mapped from overlapping tiles using seam-aware class-specific NMS.

    1. Partition strictly by `class_name` so players and balls never suppress each other.
    2. Suppress boxes truncated against an internal tile seam whenever an unclipped box of
       the same class from another tile overlaps it (`IoM >= 0.55` or `IoU >= 0.35`).
    3. Apply class-aware NMS (`IoU >= threshold`) plus containment deduplication
       (`IoM >= player_containment_iom` for players; centre proximity for small balls)
       to prevent duplicated players or shifted ball boxes at tile boundaries.
    """
    if not tiled_detections:
        return []

    fh, fw = int(frame_shape[0]), int(frame_shape[1])

    # Clamp all boxes to valid image bounds and annotate seam-clipping status
    by_class: Dict[str, List[Tuple[DetectionRecord, Optional[TileSpec], bool]]] = {}
    for det, tile in tiled_detections:
        x1 = max(0, min(fw - 1, int(det.bbox[0])))
        y1 = max(0, min(fh - 1, int(det.bbox[1])))
        x2 = max(x1 + 1, min(fw, int(det.bbox[2])))
        y2 = max(y1 + 1, min(fh, int(det.bbox[3])))
        clamped = DetectionRecord(
            frame_number=det.frame_number,
            timestamp=det.timestamp,
            class_id=det.class_id,
            class_name=det.class_name,
            bbox=(x1, y1, x2, y2),
            confidence=det.confidence,
            tracking_id=det.tracking_id,
        )
        clipped = _is_clipped_at_internal_seam(clamped.bbox, tile, margin_px=seam_margin_px) if tile is not None else False
        by_class.setdefault(clamped.class_name, []).append((clamped, tile, clipped))

    merged: List[DetectionRecord] = []

    for cls_name, items in by_class.items():
        # Stage 1: Drop internal-seam-truncated partial boxes when covered by an unclipped box
        unclipped = [(d, t) for d, t, c in items if not c]
        survivors: List[Tuple[DetectionRecord, Optional[TileSpec], bool]] = []
        for det, tile, clipped in items:
            if clipped and unclipped:
                covered_by_full = False
                for full_det, full_tile in unclipped:
                    if tile is not None and full_tile is not None and tile.tile_index == full_tile.tile_index:
                        continue
                    if compute_iom(det.bbox, full_det.bbox) >= 0.55 or compute_iou(det.bbox, full_det.bbox) >= 0.35:
                        covered_by_full = True
                        break
                if covered_by_full:
                    continue
            survivors.append((det, tile, clipped))

        # Stage 2: Class-aware NMS + containment deduplication
        # Slight priority advantage to unclipped boxes so full players win over clipped fragments
        survivors.sort(
            key=lambda item: float(item[0].confidence) - (0.08 if item[2] else 0.0),
            reverse=True,
        )

        kept: List[Tuple[DetectionRecord, Optional[TileSpec]]] = []
        iou_thr = ball_nms_iou if cls_name == "ball" else player_nms_iou

        for det, tile, _ in survivors:
            suppress = False
            dx1, dy1, dx2, dy2 = det.bbox
            d_area = max(1, (dx2 - dx1) * (dy2 - dy1))
            dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
            d_max_dim = max(dx2 - dx1, dy2 - dy1)

            for k_det, k_tile in kept:
                ov = compute_iou(det.bbox, k_det.bbox)
                if ov >= iou_thr:
                    suppress = True
                    break

                if cls_name == "ball":
                    kx1, ky1, kx2, ky2 = k_det.bbox
                    kcx, kcy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
                    k_max_dim = max(kx2 - kx1, ky2 - ky1)
                    if math.hypot(dcx - kcx, dcy - kcy) <= 0.65 * max(d_max_dim, k_max_dim, 8):
                        suppress = True
                        break
                else:
                    # Only apply containment suppression across DIFFERENT tiles (or nearly identical boxes)
                    # so true overlapping players detected inside the same tile are preserved.
                    different_tile = (
                        tile is not None and k_tile is not None and tile.tile_index != k_tile.tile_index
                    )
                    if different_tile:
                        iom = compute_iom(det.bbox, k_det.bbox)
                        if iom >= player_containment_iom:
                            kx1, ky1, kx2, ky2 = k_det.bbox
                            k_area = max(1, (kx2 - kx1) * (ky2 - ky1))
                            area_ratio = min(d_area, k_area) / float(max(d_area, k_area))
                            if area_ratio >= 0.25:
                                suppress = True
                                break

            if not suppress:
                kept.append((det, tile))

        merged.extend(d for d, _ in kept)

    return merged

