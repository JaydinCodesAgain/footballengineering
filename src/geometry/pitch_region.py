"""Geometric fixed-camera pitch-region filtering for raw fisheye and planar/undistorted cameras.

Derives the playable pitch polygon and configurable boundary margins directly from
camera calibration parameters (planar homography landmarks, fisheye intrinsics/extrinsics,
or pitch-line keypoints)—never from hard-coded match pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.geometry.pitch import PitchCalibration, load_pitch_calibration
from src.schemas import DetectionRecord


class PitchZone(str, Enum):
    PLAYABLE = "playable"
    BOUNDARY_MARGIN = "boundary_margin"
    OFF_PITCH = "off_pitch"


@dataclass(frozen=True)
class PitchRegionConfig:
    """Configurable margins (in physical metres) around the 105m x 68m playable pitch."""

    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    player_margin_m: float = 2.0
    ball_margin_m: float = 2.5
    aerial_ball_top_margin_px: int = 120
    tiling_top_headroom_px: int = 110
    tiling_side_pad_px: int = 40
    tiling_bottom_pad_px: int = 30


class PitchRegionFilter:
    """Geometrically separates playable pitch, boundary margin, and off-pitch regions.

    Supports three calibration formats:
    1. Planar `PitchCalibration` JSON (`image_points_px` + `pitch_points_m`).
    2. Fisheye calibration (`camera_intrinsics.npz` with `K, D, rvecs, tvecs`, optionally
       paired with `keypoints.json` perimeter landmarks and `homography.npy`).
    3. SoccerTrack-style `keypoints.json` alone (ordered perimeter pitch landmarks).

    Respects `coordinate_space` (`"raw_fisheye"` vs `"undistorted"` / `"planar"`).
    """

    def __init__(
        self,
        playable_polygon_px: np.ndarray,
        player_margin_polygon_px: np.ndarray,
        ball_margin_polygon_px: np.ndarray,
        coordinate_space: str = "planar",
        config: Optional[PitchRegionConfig] = None,
        pixel_to_pitch_fn: Optional[Any] = None,
        pitch_to_pixel_fn: Optional[Any] = None,
        source_description: str = "geometric_calibration",
    ) -> None:
        self.config = config or PitchRegionConfig()
        self.coordinate_space = coordinate_space
        self.source_description = source_description
        self._pixel_to_pitch_fn = pixel_to_pitch_fn
        self._pitch_to_pixel_fn = pitch_to_pixel_fn

        self.playable_polygon_px = np.asarray(playable_polygon_px, dtype=np.float32).reshape(-1, 2)
        self.player_margin_polygon_px = np.asarray(player_margin_polygon_px, dtype=np.float32).reshape(-1, 2)
        self.ball_margin_polygon_px = np.asarray(ball_margin_polygon_px, dtype=np.float32).reshape(-1, 2)

        if len(self.playable_polygon_px) < 4:
            raise ValueError("Playable pitch polygon requires at least 4 vertices.")

        self._playable_contour = self.playable_polygon_px.reshape(-1, 1, 2)
        self._player_contour = self.player_margin_polygon_px.reshape(-1, 1, 2)
        self._ball_contour = self.ball_margin_polygon_px.reshape(-1, 1, 2)

    @staticmethod
    def _sample_pitch_perimeter(
        length_m: float,
        width_m: float,
        margin_m: float = 0.0,
        steps_per_side: int = 24,
    ) -> np.ndarray:
        """Sample dense points along the rectangular pitch perimeter expanded by `margin_m`."""
        x0, x1 = -margin_m, length_m + margin_m
        y0, y1 = -margin_m, width_m + margin_m
        top = np.column_stack([np.linspace(x0, x1, steps_per_side, endpoint=False), np.full(steps_per_side, y0)])
        right = np.column_stack([np.full(steps_per_side, x1), np.linspace(y0, y1, steps_per_side, endpoint=False)])
        bottom = np.column_stack([np.linspace(x1, x0, steps_per_side, endpoint=False), np.full(steps_per_side, y1)])
        left = np.column_stack([np.full(steps_per_side, x0), np.linspace(y1, y0, steps_per_side, endpoint=False)])
        return np.vstack([top, right, bottom, left]).astype(np.float64)

    @staticmethod
    def _extract_ordered_perimeter_from_keypoints(
        keypoints: Dict[str, Sequence[float]],
        length_m: float,
        width_m: float,
        tol_m: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract ordered `(world_xy, image_uv)` perimeter points directly from pitch-line landmarks."""
        top, right, bottom, left = [], [], [], []
        for key, uv in keypoints.items():
            try:
                xs, ys = key.strip("() ").split(",")
                wx, wy = float(xs), float(ys)
                u, v = float(uv[0]), float(uv[1])
            except (ValueError, TypeError, IndexError):
                continue
            if abs(wy - 0.0) <= tol_m:
                top.append((wx, wy, u, v))
            elif abs(wx - length_m) <= tol_m:
                right.append((wx, wy, u, v))
            elif abs(wy - width_m) <= tol_m:
                bottom.append((wx, wy, u, v))
            elif abs(wx - 0.0) <= tol_m:
                left.append((wx, wy, u, v))

        top.sort(key=lambda t: t[0])
        right.sort(key=lambda t: t[1])
        bottom.sort(key=lambda t: -t[0])
        left.sort(key=lambda t: -t[1])

        ordered = top + right + bottom + left
        # Deduplicate adjacent corners
        dedup: List[Tuple[float, float, float, float]] = []
        for item in ordered:
            if not dedup or abs(item[0] - dedup[-1][0]) > 1e-3 or abs(item[1] - dedup[-1][1]) > 1e-3:
                dedup.append(item)
        if len(dedup) < 8:
            return None
        arr = np.asarray(dedup, dtype=np.float64)
        return arr[:, :2], arr[:, 2:4]

    @staticmethod
    def _expand_polygon_by_metric_margin(
        world_pts: np.ndarray,
        image_pts: np.ndarray,
        pitch_to_pixel_fn: Any,
        length_m: float,
        width_m: float,
        margin_m: float,
    ) -> np.ndarray:
        """Expand an anchored curved perimeter polygon outward by `margin_m` metres using the
        local geometric displacement predicted by `pitch_to_pixel_fn`.
        """
        if margin_m <= 0.0:
            return image_pts.copy()
        expanded_world = world_pts.copy()
        for i, (wx, wy) in enumerate(world_pts):
            dx = -margin_m if abs(wx) <= 0.3 else (margin_m if abs(wx - length_m) <= 0.3 else 0.0)
            dy = -margin_m if abs(wy) <= 0.3 else (margin_m if abs(wy - width_m) <= 0.3 else 0.0)
            expanded_world[i, 0] = wx + dx
            expanded_world[i, 1] = wy + dy

        base_proj = pitch_to_pixel_fn(world_pts)
        exp_proj = pitch_to_pixel_fn(expanded_world)
        delta_uv = exp_proj - base_proj
        return image_pts + delta_uv

    @classmethod
    def from_planar_calibration(
        cls,
        calibration: PitchCalibration,
        config: Optional[PitchRegionConfig] = None,
        coordinate_space: str = "planar",
    ) -> "PitchRegionFilter":
        cfg = config or PitchRegionConfig(
            pitch_length_m=calibration.pitch_length_m,
            pitch_width_m=calibration.pitch_width_m,
        )
        H_img_to_pitch = calibration.homography.astype(np.float64)
        H_pitch_to_img = np.linalg.inv(H_img_to_pitch)

        def pitch_to_pixel(pts_m: np.ndarray) -> np.ndarray:
            arr = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(arr, H_pitch_to_img)
            return out.reshape(-1, 2)

        def pixel_to_pitch(pts_uv: np.ndarray) -> np.ndarray:
            arr = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(arr, H_img_to_pitch)
            return out.reshape(-1, 2)

        poly_play = pitch_to_pixel(cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, 0.0))
        poly_player = pitch_to_pixel(
            cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.player_margin_m)
        )
        poly_ball = pitch_to_pixel(
            cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.ball_margin_m)
        )
        return cls(
            playable_polygon_px=poly_play,
            player_margin_polygon_px=poly_player,
            ball_margin_polygon_px=poly_ball,
            coordinate_space=coordinate_space,
            config=cfg,
            pixel_to_pitch_fn=pixel_to_pitch,
            pitch_to_pixel_fn=pitch_to_pixel,
            source_description="planar_homography_calibration",
        )

    @classmethod
    def from_fisheye_calibration(
        cls,
        intrinsics_path: str | Path,
        keypoints_path: Optional[str | Path] = None,
        homography_path: Optional[str | Path] = None,
        coordinate_space: str = "raw_fisheye",
        config: Optional[PitchRegionConfig] = None,
    ) -> "PitchRegionFilter":
        """Build a fixed-camera pitch filter from fisheye intrinsics/extrinsics and optional keypoints.

        When `coordinate_space == "raw_fisheye"`, uses `cv2.fisheye.projectPoints` (and anchors
        to dense perimeter keypoints when supplied so stitched multi-camera panoramas remain
        pixel-accurate along the touchlines).
        When `coordinate_space == "undistorted"`, uses `homography_path` (pitch -> undistorted px).
        """
        cfg = config or PitchRegionConfig()
        intr = np.load(str(intrinsics_path), allow_pickle=True)
        K = intr["K"].astype(np.float64)
        D = intr["D"].astype(np.float64)
        rvec = np.array(intr["rvecs"][0], dtype=np.float64).reshape(3, 1)
        tvec = np.array(intr["tvecs"][0], dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        R_T = R.T
        cam_pos = -(R_T @ tvec).reshape(3)

        if coordinate_space == "undistorted" and homography_path and Path(homography_path).is_file():
            H_p2u = np.load(str(homography_path)).astype(np.float64)
            H_u2p = np.linalg.inv(H_p2u)

            def pitch_to_pixel(pts_m: np.ndarray) -> np.ndarray:
                arr = np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2)
                return cv2.perspectiveTransform(arr, H_p2u).reshape(-1, 2)

            def pixel_to_pitch(pts_uv: np.ndarray) -> np.ndarray:
                arr = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 1, 2)
                return cv2.perspectiveTransform(arr, H_u2p).reshape(-1, 2)

            poly_play = pitch_to_pixel(cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, 0.0))
            poly_player = pitch_to_pixel(
                cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.player_margin_m)
            )
            poly_ball = pitch_to_pixel(
                cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.ball_margin_m)
            )
            return cls(
                playable_polygon_px=poly_play,
                player_margin_polygon_px=poly_player,
                ball_margin_polygon_px=poly_ball,
                coordinate_space="undistorted",
                config=cfg,
                pixel_to_pitch_fn=pixel_to_pitch,
                pitch_to_pixel_fn=pitch_to_pixel,
                source_description=f"undistorted_homography:{Path(homography_path).name}",
            )

        # Raw fisheye pixel space
        def pitch_to_pixel_fisheye(pts_m: np.ndarray) -> np.ndarray:
            arr = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
            world = np.column_stack([arr[:, 0], arr[:, 1], np.zeros(len(arr))]).reshape(-1, 1, 3)
            uv, _ = cv2.fisheye.projectPoints(world, rvec, tvec, K, D)
            return uv.reshape(-1, 2)

        def pixel_to_pitch_fisheye(pts_uv: np.ndarray) -> np.ndarray:
            arr = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 1, 2)
            und = cv2.fisheye.undistortPoints(arr, K, D).reshape(-1, 2)
            rays_w = (R_T @ np.hstack([und, np.ones((len(und), 1))]).T).T
            s = -cam_pos[2] / np.where(np.abs(rays_w[:, 2]) > 1e-9, rays_w[:, 2], 1e-9)
            w = cam_pos[None, :] + s[:, None] * rays_w
            return np.column_stack([w[:, 0], w[:, 1]])

        anchored = None
        if keypoints_path and Path(keypoints_path).is_file():
            kp = json.loads(Path(keypoints_path).read_text(encoding="utf-8"))
            if isinstance(kp, dict):
                anchored = cls._extract_ordered_perimeter_from_keypoints(
                    kp, cfg.pitch_length_m, cfg.pitch_width_m
                )

        if anchored is not None:
            world_perim, img_perim = anchored
            poly_play = img_perim
            poly_player = cls._expand_polygon_by_metric_margin(
                world_perim, img_perim, pitch_to_pixel_fisheye,
                cfg.pitch_length_m, cfg.pitch_width_m, cfg.player_margin_m,
            )
            poly_ball = cls._expand_polygon_by_metric_margin(
                world_perim, img_perim, pitch_to_pixel_fisheye,
                cfg.pitch_length_m, cfg.pitch_width_m, cfg.ball_margin_m,
            )
            desc = f"fisheye_plus_keypoints:{Path(intrinsics_path).name}"
        else:
            poly_play = pitch_to_pixel_fisheye(cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, 0.0))
            poly_player = pitch_to_pixel_fisheye(
                cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.player_margin_m)
            )
            poly_ball = pitch_to_pixel_fisheye(
                cls._sample_pitch_perimeter(cfg.pitch_length_m, cfg.pitch_width_m, cfg.ball_margin_m)
            )
            desc = f"fisheye_intrinsics:{Path(intrinsics_path).name}"

        return cls(
            playable_polygon_px=poly_play,
            player_margin_polygon_px=poly_player,
            ball_margin_polygon_px=poly_ball,
            coordinate_space="raw_fisheye",
            config=cfg,
            pixel_to_pitch_fn=pixel_to_pitch_fisheye,
            pitch_to_pixel_fn=pitch_to_pixel_fisheye,
            source_description=desc,
        )

    @classmethod
    def from_calibration_path(
        cls,
        path: str | Path,
        coordinate_space: str = "raw_fisheye",
        config: Optional[PitchRegionConfig] = None,
    ) -> "PitchRegionFilter":
        """Auto-load from either a planar `PitchCalibration` JSON, a fisheye `.npz`,
        a SoccerTrack `keypoints.json`, or a calibration directory containing them.
        """
        p = Path(path)
        if p.is_dir():
            intr_files = sorted(p.glob("*_camera_intrinsics.npz"))
            kp_files = sorted(p.glob("*_keypoints.json"))
            h_files = sorted(p.glob("*_homography.npy"))
            if intr_files:
                return cls.from_fisheye_calibration(
                    intrinsics_path=intr_files[0],
                    keypoints_path=kp_files[0] if kp_files else None,
                    homography_path=h_files[0] if h_files else None,
                    coordinate_space=coordinate_space,
                    config=config,
                )
            raise FileNotFoundError(f"No calibration files found in directory: {p}")

        if p.suffix.lower() == ".npz":
            parent = p.parent
            stem_prefix = p.name.replace("_camera_intrinsics.npz", "")
            kp = parent / f"{stem_prefix}_keypoints.json"
            hom = parent / f"{stem_prefix}_homography.npy"
            return cls.from_fisheye_calibration(
                intrinsics_path=p,
                keypoints_path=kp if kp.is_file() else None,
                homography_path=hom if hom.is_file() else None,
                coordinate_space=coordinate_space,
                config=config,
            )

        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "image_points_px" in raw and "pitch_points_m" in raw:
            return cls.from_planar_calibration(load_pitch_calibration(p), config=config, coordinate_space="planar")

        if isinstance(raw, dict):
            # SoccerTrack keypoints.json — check for sibling camera_intrinsics.npz first
            parent = p.parent
            stem_prefix = p.name.replace("_keypoints.json", "")
            intr = parent / f"{stem_prefix}_camera_intrinsics.npz"
            hom = parent / f"{stem_prefix}_homography.npy"
            if intr.is_file():
                return cls.from_fisheye_calibration(
                    intrinsics_path=intr,
                    keypoints_path=p,
                    homography_path=hom if hom.is_file() else None,
                    coordinate_space=coordinate_space,
                    config=config,
                )
            cfg = config or PitchRegionConfig()
            anchored = cls._extract_ordered_perimeter_from_keypoints(raw, cfg.pitch_length_m, cfg.pitch_width_m)
            if anchored is None:
                raise ValueError(f"Could not extract pitch perimeter landmarks from {p}")
            world_perim, img_perim = anchored
            # Fit fallback homography to estimate metric outward margin
            H_p2i, _ = cv2.findHomography(world_perim.astype(np.float32), img_perim.astype(np.float32), 0)
            if H_p2i is None:
                raise ValueError(f"Homography fit failed on keypoints in {p}")

            def _p2i(pts_m: np.ndarray) -> np.ndarray:
                return cv2.perspectiveTransform(np.asarray(pts_m, dtype=np.float64).reshape(-1, 1, 2), H_p2i).reshape(-1, 2)

            poly_player = cls._expand_polygon_by_metric_margin(
                world_perim, img_perim, _p2i, cfg.pitch_length_m, cfg.pitch_width_m, cfg.player_margin_m
            )
            poly_ball = cls._expand_polygon_by_metric_margin(
                world_perim, img_perim, _p2i, cfg.pitch_length_m, cfg.pitch_width_m, cfg.ball_margin_m
            )
            return cls(
                playable_polygon_px=img_perim,
                player_margin_polygon_px=poly_player,
                ball_margin_polygon_px=poly_ball,
                coordinate_space=coordinate_space,
                config=cfg,
                pitch_to_pixel_fn=_p2i,
                source_description=f"keypoints_only:{p.name}",
            )

        raise ValueError(f"Unsupported calibration file format: {p}")

    def classify_point(self, x_px: float, y_px: float, for_ball: bool = False) -> PitchZone:
        """Classify a pixel point `(x_px, y_px)` into PLAYABLE, BOUNDARY_MARGIN, or OFF_PITCH."""
        pt = (float(x_px), float(y_px))
        if cv2.pointPolygonTest(self._playable_contour, pt, False) >= 0:
            return PitchZone.PLAYABLE
        contour = self._ball_contour if for_ball else self._player_contour
        if cv2.pointPolygonTest(contour, pt, False) >= 0:
            return PitchZone.BOUNDARY_MARGIN
        return PitchZone.OFF_PITCH

    def is_player_in_region(self, bbox: Tuple[int, int, int, int], include_margin: bool = True) -> bool:
        """Check whether a player's foot point `(bottom-centre)` lies inside the pitch region."""
        fx = (bbox[0] + bbox[2]) / 2.0
        fy = float(bbox[3])
        zone = self.classify_point(fx, fy, for_ball=False)
        if zone == PitchZone.PLAYABLE:
            return True
        return include_margin and zone == PitchZone.BOUNDARY_MARGIN

    def is_ball_in_region(
        self,
        bbox: Tuple[int, int, int, int],
        include_margin: bool = True,
        allow_aerial_above_far_line: bool = True,
        near_active_trajectory: bool = False,
    ) -> bool:
        """Check whether a ball candidate lies inside the playable pitch (+ configurable margin),
        with optional headroom strictly above the far touchline for lofted aerial passes.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        zone = self.classify_point(cx, cy, for_ball=True)
        if zone == PitchZone.PLAYABLE:
            return True
        if include_margin and zone == PitchZone.BOUNDARY_MARGIN:
            return True
        if allow_aerial_above_far_line and near_active_trajectory:
            min_x = float(np.min(self.ball_margin_polygon_px[:, 0]))
            max_x = float(np.max(self.ball_margin_polygon_px[:, 0]))
            min_y = float(np.min(self.playable_polygon_px[:, 1]))
            if min_x <= cx <= max_x and (min_y - self.config.aerial_ball_top_margin_px) <= cy <= min_y:
                return True
        return False

    def filter_detections(
        self,
        detections: Sequence[DetectionRecord],
        include_margin: bool = True,
        active_ball_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[DetectionRecord], Dict[str, int]]:
        """Filter players and balls using the geometric pitch region and return audit counts."""
        kept: List[DetectionRecord] = []
        stats = {
            "players_kept": 0,
            "players_rejected_off_pitch": 0,
            "balls_kept": 0,
            "balls_rejected_off_pitch": 0,
        }
        for d in detections:
            if d.class_name == "player":
                if self.is_player_in_region(d.bbox, include_margin=include_margin):
                    kept.append(d)
                    stats["players_kept"] += 1
                else:
                    stats["players_rejected_off_pitch"] += 1
            elif d.class_name == "ball":
                bcx = (d.bbox[0] + d.bbox[2]) / 2.0
                bcy = (d.bbox[1] + d.bbox[3]) / 2.0
                near_traj = (
                    active_ball_pos is not None
                    and float(np.hypot(bcx - active_ball_pos[0], bcy - active_ball_pos[1])) <= 140.0
                )
                if self.is_ball_in_region(d.bbox, include_margin=include_margin, near_active_trajectory=near_traj):
                    kept.append(d)
                    stats["balls_kept"] += 1
                else:
                    stats["balls_rejected_off_pitch"] += 1
            else:
                kept.append(d)
        return kept, stats

    def get_tiling_roi_bbox(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Return a tight pixel bounding box `(x1, y1, x2, y2)` enclosing the pitch + margin + headroom
        so multi-tile inference avoids wasting compute on sky, benches, and black panoramic letterbox bars.
        """
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        pts = np.vstack([self.player_margin_polygon_px, self.ball_margin_polygon_px])
        x1 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - int(self.config.tiling_side_pad_px))
        y1 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - int(self.config.tiling_top_headroom_px))
        x2 = min(fw, int(np.ceil(float(np.max(pts[:, 0])))) + int(self.config.tiling_side_pad_px))
        y2 = min(fh, int(np.ceil(float(np.max(pts[:, 1])))) + int(self.config.tiling_bottom_pad_px))
        return (x1, y1, x2, y2)

    def build_binary_mask(self, frame_shape: Tuple[int, int], use_ball_margin: bool = True) -> np.ndarray:
        """Render a uint8 binary mask (255 inside pitch region, 0 outside) for fast motion masking."""
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        mask = np.zeros((fh, fw), dtype=np.uint8)
        poly = self.ball_margin_polygon_px if use_ball_margin else self.playable_polygon_px
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
        return mask

    def project_pixel_to_pitch_m(self, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
        """Project a pixel point to metric pitch coordinates `(x_m in [0, L], y_m in [0, W])`."""
        if self._pixel_to_pitch_fn is None:
            return None
        res = self._pixel_to_pitch_fn(np.array([[float(x_px), float(y_px)]], dtype=np.float64))[0]
        xm, ym = float(res[0]), float(res[1])
        if not (np.isfinite(xm) and np.isfinite(ym)):
            return None
        return (xm, ym)

