"""Explicit, reproducible fixed-camera pitch calibration using planar homography."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchCalibration:
    """Maps image pixels to a football-pitch coordinate system in metres.

    ``image_points_px`` and ``pitch_points_m`` must describe the same physical
    landmarks in the same order.  Calibration is valid only while the camera is
    fixed; callers must not silently reuse it after a move, zoom, or crop.
    """

    image_points_px: Tuple[Tuple[float, float], ...]
    pitch_points_m: Tuple[Tuple[float, float], ...]
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    def __post_init__(self) -> None:
        if len(self.image_points_px) < 4 or len(self.image_points_px) != len(self.pitch_points_m):
            raise ValueError("Pitch calibration requires at least four matching image and pitch landmarks.")
        if self.pitch_length_m <= 0 or self.pitch_width_m <= 0:
            raise ValueError("Pitch dimensions must be positive.")

    @property
    def homography(self) -> np.ndarray:
        matrix, mask = cv2.findHomography(
            np.asarray(self.image_points_px, dtype=np.float32),
            np.asarray(self.pitch_points_m, dtype=np.float32),
            method=0,
        )
        if matrix is None or mask is None:
            raise ValueError("Could not compute a homography from supplied pitch landmarks.")
        return matrix

    def project_pixel(self, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
        projected = cv2.perspectiveTransform(
            np.asarray([[[x_px, y_px]]], dtype=np.float32), self.homography
        )[0, 0]
        x_m, y_m = float(projected[0]), float(projected[1])
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            return None
        # A small tolerance permits boundary foot points while rejecting clearly
        # invalid projections caused by a wrong camera calibration.
        if x_m < -2 or y_m < -2 or x_m > self.pitch_length_m + 2 or y_m > self.pitch_width_m + 2:
            return None
        return round(x_m, 3), round(y_m, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_points_px": [list(point) for point in self.image_points_px],
            "pitch_points_m": [list(point) for point in self.pitch_points_m],
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
        }


def load_pitch_calibration(path: str | Path) -> PitchCalibration:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    try:
        image_points = tuple(tuple(map(float, point)) for point in raw["image_points_px"])
        pitch_points = tuple(tuple(map(float, point)) for point in raw["pitch_points_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pitch calibration file: {source}") from exc
    return PitchCalibration(
        image_points_px=image_points,
        pitch_points_m=pitch_points,
        pitch_length_m=float(raw.get("pitch_length_m", 105.0)),
        pitch_width_m=float(raw.get("pitch_width_m", 68.0)),
    )
