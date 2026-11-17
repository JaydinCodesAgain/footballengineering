"""Tests for fixed-camera image-to-pitch projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.geometry.pitch import PitchCalibration, load_pitch_calibration


def test_projects_known_rectangular_pitch_landmarks() -> None:
    calibration = PitchCalibration(
        image_points_px=((100, 50), (1100, 50), (1100, 650), (100, 650)),
        pitch_points_m=((0, 0), (105, 0), (105, 68), (0, 68)),
    )
    assert calibration.project_pixel(600, 350) == pytest.approx((52.5, 34.0), abs=0.01)
    assert calibration.project_pixel(1500, 900) is None


def test_loads_calibration_and_rejects_insufficient_landmarks(tmp_path: Path) -> None:
    path = tmp_path / "pitch.json"
    path.write_text(json.dumps({"image_points_px": [[0, 0], [100, 0], [100, 60], [0, 60]], "pitch_points_m": [[0, 0], [105, 0], [105, 68], [0, 68]]}), encoding="utf-8")
    assert load_pitch_calibration(path).pitch_length_m == 105.0
    with pytest.raises(ValueError, match="at least four"):
        PitchCalibration(((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 0), (1, 1)))
