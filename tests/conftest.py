"""Synthetic test fixtures for the Football AI Engineering test suite."""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_synthetic_kit_crop(
    primary_bgr: tuple[int, int, int],
    secondary_bgr: tuple[int, int, int] | None = None,
    striped: bool = False,
    grass_border_px: int = 4,
    height: int = 64,
    width: int = 48,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a realistic textured synthetic jersey crop (sharp enough to pass Laplacian blur checks)
    with optional vertical stripes and pitch-green border pixels.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array(primary_bgr, dtype=np.uint8)

    if secondary_bgr is not None:
        if striped:
            for x in range(0, width, 8):
                canvas[:, x : min(width, x + 4)] = np.array(secondary_bgr, dtype=np.uint8)
        else:
            # Upper chest vs lower torso accent
            canvas[int(height * 0.65) :, :] = np.array(secondary_bgr, dtype=np.uint8)

    # Add pitch-green border on left/right edges to simulate background intrusion
    if grass_border_px > 0:
        grass_bgr = np.array([40, 160, 50], dtype=np.uint8)
        canvas[:, :grass_border_px] = grass_bgr
        canvas[:, width - grass_border_px :] = grass_bgr

    # Draw a crisp number/badge texture in the center so Laplacian variance is high
    cv2.rectangle(canvas, (width // 2 - 5, height // 2 - 8), (width // 2 + 5, height // 2 + 8), (240, 240, 240), 1)
    noise = rng.integers(-8, 9, size=canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


@pytest.fixture
def synthetic_crops_by_class() -> dict[str, list[np.ndarray]]:
    """
    Return synthetic jersey crops for Team_A (Red/Blue), Team_B (White/Red),
    Referee (Yellow/Black), and Unknown (Green pitch / dark blur).
    """
    team_a = [
        make_synthetic_kit_crop((30, 25, 210), (140, 30, 20), striped=False, seed=10 + i)
        for i in range(10)
    ]
    team_b = [
        make_synthetic_kit_crop((235, 235, 235), (30, 30, 200), striped=True, seed=30 + i)
        for i in range(10)
    ]
    referee = [
        make_synthetic_kit_crop((15, 230, 240), (25, 25, 25), striped=False, seed=50 + i)
        for i in range(8)
    ]
    unknown = [
        make_synthetic_kit_crop((180, 40, 180), (40, 180, 180), striped=False, seed=70 + i)
        for i in range(4)
    ]
    return {
        "Team_A": team_a,
        "Team_B": team_b,
        "Referee": referee,
        "Unknown": unknown,
    }

