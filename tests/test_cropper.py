"""Tests for jersey cropping, blur rejection, grass contamination masking, and occlusion filtering."""

from __future__ import annotations

import cv2
import numpy as np

from src.classification.cropper import (
    compute_blur_score,
    compute_grass_mask,
    compute_occlusion_ratio,
    extract_torso_crop,
)
from src.config import CroppingConfig
from tests.conftest import make_synthetic_kit_crop


def test_extract_torso_crop_valid() -> None:
    frame = np.full((300, 400, 3), (40, 160, 50), dtype=np.uint8)
    kit = make_synthetic_kit_crop((20, 20, 220), grass_border_px=0, height=100, width=60)
    frame[80:180, 120:180] = kit

    cfg = CroppingConfig(min_player_height=30, min_blur_variance=5.0, max_grass_ratio=0.80)
    crop, torso_bbox, metrics = extract_torso_crop(frame, (120, 80, 180, 180), config=cfg)

    assert crop is not None
    assert metrics.is_valid
    assert metrics.rejection_reason is None
    assert torso_bbox[2] > torso_bbox[0]
    assert torso_bbox[3] > torso_bbox[1]
    assert metrics.grass_ratio < 0.25


def test_extract_torso_crop_rejects_small_and_invalid_boxes() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cfg = CroppingConfig(min_player_height=35)

    crop_small, _, metrics_small = extract_torso_crop(frame, (10, 10, 25, 30), config=cfg)
    assert crop_small is None
    assert not metrics_small.is_valid
    assert metrics_small.rejection_reason == "player_bbox_too_small"

    crop_inv, _, metrics_inv = extract_torso_crop(frame, (100, 100, 50, 50), config=cfg)
    assert crop_inv is None
    assert not metrics_inv.is_valid


def test_extract_torso_crop_rejects_blurry_and_grass_dominated() -> None:
    # Uniform flat red image has near-zero Laplacian variance (blurry/featureless)
    frame_blur = np.full((200, 200, 3), (10, 10, 200), dtype=np.uint8)
    cfg = CroppingConfig(min_blur_variance=15.0)
    crop_b, _, metrics_b = extract_torso_crop(frame_blur, (40, 40, 120, 160), config=cfg)
    assert crop_b is None
    assert not metrics_b.is_valid
    assert "too_blurry" in (metrics_b.rejection_reason or "")

    # Textured pitch grass image has high blur variance but >80% green grass pixels
    rng = np.random.default_rng(99)
    frame_grass = np.full((200, 200, 3), (40, 170, 50), dtype=np.uint8)
    noise = rng.integers(-25, 25, size=frame_grass.shape, dtype=np.int16)
    frame_grass = np.clip(frame_grass.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cfg_grass = CroppingConfig(min_blur_variance=2.0, max_grass_ratio=0.70)
    crop_g, _, metrics_g = extract_torso_crop(frame_grass, (40, 40, 120, 160), config=cfg_grass)
    assert crop_g is None
    assert not metrics_g.is_valid
    assert "excessive_grass" in (metrics_g.rejection_reason or "")


def test_occlusion_rejection() -> None:
    frame = np.full((300, 400, 3), (40, 160, 50), dtype=np.uint8)
    kit = make_synthetic_kit_crop((220, 30, 30), grass_border_px=0, height=100, width=60)
    frame[80:180, 120:180] = kit

    target_box = (120, 80, 180, 180)
    overlapping_other = (122, 82, 178, 178)  # Nearly 100% overlap on torso

    cfg = CroppingConfig(max_occlusion_ratio=0.50, min_blur_variance=2.0)
    crop, _, metrics = extract_torso_crop(
        frame, target_box, config=cfg, other_player_bboxes=[target_box, overlapping_other]
    )
    assert crop is None
    assert not metrics.is_valid
    assert metrics.occlusion_ratio > 0.80
    assert "heavily_occluded" in (metrics.rejection_reason or "")

