"""Unit and integration tests for Sprint 3.2:
1. Native-resolution overlapping multi-tile detection & seam-aware merging (`src/detection/tiling.py`).
2. Geometric fixed-camera pitch-region filtering (`src/geometry/pitch_region.py`).
3. Persistent fixed-camera static-background distractor masking (`src/possession/ball_tracker.py`).
4. Motion-assisted small-ball candidate recovery (`src/detection/motion_ball.py`).
5. Direct Ultralytics tracker NumPy adapter and backward-compatible CLI config overrides.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import BallTrackingConfig, DetectionConfig
from src.detection.detector import FootballDetector
from src.detection.motion_ball import FixedCameraMotionBallDetector, MotionBallConfig
from src.detection.tiling import (
    TileSpec,
    compute_iom,
    estimate_active_content_bbox,
    extract_tile_views,
    generate_tile_grid,
    merge_tiled_detections,
)
from src.geometry.pitch import PitchCalibration
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter, PitchZone
from src.possession.ball_tracker import KalmanBallTracker, filter_ball_candidates
from src.schemas import DetectionRecord
from src.tracking.tracker import _NumpyDetections


def _build_synthetic_planar_pitch_filter(
    player_margin_m: float = 2.0,
    ball_margin_m: float = 2.5,
) -> PitchRegionFilter:
    """Create a 4000x1200 synthetic fixed camera observing a 105x68m rectangular pitch."""
    # Map (0..105, 0..68) linearly into x=[400..3600], y=[200..900] (scale ~30.48 px/m in X, ~10.29 px/m in Y)
    cal = PitchCalibration(
        image_points_px=((400.0, 200.0), (3600.0, 200.0), (3600.0, 900.0), (400.0, 900.0)),
        pitch_points_m=((0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)),
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    cfg = PitchRegionConfig(
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        player_margin_m=player_margin_m,
        ball_margin_m=ball_margin_m,
    )
    return PitchRegionFilter.from_planar_calibration(cal, config=cfg)


# ---------------------------------------------------------------------------
# 1. Native-resolution multi-tile detection & seam-aware merging
# ---------------------------------------------------------------------------


def test_generate_tile_grid_covers_panoramic_frame_with_overlap() -> None:
    tiles = generate_tile_grid(
        frame_shape=(1080, 4096),
        tile_width=1280,
        tile_height=1080,
        overlap=0.22,
    )
    assert len(tiles) == 4
    assert tiles[0].x1 == 0
    assert tiles[-1].x2 == 4096
    assert all(t.y1 == 0 and t.y2 == 1080 for t in tiles)
    for a, b in zip(tiles, tiles[1:]):
        assert a.has_right_neighbor and b.has_left_neighbor
        overlap_px = a.x2 - b.x1
        assert overlap_px >= int(1280 * 0.22) - 1
    assert not tiles[0].has_left_neighbor
    assert not tiles[-1].has_right_neighbor


def test_extract_tile_views_is_zero_copy() -> None:
    frame = np.zeros((1080, 4096, 3), dtype=np.uint8)
    tiles = generate_tile_grid(frame.shape[:2], tile_width=1280, tile_height=1080, overlap=0.20)
    views = extract_tile_views(frame, tiles)
    assert len(views) == len(tiles)
    for view, tile in zip(views, tiles):
        assert view.shape == (tile.height, tile.width, 3)
        assert np.shares_memory(view, frame)


def test_estimate_active_content_bbox_skips_black_letterbox_bars() -> None:
    frame = np.zeros((1906, 3840, 3), dtype=np.uint8)
    frame[300:1600, 100:3740] = 90
    x1, y1, x2, y2 = estimate_active_content_bbox(frame)
    assert y1 > 150 and y1 <= 300
    assert y2 >= 1600 and y2 < 1800
    assert x1 <= 100 and x2 >= 3740


def test_merge_tiled_detections_suppresses_seam_clipped_duplicate_and_preserves_ball() -> None:
    t0 = TileSpec(0, 0, 0, 0, 0, 1280, 1080, False, True, False, False)
    t1 = TileSpec(1, 0, 1, 960, 0, 2240, 1080, True, False, False, False)

    full_player = DetectionRecord(0, 0.0, 0, "player", (1255, 400, 1295, 500), 0.78, None)
    clipped_half = DetectionRecord(0, 0.0, 0, "player", (1256, 402, 1279, 498), 0.80, None)
    ball = DetectionRecord(0, 0.0, 32, "ball", (1270, 488, 1282, 500), 0.32, None)
    ball_dup = DetectionRecord(0, 0.0, 32, "ball", (1272, 489, 1283, 500), 0.25, None)

    merged = merge_tiled_detections(
        [(clipped_half, t0), (full_player, t1), (ball, t1), (ball_dup, t0)],
        frame_shape=(1080, 2240),
    )
    players = [d for d in merged if d.class_name == "player"]
    balls = [d for d in merged if d.class_name == "ball"]

    assert len(players) == 1
    assert players[0].bbox == (1255, 400, 1295, 500)
    assert len(balls) == 1
    assert balls[0].confidence == pytest.approx(0.32)


def test_merge_tiled_detections_keeps_two_overlapping_players_in_same_tile() -> None:
    t0 = TileSpec(0, 0, 0, 0, 0, 1280, 1080, False, False, False, False)
    p1 = DetectionRecord(0, 0.0, 0, "player", (500, 400, 540, 500), 0.85, None)
    p2 = DetectionRecord(0, 0.0, 0, "player", (525, 405, 565, 505), 0.80, None)
    merged = merge_tiled_detections([(p1, t0), (p2, t0)], frame_shape=(1080, 1280))
    assert len(merged) == 2


def test_detector_auto_mode_dispatches_normal_vs_panoramic() -> None:
    det_auto = FootballDetector(DetectionConfig(inference_mode="auto", auto_tile_min_width_px=2400))
    assert not det_auto.should_use_tiled((720, 1280))
    assert not det_auto.should_use_tiled((1080, 1920))
    assert det_auto.should_use_tiled((1080, 4096))
    assert det_auto.should_use_tiled((1906, 3840))

    det_full = FootballDetector(DetectionConfig(inference_mode="full_frame"))
    assert not det_full.should_use_tiled((1080, 4096))

    det_tiled = FootballDetector(DetectionConfig(inference_mode="tiled"))
    assert det_tiled.should_use_tiled((720, 1280))


def test_compute_iom_containment_metric() -> None:
    outer = (100, 100, 200, 200)
    inner = (120, 120, 160, 160)
    assert compute_iom(outer, inner) == pytest.approx(1.0)
    assert compute_iom((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Geometric fixed-camera pitch-region filtering
# ---------------------------------------------------------------------------


def test_pitch_region_filter_classifies_playable_margin_and_off_pitch() -> None:
    prf = _build_synthetic_planar_pitch_filter(player_margin_m=2.0, ball_margin_m=2.5)

    # Centre circle is strictly playable
    assert prf.classify_point(2000.0, 550.0) == PitchZone.PLAYABLE
    # Throw-in taker standing 1.0 m outside near touchline (y = 900 + ~10.3 px) -> BOUNDARY_MARGIN
    assert prf.classify_point(2000.0, 910.0, for_ball=False) == PitchZone.BOUNDARY_MARGIN
    # Sideline bench / spare curb ball 8 m outside near touchline (y = 985) -> OFF_PITCH
    assert prf.classify_point(2000.0, 985.0, for_ball=True) == PitchZone.OFF_PITCH

    # Player & ball filtering with aerial ball headroom above far touchline
    on_pitch_player = DetectionRecord(0, 0.0, 0, "player", (1980, 450, 2020, 550), 0.90, None)
    throw_in_player = DetectionRecord(0, 0.0, 0, "player", (1980, 815, 2020, 912), 0.85, None)
    bench_coach = DetectionRecord(0, 0.0, 0, "player", (1980, 900, 2020, 1005), 0.88, None)
    spare_ball = DetectionRecord(0, 0.0, 32, "ball", (3486, 980, 3498, 992), 0.45, None)
    aerial_ball = DetectionRecord(0, 0.0, 32, "ball", (1995, 145, 2005, 155), 0.35, None)

    kept, stats = prf.filter_detections(
        [on_pitch_player, throw_in_player, bench_coach, spare_ball, aerial_ball],
        active_ball_pos=(2000.0, 190.0),  # active trajectory rising above far touchline
    )
    assert stats["players_kept"] == 2
    assert stats["players_rejected_off_pitch"] == 1
    assert stats["balls_kept"] == 1  # aerial ball kept via active trajectory headroom
    assert stats["balls_rejected_off_pitch"] == 1  # sideline spare ball rejected
    assert bench_coach not in kept
    assert spare_ball not in kept
    assert aerial_ball in kept

    # Without active trajectory, an isolated false detection above the far touchline is rejected
    assert not prf.is_ball_in_region(aerial_ball.bbox, near_active_trajectory=False)

    # Tiling ROI encloses playable + margin + headroom cleanly inside frame bounds
    roi = prf.get_tiling_roi_bbox((1200, 4000))
    assert roi[0] < 400 and roi[2] > 3600
    assert roi[1] < 200 and roi[3] > 925 and roi[3] <= 1200

    # Pixel-to-pitch round trip
    pm = prf.project_pixel_to_pitch_m(2000.0, 550.0)
    assert pm is not None
    assert pm[0] == pytest.approx(52.5, abs=0.2)
    assert pm[1] == pytest.approx(34.0, abs=0.2)

    # filter_ball_candidates rejects isolated spare balls sitting in BOUNDARY_MARGIN,
    # but keeps legitimate throw-in balls near a player in BOUNDARY_MARGIN.
    margin_spare = DetectionRecord(0, 0.0, 32, "ball", (3488, 908, 3498, 918), 0.55, None)
    kept_iso = filter_ball_candidates(
        [margin_spare], [on_pitch_player], (1200, 4000), pitch_region_filter=prf
    )
    assert len(kept_iso) == 0
    throw_in_taker = DetectionRecord(0, 0.0, 0, "player", (3475, 845, 3505, 915), 0.88, None)
    kept_throw = filter_ball_candidates(
        [margin_spare], [on_pitch_player, throw_in_taker], (1200, 4000), pitch_region_filter=prf
    )
    assert len(kept_throw) == 1


def test_pitch_region_filter_respects_raw_fisheye_vs_undistorted_spaces(tmp_path: Path) -> None:
    """Verify fisheye .npz + keypoints.json + homography.npy loader handles both coordinate spaces."""
    # Construct a valid synthetic elevated camera looking down at a 105x68m pitch (Z=0)
    K = np.array([[1100.0, 0.0, 2000.0], [0.0, 1100.0, 600.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[-0.08], [0.01], [0.0], [0.0]], dtype=np.float64)
    # Camera placed at world (52.5, 110.0, -35.0) looking toward (52.5, 34.0, 0.0)
    rvec = np.array([[0.95], [0.0], [0.0]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = np.array([[52.5], [105.0], [-32.0]], dtype=np.float64)
    tvec = -(R @ cam_pos)

    intr_path = tmp_path / "999999_camera_intrinsics.npz"
    np.savez(intr_path, K=K, D=D, rvecs=np.array([rvec]), tvecs=np.array([tvec]))

    # Generate perimeter keypoints in raw fisheye pixel coordinates
    perim_m = PitchRegionFilter._sample_pitch_perimeter(105.0, 68.0, margin_m=0.0, steps_per_side=6)
    world3 = np.column_stack([perim_m[:, 0], perim_m[:, 1], np.zeros(len(perim_m))]).reshape(-1, 1, 3)
    uv_fisheye, _ = cv2.fisheye.projectPoints(world3, rvec, tvec, K, D)
    uv_fisheye = uv_fisheye.reshape(-1, 2)
    kp_dict = {f"({x:.2f}, {y:.2f})": [float(u), float(v)] for (x, y), (u, v) in zip(perim_m, uv_fisheye)}
    kp_path = tmp_path / "999999_keypoints.json"
    kp_path.write_text(json.dumps(kp_dict), encoding="utf-8")

    # Separate undistorted homography with an intentional +120px shift to distinguish coordinate spaces
    H_p2u, _ = cv2.findHomography(
        np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32),
        np.array([[520, 320], [3720, 320], [3720, 1020], [520, 1020]], dtype=np.float32),
    )
    hom_path = tmp_path / "999999_homography.npy"
    np.save(hom_path, H_p2u)

    prf_raw = PitchRegionFilter.from_calibration_path(tmp_path, coordinate_space="raw_fisheye")
    prf_und = PitchRegionFilter.from_calibration_path(tmp_path, coordinate_space="undistorted")

    assert prf_raw.coordinate_space == "raw_fisheye"
    assert prf_und.coordinate_space == "undistorted"
    assert "fisheye_plus_keypoints" in prf_raw.source_description
    assert "undistorted_homography" in prf_und.source_description

    # Centre pitch point must project into PLAYABLE in its respective coordinate space
    centre_raw, _ = cv2.fisheye.projectPoints(np.array([[[52.5, 34.0, 0.0]]]), rvec, tvec, K, D)
    cr_u, cr_v = float(centre_raw[0, 0, 0]), float(centre_raw[0, 0, 1])
    assert prf_raw.classify_point(cr_u, cr_v) == PitchZone.PLAYABLE
    proj_back = prf_raw.project_pixel_to_pitch_m(cr_u, cr_v)
    assert proj_back is not None
    assert proj_back[0] == pytest.approx(52.5, abs=0.5)
    assert proj_back[1] == pytest.approx(34.0, abs=0.5)

    assert prf_und.classify_point(2120.0, 670.0) == PitchZone.PLAYABLE


# ---------------------------------------------------------------------------
# 3. Persistent fixed-camera static-background distractor mask
# ---------------------------------------------------------------------------


def test_fixed_camera_persistent_static_distractor_never_reappears_after_ttl() -> None:
    """Regression test for SoccerTrack 128057 spare ball at (3492, 828):
    A stationary sideline spare ball inside the boundary margin must be permanently
    masked on fixed cameras and NEVER reappear after the 250-frame broadcast TTL expires,
    even if a player jogs near the touchline.
    """
    prf = _build_synthetic_planar_pitch_filter(player_margin_m=2.0, ball_margin_m=3.0)
    cfg = BallTrackingConfig(
        fixed_camera=True,
        persistent_static_distractors=True,
        static_zone_ttl_frames=40,  # Short TTL to verify permanent fixed-camera masking ignores it
        static_distractor_min_hits=4,
    )
    tracker = KalmanBallTracker(match_id="128057_sim", config=cfg, pitch_region_filter=prf)

    # Sideline spare ball sits in the boundary margin (y=915, outside playable y<=900, inside ball margin y<=931)
    spare_ball = DetectionRecord(0, 0.0, 32, "ball", (3487, 910, 3497, 920), 0.55, None)
    far_player = DetectionRecord(0, 0.0, 0, "player", (1200, 450, 1230, 520), 0.90, 1)

    for f in range(0, 20, 2):
        tracker.step(
            frame_number=f,
            timestamp=f / 25.0,
            ball_candidates=[spare_ball],
            player_detections=[far_player],
        )

    assert len(tracker.permanent_static_zones) >= 1

    # Advance well beyond static_zone_ttl_frames (40) AND bring a player near the sideline
    near_sideline_player = DetectionRecord(0, 0.0, 0, "player", (3475, 835, 3505, 905), 0.90, 2)
    late_rec = tracker.step(
        frame_number=500,
        timestamp=500 / 25.0,
        ball_candidates=[spare_ball],
        player_detections=[near_sideline_player],
    )
    assert not late_rec.is_observed
    assert len(tracker.permanent_static_zones) >= 1

    # Contrast: an on-pitch PLAYABLE stationary spot (e.g. kickoff ball) DOES release when a player steps onto it
    tracker._register_static_distractor(2000.0, 550.0, frame_number=500)
    kickoff_player = DetectionRecord(0, 0.0, 0, "player", (1985, 495, 2015, 555), 0.90, 3)
    tracker._prune_blacklisted_zones(frame_number=502, player_detections=[kickoff_player], med_ph=60.0)
    assert all(
        np.hypot(zx - 2000.0, zy - 550.0) > 5.0 for zx, zy, _ in tracker.blacklisted_static_zones
    )


def test_broadcast_mode_still_expires_static_zone_after_ttl() -> None:
    cfg = BallTrackingConfig(fixed_camera=False, persistent_static_distractors=False, static_zone_ttl_frames=30)
    tracker = KalmanBallTracker(match_id="broadcast_sim", config=cfg)
    tracker._register_static_distractor(500.0, 400.0, frame_number=10)
    assert len(tracker.blacklisted_static_zones) == 1
    assert len(tracker.permanent_static_zones) == 0

    tracker._prune_blacklisted_zones(frame_number=50, player_detections=[], med_ph=70.0)
    assert len(tracker.blacklisted_static_zones) == 0


# ---------------------------------------------------------------------------
# 4. Fixed-camera motion-assisted small-football candidate recovery
# ---------------------------------------------------------------------------


def test_motion_ball_detector_recovers_small_moving_ball_and_masks_players_and_off_pitch() -> None:
    prf = _build_synthetic_planar_pitch_filter(player_margin_m=2.0, ball_margin_m=2.5)
    det = FixedCameraMotionBallDetector(
        config=MotionBallConfig(
            enabled=True,
            diff_threshold=15,
            bg_diff_threshold=15,
            min_local_contrast=20.0,
            require_three_frame_or_kalman=True,
            require_kalman_corroboration=False,
            verify_with_upscaled_yolo=False,
        ),
        pitch_region_filter=prf,
    )

    # Synthetic 1200x4000 green pitch panorama
    base = np.zeros((1200, 4000, 3), dtype=np.uint8)
    base[180:950, 350:3650] = (35, 115, 45)  # BGR pitch turf green

    recovered = []
    for step_idx in range(7):
        fno = step_idx * 2
        frame = base.copy()
        # Moving 10x10 bright white football inside playable pitch
        bx = 1600 + step_idx * 22
        by = 540
        cv2.circle(frame, (bx, by), 5, (245, 245, 245), -1)

        # Moving white decoy inside a detected player bbox (must be masked out)
        px = 2400 + step_idx * 10
        cv2.rectangle(frame, (px - 15, 480), (px + 15, 560), (235, 235, 235), -1)
        player_det = DetectionRecord(fno, fno / 25.0, 0, "player", (px - 18, 476, px + 18, 564), 0.90, 7)

        # Moving off-pitch bright spot below the touchline (must be masked by pitch ROI)
        cv2.circle(frame, (1800 + step_idx * 18, 1060), 5, (245, 245, 245), -1)

        cands = det.detect_candidates(
            frame_bgr=frame,
            frame_number=fno,
            timestamp=fno / 25.0,
            player_detections=[player_det],
        )
        if step_idx >= 3:
            recovered.append((bx, by, cands))

    # After 3-frame trajectory warm-up (frames 0, 2, 4 -> confirmed from step 3 onward), recover true moving ball only
    assert len(recovered) == 4
    for bx, by, cands in recovered:
        assert len(cands) == 1
        cx = (cands[0].bbox[0] + cands[0].bbox[2]) / 2.0
        cy = (cands[0].bbox[1] + cands[0].bbox[3]) / 2.0
        assert np.hypot(cx - bx, cy - by) <= 6.0


# ---------------------------------------------------------------------------
# 5. Direct Ultralytics tracker adapter & CLI compatibility
# ---------------------------------------------------------------------------


def test_numpy_detections_adapter_indexing_and_empty() -> None:
    empty = _NumpyDetections.from_records([])
    assert len(empty) == 0
    assert empty.xywh.shape == (0, 4)

    recs = [
        DetectionRecord(0, 0.0, 0, "player", (100, 200, 140, 300), 0.85, None),
        DetectionRecord(0, 0.0, 0, "player", (500, 250, 530, 330), 0.42, None),
    ]
    nd = _NumpyDetections.from_records(recs)
    assert len(nd) == 2
    assert nd.xywh[0].tolist() == pytest.approx([120.0, 250.0, 40.0, 100.0])
    high = nd[nd.conf > 0.5]
    assert len(high) == 1
    assert high.conf[0] == pytest.approx(0.85)


def test_cli_backward_compatibility_and_sprint3_2_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify legacy CLI invocations remain unchanged and new Sprint 3.2 flags wire cleanly."""
    import sys
    from src.pipeline import cli

    captured = {}

    def _fake_run_match_pipeline(config, experiment_name=None, max_tracking_frames=180, **kwargs):
        captured["config"] = config
        captured["experiment_name"] = experiment_name
        captured["max_frames"] = max_tracking_frames
        return {
            "experiment_id": experiment_name or "test",
            "output_dir": "outputs/test",
            "classification_summary": {"overall_accuracy": None, "macro_f1": None, "coverage": None},
            "tracking_summary": {
                "processed_frames": 10,
                "distinct_tracking_ids": 2,
                "raw_team_label_changes": 0,
                "smoothed_team_label_changes": 0,
                "stable_label_coverage": 1.0,
            },
            "detection_summary": {"ball_coverage_pct": 0.0},
            "possession_summary": {"attributed_possession_share": {}, "ball_tracking": {}, "possession_coverage_pct": 0.0},
            "report_files": {},
        }

    monkeypatch.setattr(cli, "run_match_pipeline", _fake_run_match_pipeline)

    # 1. Legacy broadcast CLI preserves full_frame + camera-cut defaults
    monkeypatch.setattr(
        sys, "argv", ["cli", "--video", "demo.mp4", "--auto-teams", "--enable-possession", "--max-frames", "10"]
    )
    cli.main()
    cfg_legacy = captured["config"]
    assert cfg_legacy.detection.inference_mode == "full_frame"
    assert cfg_legacy.ball_tracking.fixed_camera is False
    assert cfg_legacy.tracking.reset_on_camera_cut is True

    # 2. Sprint 3.2 fixed-camera panoramic CLI wires tiling, pitch filter, and persistent distractors
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "--video", "pano.mp4",
            "--auto-teams",
            "--enable-possession",
            "--detection-mode", "tiled",
            "--model", "yolo11s.pt",
            "--person-conf", "0.28",
            "--ball-conf", "0.14",
            "--tile-width", "1280",
            "--tile-height", "1280",
            "--tile-imgsz", "1280",
            "--fixed-camera",
            "--pitch-calibration", "cal_dir",
            "--tracker", "bytetrack.yaml",
        ],
    )
    cli.main()
    cfg_pano = captured["config"]
    assert cfg_pano.detection.inference_mode == "tiled"
    assert cfg_pano.detection.model_name == "yolo11s.pt"
    assert cfg_pano.detection.person_conf == pytest.approx(0.28)
    assert cfg_pano.detection.ball_conf == pytest.approx(0.14)
    assert cfg_pano.ball_tracking.fixed_camera is True
    assert cfg_pano.ball_tracking.persistent_static_distractors is True
    assert cfg_pano.ball_tracking.enable_motion_ball_recovery is True
    assert cfg_pano.ball_tracking.roi_upscale_factor == pytest.approx(2.0)
    assert cfg_pano.tracking.reset_on_camera_cut is False
    assert cfg_pano.tracking.tracker_config == "bytetrack.yaml"
    assert cfg_pano.pitch.calibration_path == "cal_dir"
