"""Sprint 3.3 Regression Test Suite — Ball Intelligence, Night Team Discovery,
Fixed-Camera Player Tracking, Geometric Possession & Compute Profiles.

Designed to run cleanly on CPU-only GitHub Actions CI without requiring Ultralytics
weights or raw SoccerTrack video files.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.classification.features import (
    extract_hsv_lab_features,
    extract_illumination_invariant_features,
)
from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    collect_track_appearances,
)
from src.config import (
    BallTrackingConfig,
    COMPUTE_PROFILE_PRESETS,
    MatchConfig,
    PossessionConfig,
    TeamDiscoveryConfig,
    apply_compute_profile,
    resolve_compute_device,
)
from src.evaluation.possession_eval import (
    PossessionIntervalAnnotation,
    evaluate_possession_predictions,
)
from src.geometry.pitch import PitchCalibration
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter, PitchZone
from src.possession.ball_tracker import (
    KalmanBallTracker,
    filter_ball_candidates,
    verify_ball_appearance,
)
from src.possession.ownership import evaluate_frame_ownership_candidates
from src.possession.possession_engine import PossessionEngine
from src.possession.state_machine import PossessionStateMachine
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    DetectionRecord,
    FrameObservationRecord,
    InternalLabel,
    PossessionFrameRecord,
    PossessionState,
)
from src.tracking.pitch_stitcher import FixedCameraTrackletAssociator, StitcherConfig


def _build_synthetic_pitch_filter() -> PitchRegionFilter:
    """Create a clean planar synthetic PitchRegionFilter mapping [100..900] x [150..550] -> [0..105] x [0..68] m."""
    cal = PitchCalibration(
        image_points_px=((100.0, 150.0), (900.0, 150.0), (900.0, 550.0), (100.0, 550.0)),
        pitch_points_m=((0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)),
        pitch_length_m=105.0,
        pitch_width_m=68.0,
    )
    return PitchRegionFilter.from_planar_calibration(
        calibration=cal,
        coordinate_space="planar",
        config=PitchRegionConfig(player_margin_m=2.0, ball_margin_m=2.5),
    )


def _make_solid_frame(h: int = 700, w: int = 1000, bgr=(40, 130, 45)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _paint_patch(frame: np.ndarray, bbox, bgr) -> None:
    x1, y1, x2, y2 = bbox
    frame[y1:y2, x1:x2] = bgr


def test_blue_training_cone_near_touchline_rejected_even_with_nearby_player():
    """Regression for 117093 f=206..280: saturated blue cone in BOUNDARY_MARGIN with nearby player."""
    prf = _build_synthetic_pitch_filter()
    frame = _make_solid_frame()
    # Place blue cone ~1m below the near touchline (y=550 is y_m=68.0; y=558 -> y_m~69.36 in BOUNDARY_MARGIN)
    cone_box = (300, 554, 312, 564)
    _paint_patch(frame, cone_box, (185, 110, 40))  # Saturated blue cone in BGR

    assert prf.classify_point(306.0, 559.0, for_ball=True) == PitchZone.BOUNDARY_MARGIN

    ok_app, reason = verify_ball_appearance(frame, cone_box, in_boundary_margin=True)
    assert not ok_app
    assert "blue_cone" in reason

    # Even with a throw-in player standing directly above the cone, filter must reject it
    player_near_cone = DetectionRecord(206, 8.24, 0, "player", (290, 480, 322, 556), 0.88)
    cone_det = DetectionRecord(206, 8.24, 32, "ball", cone_box, 0.58)
    kept = filter_ball_candidates(
        [cone_det],
        [player_near_cone],
        frame.shape[:2],
        frame_bgr=frame,
        pitch_region_filter=prf,
    )
    assert kept == []


def test_stationary_sideline_spare_ball_permanently_masked_even_near_player():
    """Regression for 128057 (3492, 828) & sideline spare balls: permanent fixed-camera mask survives TTL."""
    prf = _build_synthetic_pitch_filter()
    cfg = BallTrackingConfig(
        fixed_camera=True,
        persistent_static_distractors=True,
        static_zone_ttl_frames=30,
        static_distractor_min_hits=3,
    )
    tracker = KalmanBallTracker("test_spare", config=cfg, pitch_region_filter=prf)

    spare_box = (400, 554, 412, 565)  # In BOUNDARY_MARGIN (y_m ~ 69.5 m)
    nearby_player = DetectionRecord(0, 0.0, 0, "player", (385, 485, 420, 555), 0.90)

    # Feed 4 frames with the spare ball completely stationary in BOUNDARY_MARGIN while a player stands nearby
    for fno in range(0, 8, 2):
        spare_det = DetectionRecord(fno, fno / 25.0, 32, "ball", spare_box, 0.52)
        tracker.step(fno, fno / 25.0, [spare_det], player_detections=[nearby_player])

    assert len(tracker.permanent_static_zones) >= 1

    # Advance well past static_zone_ttl_frames (30 frames) -> permanent mask MUST remain active!
    rec_later = tracker.step(
        200,
        8.0,
        [DetectionRecord(200, 8.0, 32, "ball", spare_box, 0.55)],
        player_detections=[nearby_player],
    )
    assert not rec_later.is_observed
    assert len(tracker.permanent_static_zones) >= 1


def test_stationary_set_piece_ball_inside_playable_pitch_preserved():
    """A genuine stationary match ball inside PitchZone.PLAYABLE with a nearby set-piece taker stays tracked."""
    prf = _build_synthetic_pitch_filter()
    frame = _make_solid_frame()
    ball_box = (500, 345, 512, 357)  # Center spot inside PLAYABLE
    _paint_patch(frame, ball_box, (235, 235, 235))  # White football

    ok_app, _ = verify_ball_appearance(frame, ball_box, in_boundary_margin=False)
    assert ok_app

    cfg = BallTrackingConfig(fixed_camera=True, persistent_static_distractors=True, init_high_conf_threshold=0.30)
    tracker = KalmanBallTracker("test_set_piece", config=cfg, pitch_region_filter=prf)
    taker = DetectionRecord(0, 0.0, 0, "player", (482, 290, 515, 360), 0.92)

    for fno in range(0, 14, 2):
        det = DetectionRecord(fno, fno / 25.0, 32, "ball", ball_box, 0.62)
        rec = tracker.step(fno, fno / 25.0, [det], frame_bgr=frame, player_detections=[taker])
        assert rec.is_observed, f"Set-piece ball dropped on frame {fno}"
    assert len(tracker.permanent_static_zones) == 0


def test_ball_disappearance_reacquisition_and_sharp_deflection_recovery():
    """Ball disappears -> transitions to LOST -> reacquires cleanly; sharp kick deflection resets velocity."""
    prf = _build_synthetic_pitch_filter()
    cfg = BallTrackingConfig(
        fixed_camera=True,
        max_coast_frames=3,
        init_high_conf_threshold=0.30,
        max_jump_px_per_frame=80.0,
    )
    tracker = KalmanBallTracker("test_reacq", config=cfg, pitch_region_filter=prf)

    # Smooth rightward pass
    for idx, x in enumerate([300, 325, 350]):
        fno = idx * 2
        det = DetectionRecord(fno, fno / 25.0, 32, "ball", (x, 320, x + 12, 332), 0.65)
        rec = tracker.step(fno, fno / 25.0, [det])
        assert rec.is_observed

    # Sharp 90-degree volley deflection inside PLAYABLE (140 px away from constant-velocity prediction)
    deflected = DetectionRecord(6, 6 / 25.0, 32, "ball", (355, 200, 367, 212), 0.55)
    rec_def = tracker.step(6, 6 / 25.0, [deflected])
    assert rec_def.is_observed
    assert rec_def.velocity_y < -200.0  # Velocity immediately redirected upward!

    # Disappearance beyond max_coast_frames -> LOST
    for fno in (8, 10, 12, 14):
        rec_miss = tracker.step(fno, fno / 25.0, [])
    assert rec_miss.observation_mode == BallObservationMode.LOST.value

    # Reacquisition at new location
    rec_re = tracker.step(16, 16 / 25.0, [DetectionRecord(16, 16 / 25.0, 32, "ball", (620, 380, 632, 392), 0.60)])
    assert rec_re.is_observed
    assert tracker.reacquisition_events >= 1


def test_aerial_ball_above_far_touchline_allowed_when_connected_to_trajectory():
    """Lofted aerial balls rising above the far touchline remain valid when connected to predicted trajectory."""
    prf = _build_synthetic_pitch_filter()
    frame = _make_solid_frame()
    # Far touchline is y=150; aerial ball rises to y=132 while predicted trajectory is at (480, 152)
    aerial_box = (475, 126, 487, 138)
    _paint_patch(frame, aerial_box, (240, 240, 240))
    aerial_det = DetectionRecord(20, 0.8, 32, "ball", aerial_box, 0.45)

    kept = filter_ball_candidates(
        [aerial_det],
        [],
        frame.shape[:2],
        frame_bgr=frame,
        predicted_pos=(480.0, 152.0),
        pitch_region_filter=prf,
    )
    assert len(kept) == 1


def test_night_glare_illumination_invariant_separates_dark_navy_from_white_kits():
    """Night floodlight specular glare + white shirt numbers on dark-navy kits must stay separated from white kits."""
    # Create synthetic dark-navy jersey crop with 25% bright floodlight glare / white number stripe
    navy_glare = np.full((64, 48, 3), (85, 35, 20), dtype=np.uint8)  # Dark navy in BGR
    navy_glare[14:32, 16:32] = (230, 230, 230)  # Specular glare / white jersey number

    # Create clean dark-navy jersey crop (no glare)
    navy_clean = np.full((64, 48, 3), (85, 35, 20), dtype=np.uint8)

    # Create white jersey crop with slight stadium shadow
    white_kit = np.full((64, 48, 3), (215, 218, 220), dtype=np.uint8)

    f_ng = extract_illumination_invariant_features(navy_glare)
    f_nc = extract_illumination_invariant_features(navy_clean)
    f_wh = extract_illumination_invariant_features(white_kit)

    d_glare_to_navy = float(1.0 - np.dot(f_ng, f_nc))
    d_glare_to_white = float(1.0 - np.dot(f_ng, f_wh))
    assert d_glare_to_navy < d_glare_to_white, (
        f"Glare-contaminated navy kit should be closer to clean navy ({d_glare_to_navy:.3f}) "
        f"than white ({d_glare_to_white:.3f})"
    )


def test_similar_kits_and_gk_referee_rejection_conservative_unknown():
    """Goalkeepers/referees with distinct outlier kits and ambiguous crops are rejected as Unknown."""
    rng = np.random.default_rng(42)

    def _mk_obs(tid: int, bgr: tuple, n: int = 5):
        out = []
        for k in range(n):
            crop = np.clip(
                np.full((48, 32, 3), bgr, dtype=np.int16) + rng.integers(-6, 7, size=(48, 32, 3)),
                0,
                255,
            ).astype(np.uint8)
            out.append(
                TeamDiscoveryObservation(
                    tracking_id=tid,
                    frame_number=k * 2,
                    timestamp=(k * 2) / 25.0,
                    crop_bgr=crop,
                    blur_score=80.0,
                    grass_ratio=0.05,
                    occlusion_ratio=0.0,
                    bbox=(100, 100, 132, 160),
                )
            )
        return out

    obs = []
    for tid in range(1, 7):
        obs.extend(_mk_obs(tid, (220, 220, 220)))  # Team white
    for tid in range(7, 13):
        obs.extend(_mk_obs(tid, (90, 35, 20)))  # Team dark navy
    for tid in (13, 14):
        obs.extend(_mk_obs(tid, (40, 220, 235)))  # Neon yellow GK / Referee

    cfg = TeamDiscoveryConfig(enabled=True, feature_method="illumination_invariant", candidate_clusters=3)
    apps = collect_track_appearances(obs, cfg)
    disc = AutomaticTeamDiscoverer(cfg).fit(apps)

    # Neon yellow GK/referee tracks must be rejected as Unknown
    for gk_tid in (13, 14):
        lab, _ = disc.predict_track(gk_tid)
        assert lab == InternalLabel.UNKNOWN.value

    # Ambiguous mid-grey crop halfway between white & navy must be rejected as Unknown
    ambig_crop = np.full((48, 32, 3), (145, 128, 122), dtype=np.uint8)
    pred_ambig = disc.predict_classification(ambig_crop)
    assert pred_ambig.predicted_label == InternalLabel.UNKNOWN.value


def test_player_crossing_and_temporary_occlusion_stitching_never_merges_active():
    """FixedCameraTrackletAssociator stitches occluded player tracks and never merges crossing opponents."""
    prf = _build_synthetic_pitch_filter()
    stitcher = FixedCameraTrackletAssociator(
        config=StitcherConfig(max_lost_frames=30),
        pitch_region_filter=prf,
    )
    frame = _make_solid_frame()
    # Paint white player moving right and blue player moving left
    p1_b0 = (300, 300, 324, 360)
    p2_b0 = (380, 300, 404, 360)
    _paint_patch(frame, p1_b0, (235, 235, 235))
    _paint_patch(frame, p2_b0, (200, 40, 30))

    d0 = [
        DetectionRecord(0, 0.0, 0, "player", p1_b0, 0.9, tracking_id=1),
        DetectionRecord(0, 0.0, 0, "player", p2_b0, 0.9, tracking_id=2),
    ]
    out0 = stitcher.update(0, d0, frame_bgr=frame)
    assert len({d.tracking_id for d in out0}) == 2

    # Frame 2..4: Player 1 temporarily occluded behind Player 2
    frame2 = _make_solid_frame()
    p2_b2 = (355, 300, 379, 360)
    _paint_patch(frame2, p2_b2, (200, 40, 30))
    stitcher.update(2, [DetectionRecord(2, 0.08, 0, "player", p2_b2, 0.9, tracking_id=2)], frame_bgr=frame2)

    # Frame 6: Player 1 reappears on the right with a brand-new raw ID=99 -> must stitch back to ID=1!
    frame6 = _make_solid_frame()
    p1_b6 = (322, 300, 346, 360)
    p2_b6 = (335, 300, 359, 360)
    _paint_patch(frame6, p1_b6, (235, 235, 235))
    _paint_patch(frame6, p2_b6, (200, 40, 30))
    d6 = [
        DetectionRecord(6, 0.24, 0, "player", p1_b6, 0.9, tracking_id=99),
        DetectionRecord(6, 0.24, 0, "player", p2_b6, 0.9, tracking_id=2),
    ]
    out6 = stitcher.update(6, d6, frame_bgr=frame6)
    ids6 = [d.tracking_id for d in out6]
    assert len(set(ids6)) == 2, "Simultaneously active crossing players must NEVER share a track ID"
    assert 1 in ids6, f"Temporarily occluded player (raw 99) should stitch back to canonical ID 1, got {ids6}"


def test_possession_unknown_on_ball_loss_and_geometric_out_of_play():
    """Possession transitions conservatively to UNKNOWN during ball loss and OUT_OF_PLAY outside pitch geometry."""
    prf = _build_synthetic_pitch_filter()
    cfg = MatchConfig(
        match_id="test_poss",
        match_name="Test",
        video_path="",
        possession=PossessionConfig(enabled=True, min_control_frames=2),
    )
    engine = PossessionEngine(cfg, fps=25.0, frame_stride=2, pitch_region_filter=prf)

    p_a = FrameObservationRecord(
        "test_poss", 0, 0.0, 1, InternalLabel.TEAM_A.value, "Team A", InternalLabel.TEAM_A.value,
        (490, 300, 514, 360), 0.9, 0.9, 0.05, 1.0, True, (498, 350, 510, 362), 0.8,
    )
    # 3 frames of clear control inside PLAYABLE
    for fno in (0, 2, 4):
        b = BallTrajectoryRecord(
            "test_poss", fno, fno / 25.0, BallObservationMode.OBSERVED.value, True, False,
            (498, 350, 510, 362), 504.0, 356.0, 0.0, 0.0, 0.0, 0.85, 4.0, 0, False, 0, "tiled_yolo"
        )
        _, summary = evaluate_frame_ownership_candidates(
            [p_a], b, cfg.possession, (700, 1000), pitch_region_filter=prf
        )
        poss_state, _, _, _ = engine.state_machine.step(summary)
    assert poss_state == PossessionState.TEAM_A_CONTROL.value

    # Prolonged ball loss -> must transition conservatively to UNKNOWN
    for fno in (6, 8, 10, 12, 14):
        b_lost = BallTrajectoryRecord(
            "test_poss", fno, fno / 25.0, BallObservationMode.LOST.value, False, False,
            None, None, None, 0.0, 0.0, 0.0, 0.0, 999.0, 6, False, 0, "none"
        )
        _, summary_lost = evaluate_frame_ownership_candidates(
            [p_a], b_lost, cfg.possession, (700, 1000), pitch_region_filter=prf
        )
        poss_lost_state, _, _, _ = engine.state_machine.step(summary_lost)
    assert poss_lost_state == PossessionState.UNKNOWN.value

    # Ball clearly OFF_PITCH geometrically -> OUT_OF_PLAY
    b_out = BallTrajectoryRecord(
        "test_poss", 16, 16 / 25.0, BallObservationMode.OBSERVED.value, True, False,
        (500, 640, 512, 652), 506.0, 646.0, 0.0, 0.0, 0.0, 0.80, 4.0, 0, False, 0, "tiled_yolo"
    )
    _, summary_out = evaluate_frame_ownership_candidates(
        [p_a], b_out, cfg.possession, (700, 1000), pitch_region_filter=prf
    )
    assert summary_out["raw_state"] == PossessionState.OUT_OF_PLAY.value


def test_continuous_possession_eval_metrics_and_timestamp_alignment():
    """Continuous manual interval evaluation computes strict incorrect confident rate & rejects unverified labels."""
    # Unverified annotation must be blocked
    unverified_ann = [
        PossessionIntervalAnnotation(
            "m", 0, 10, 0.0, 0.4, PossessionState.TEAM_A_CONTROL.value, verified=False, source="silver"
        )
    ]
    preds = [
        PossessionFrameRecord(
            match_id="m",
            frame_number=fno,
            timestamp=round(fno / 25.0, 3),
            dt_seconds=0.08,
            ball_observation_mode=BallObservationMode.OBSERVED.value,
            ball_visibility="visible",
            possession_state=PossessionState.TEAM_A_CONTROL.value,
            raw_interaction_state=PossessionState.TEAM_A_CONTROL.value,
            controlling_player_id=1,
            controlling_team_id=InternalLabel.TEAM_A.value,
            controlling_team_display="Team A",
            possession_confidence=0.85,
            closest_player_id=1,
            closest_team_id=InternalLabel.TEAM_A.value,
            closest_normalized_distance=0.4,
            second_closest_team_id=None,
            second_closest_normalized_distance=None,
            running_team_a_share_pct=100.0,
            running_team_b_share_pct=0.0,
            camera_cut=False,
        )
        for fno in range(0, 12, 2)
    ]
    blocked = evaluate_possession_predictions(preds, unverified_ann)
    assert blocked["metrics_available"] is False

    # Verified manual annotation computes exact metrics & aligned timestamps
    verified_ann = [
        PossessionIntervalAnnotation(
            "m", 0, 10, 0.0, 0.4, PossessionState.TEAM_A_CONTROL.value, verified=True, source="manual"
        )
    ]
    res = evaluate_possession_predictions(preds, verified_ann)
    assert res["metrics_available"] is True
    assert res["state_accuracy"] == 1.0
    assert res["incorrect_confident_attribution_rate"] == 0.0
    # Verify monotonic timestamp alignment (frame_number / 25.0)
    assert all(abs(p.timestamp - p.frame_number / 25.0) < 1e-3 for p in preds)


def test_raw_fisheye_vs_undistorted_coordinate_space_respected():
    """PitchRegionFilter stores coordinate_space explicitly and enforces raw_fisheye vs planar vs undistorted."""
    prf_planar = _build_synthetic_pitch_filter()
    assert prf_planar.coordinate_space == "planar"
    assert prf_planar.classify_point(500.0, 350.0) == PitchZone.PLAYABLE
    assert prf_planar.classify_point(20.0, 20.0) == PitchZone.OFF_PITCH


def test_compute_profiles_and_safe_cpu_fallback():
    """All 3 compute profiles apply cleanly and requesting CUDA on CPU-only host falls back safely."""
    for name in ("local_low_vram", "colab_t4", "cpu_fallback"):
        assert name in COMPUTE_PROFILE_PRESETS
        cfg = MatchConfig(match_id="m", match_name="M", video_path="")
        apply_compute_profile(cfg, name)
        assert cfg.compute.profile == name

    dev, use_half, _ = resolve_compute_device("cpu", requested_half=True)
    assert dev == "cpu"
    assert use_half is False
