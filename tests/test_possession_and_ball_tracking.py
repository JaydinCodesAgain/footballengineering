"""Automated tests for Phase 1–7: dataset regeneration, Kalman ball tracking, ownership, state machine, and possession metrics."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.classification.dataset import DatasetIntegrityError, JerseyCropDataset
from src.config import BallTrackingConfig, MatchConfig, PossessionConfig
from src.evaluation.possession_eval import (
    PossessionAnnotationStore,
    PossessionIntervalAnnotation,
    evaluate_possession_predictions,
    validate_no_overlapping_intervals,
)
from src.detection.annotation_workflow import (
    DetectionAnnotationStore,
    sample_representative_frame_numbers,
)
from src.evaluation.report import (
    export_ball_trajectory_records,
    export_player_ownership_records,
    export_possession_timeline_records,
    generate_possession_report,
)
from src.possession.ball_tracker import (
    KalmanBallTracker,
    detect_camera_cut,
    filter_ball_candidates,
)
from src.possession.metrics import compute_possession_summary
from src.possession.ownership import (
    compute_normalized_ball_distance,
    evaluate_frame_ownership_candidates,
)
from src.possession.possession_engine import PossessionEngine
from src.possession.state_machine import PossessionStateMachine
from src.schemas import (
    BallObservationMode,
    CropQualityMetrics,
    DetectionRecord,
    FrameObservationRecord,
    InternalLabel,
    PossessionState,
)
from tests.conftest import make_synthetic_kit_crop


def _make_match_config() -> MatchConfig:
    return MatchConfig(
        match_id="test_match",
        match_name="Norway vs Wales",
        video_path="dummy.mp4",
        label_mapping={
            InternalLabel.TEAM_A.value: "Norway",
            InternalLabel.TEAM_B.value: "Wales",
            InternalLabel.REFEREE.value: "REF",
            InternalLabel.UNKNOWN.value: "Unknown",
        },
    )


def _make_player_obs(
    frame_no: int,
    ts: float,
    player_id: int,
    team_id: str,
    bbox: tuple[int, int, int, int],
    team_conf: float = 0.85,
) -> FrameObservationRecord:
    return FrameObservationRecord(
        match_id="test_match",
        frame_number=frame_no,
        timestamp=ts,
        player_id=player_id,
        team_id=team_id,
        team_display_name=team_id,
        raw_team_id=team_id,
        player_bbox=bbox,
        player_detection_confidence=0.90,
        classification_confidence=team_conf,
        classification_distance=0.12,
        track_team_confidence=team_conf,
        is_team_locked=True,
    )


def test_dataset_integrity_validation_and_deterministic_regeneration(tmp_path: Path) -> None:
    cfg = _make_match_config()
    video_path = tmp_path / "synthetic_match.mp4"

    # Write a 5-frame synthetic MP4 video with a crisp textured jersey patch on frame 2
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (320, 240),
    )
    kit = make_synthetic_kit_crop((20, 20, 220), (240, 240, 240), striped=True, height=64, width=48)
    for f in range(5):
        frm = np.full((240, 320, 3), (40, 160, 50), dtype=np.uint8)
        if f == 2:
            frm[80:144, 100:148] = kit
        writer.write(frm)
    writer.release()

    # Read back frame 2 from decoded video so quality metrics match codec compression
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    _, decoded_frame2 = cap.read()
    cap.release()
    decoded_crop = decoded_frame2[80:144, 100:148].copy()

    from src.classification.cropper import compute_blur_score, compute_grass_mask

    blur_s = compute_blur_score(decoded_crop)
    _, grass_r = compute_grass_mask(decoded_crop)
    qm = CropQualityMetrics(48, 64, blur_s, grass_r, 0.0, False, True)

    ds = JerseyCropDataset(tmp_path / "crops", cfg)
    rec = ds.add_crop(
        decoded_crop,
        frame_number=2,
        timestamp=0.08,
        bbox=(90, 60, 158, 180),
        torso_bbox=(100, 80, 148, 144),
        quality=qm,
        ground_truth_label="Norway",
        label_verified=True,
        split="train",
    )
    ds.save()
    assert ds.validate_integrity()["is_valid"] is True

    # Simulate fresh git clone by deleting the PNG file while keeping manifest.json
    img_file = tmp_path / "crops" / rec.image_path
    img_file.unlink()

    ds_fresh = JerseyCropDataset(tmp_path / "crops", cfg)
    integrity_missing = ds_fresh.validate_integrity()
    assert integrity_missing["is_valid"] is False
    assert rec.crop_id in integrity_missing["missing_crop_ids"]

    # Attempting to load without regenerating or with missing video must raise DatasetIntegrityError
    with pytest.raises(DatasetIntegrityError, match="missing"):
        ds_fresh.load_crop_image(rec.crop_id)

    with pytest.raises(DatasetIntegrityError, match="source video was not found"):
        ds_fresh.regenerate_missing_crops(tmp_path / "nonexistent.mp4")

    # Regenerate from actual source video -> restores image and preserves verified label
    regen_res = ds_fresh.regenerate_missing_crops(video_path)
    assert regen_res["regenerated_count"] == 1
    assert ds_fresh.validate_integrity()["is_valid"] is True
    assert ds_fresh.records[rec.crop_id].ground_truth_label == InternalLabel.TEAM_A.value
    assert ds_fresh.records[rec.crop_id].label_verified is True

    # Regression test 1: Dimension-mismatched image file on disk (e.g. overwritten with 10x10 image)
    # while manifest metadata and torso_bbox are valid -> must be included in to_recover and repaired!
    cv2.imwrite(str(img_file), np.zeros((10, 10, 3), dtype=np.uint8))
    mismatch_check = ds_fresh.validate_integrity()
    assert mismatch_check["is_valid"] is False
    assert rec.crop_id in mismatch_check["dimension_mismatch_ids"]

    repair_mismatch = ds_fresh.regenerate_missing_crops(video_path)
    assert repair_mismatch["regenerated_count"] == 1
    assert ds_fresh.validate_integrity()["is_valid"] is True
    assert ds_fresh.records[rec.crop_id].label_verified is True

    # Regression test 2: Corrupted (0-byte undecodable) image file on disk -> repaired!
    img_file.write_bytes(b"not_a_png")
    corrupt_check = ds_fresh.validate_integrity()
    assert corrupt_check["is_valid"] is False
    assert rec.crop_id in corrupt_check["corrupted_crop_ids"]

    repair_corrupt = ds_fresh.regenerate_missing_crops(video_path)
    assert repair_corrupt["regenerated_count"] == 1
    assert ds_fresh.validate_integrity()["is_valid"] is True

    # Regression test 3: Partial regeneration when one record is valid and a second record has mismatched metadata
    rec2 = ds_fresh.add_crop(
        decoded_crop,
        frame_number=2,
        timestamp=0.08,
        bbox=(90, 60, 158, 180),
        torso_bbox=(100, 80, 148, 144),
        quality=qm,
        ground_truth_label="Wales",
        label_verified=True,
        split="val",
        crop_id_override="test_match_bad_record",
    )
    img_file.unlink()
    (tmp_path / "crops" / rec2.image_path).unlink()
    ds_fresh.records[rec2.crop_id].quality.width = 999  # Force identity mismatch on rec2 only

    with pytest.raises(DatasetIntegrityError, match="Failed to verify crop identity"):
        ds_fresh.regenerate_missing_crops(video_path)

    # Verify rec1 was successfully regenerated and saved, while rec2 was marked unverified
    assert img_file.is_file()
    assert ds_fresh.records[rec.crop_id].label_verified is True
    assert ds_fresh.records[rec2.crop_id].label_verified is False


def test_kalman_ball_tracker_association_occlusion_outlier_rejection_and_camera_cut() -> None:
    cfg = BallTrackingConfig(
        max_coast_frames=3,
        max_jump_px_per_frame=100.0,
        reacquire_confirm_frames=2,
        interpolation_conf_decay=0.80,
    )
    tracker = KalmanBallTracker(match_id="m1", config=cfg)

    # Frame 0: initial observation at (200, 300)
    b0 = DetectionRecord(0, 0.0, 32, "ball", (192, 292, 208, 308), 0.80)
    r0 = tracker.step(0, 0.0, [b0])
    assert r0.observation_mode == BallObservationMode.OBSERVED.value
    assert r0.is_observed is True and r0.is_interpolated is False
    assert pytest.approx(r0.center_x, abs=1.0) == 200.0

    # Frame 2: valid move to (230, 300) + an implausible distant false positive at (950, 50)
    b_valid = DetectionRecord(2, 0.08, 32, "ball", (222, 292, 238, 308), 0.72)
    b_outlier = DetectionRecord(2, 0.08, 32, "ball", (942, 42, 958, 58), 0.90)
    r2 = tracker.step(2, 0.08, [b_valid, b_outlier])
    assert r2.observation_mode == BallObservationMode.OBSERVED.value
    assert r2.rejected_outliers_count == 1
    assert r2.center_x is not None and 215.0 <= r2.center_x <= 240.0
    assert r2.velocity_x > 0.0

    # Frame 3: single outlier at (900, 600) with no in-gate ball -> rejected, tracker coasts (INTERPOLATED)
    r3 = tracker.step(3, 0.12, [DetectionRecord(3, 0.12, 32, "ball", (892, 592, 908, 608), 0.85)])
    assert r3.observation_mode == BallObservationMode.INTERPOLATED.value
    assert r3.is_observed is False and r3.is_interpolated is True
    assert r3.rejected_outliers_count == 1
    assert r3.confidence < r2.confidence

    # Frames 4, 5, 6: missing detections -> coasts while frames_since_observed <= max_coast_frames=3, then LOST
    r4 = tracker.step(4, 0.16, [])
    r5 = tracker.step(5, 0.20, [])
    assert r4.observation_mode == BallObservationMode.INTERPOLATED.value
    assert r5.observation_mode == BallObservationMode.INTERPOLATED.value

    r6 = tracker.step(6, 0.24, [])
    assert r6.observation_mode == BallObservationMode.LOST.value
    assert r6.bbox is None and r6.center_x is None

    # Re-acquire and then test camera cut reset
    tracker.step(7, 0.28, [DetectionRecord(7, 0.28, 32, "ball", (400, 400, 416, 416), 0.88)])
    frm_pitch = np.full((180, 320, 3), (40, 170, 50), dtype=np.uint8)  # Green pitch
    frm_crowd = np.full((180, 320, 3), (220, 20, 20), dtype=np.uint8)  # Blue/red studio cut
    assert detect_camera_cut(frm_pitch, frm_crowd) is True

    tracker.step(8, 0.32, [DetectionRecord(8, 0.32, 32, "ball", (410, 400, 426, 416), 0.88)], frame_bgr=frm_pitch)
    r_cut = tracker.step(9, 0.36, [], frame_bgr=frm_crowd)
    assert r_cut.camera_cut_reset is True
    assert r_cut.observation_mode == BallObservationMode.LOST.value


def test_scale_normalized_distance_and_ownership_exclusions() -> None:
    # Far player (height=40px) vs near player (height=120px) both at 20px from the ball
    far_box = (100, 100, 120, 140)   # foot at (110, 138)
    near_box = (300, 200, 350, 320)  # foot at (325, 314)

    _, norm_far, _ = compute_normalized_ball_distance(far_box, (110.0, 158.0))
    _, norm_near, _ = compute_normalized_ball_distance(near_box, (325.0, 334.0))
    # 20px is 0.50 of the far player's height, but only 0.167 of the near player's height
    assert norm_far > norm_near * 2.5

    p_cfg = PossessionConfig(control_distance_threshold=0.70, exclude_referees=True)
    tracker = KalmanBallTracker("test_match")
    ball_rec = tracker.step(1, 0.04, [DetectionRecord(1, 0.04, 32, "ball", (195, 290, 205, 300), 0.85)])

    # Referee standing directly on the ball (200, 300) must be excluded from ownership
    ref_obs = _make_player_obs(1, 0.04, 99, InternalLabel.REFEREE.value, (180, 200, 220, 300))
    proximities, raw_info = evaluate_frame_ownership_candidates([ref_obs], ball_rec, config=p_cfg)
    assert proximities[0].eligible_for_ownership is False
    assert raw_info["raw_state"] == PossessionState.LOOSE_BALL.value
    assert raw_info["controlling_team_id"] is None

    # Unknown / Goalkeeper player standing directly on the ball must return UNKNOWN, never inventing Team_A/B
    gk_obs = _make_player_obs(1, 0.04, 1, InternalLabel.UNKNOWN.value, (180, 200, 220, 300))
    _, raw_gk = evaluate_frame_ownership_candidates([gk_obs], ball_rec, config=p_cfg)
    assert raw_gk["raw_state"] == PossessionState.UNKNOWN.value
    assert raw_gk["controlling_team_id"] is None

    # Equidistant Team_A and Team_B players within contested threshold -> CONTESTED
    p_a = _make_player_obs(1, 0.04, 10, InternalLabel.TEAM_A.value, (170, 200, 210, 300))
    p_b = _make_player_obs(1, 0.04, 20, InternalLabel.TEAM_B.value, (190, 200, 230, 300))
    _, raw_duel = evaluate_frame_ownership_candidates([p_a, p_b], ball_rec, config=p_cfg)
    assert raw_duel["raw_state"] == PossessionState.CONTESTED.value


def test_possession_state_machine_hysteresis_and_pass_continuity() -> None:
    cfg = PossessionConfig(min_control_frames=2, max_pass_retention_frames=4)
    sm = PossessionStateMachine(cfg)

    # Frame 1: Team A candidate -> hysteresis holds in LOOSE_BALL until min_control_frames=2
    st1, t1, _, _ = sm.step({"raw_state": PossessionState.TEAM_A_CONTROL.value, "controlling_player_id": 7, "confidence": 0.8})
    assert st1 == PossessionState.LOOSE_BALL.value and t1 is None

    # Frame 2: Team A confirmed!
    st2, t2, pid2, _ = sm.step({"raw_state": PossessionState.TEAM_A_CONTROL.value, "controlling_player_id": 7, "confidence": 0.85})
    assert st2 == PossessionState.TEAM_A_CONTROL.value
    assert t2 == InternalLabel.TEAM_A.value and pid2 == 7

    # Frame 3: Single-frame brush by Team B -> hysteresis blocks instant turnover, emits CONTESTED
    st3, t3, _, _ = sm.step({"raw_state": PossessionState.TEAM_B_CONTROL.value, "controlling_player_id": 11, "confidence": 0.75})
    assert st3 == PossessionState.CONTESTED.value and t3 is None

    # Frame 4: Pass in flight (LOOSE_BALL) followed on Frame 5 by Team A player #9 receiving -> immediate TEAM_A_CONTROL
    st4, _, _, _ = sm.step({"raw_state": PossessionState.LOOSE_BALL.value, "controlling_player_id": None, "confidence": 0.70})
    assert st4 == PossessionState.LOOSE_BALL.value
    st5, t5, pid5, _ = sm.step({"raw_state": PossessionState.TEAM_A_CONTROL.value, "controlling_player_id": 9, "confidence": 0.88})
    assert st5 == PossessionState.TEAM_A_CONTROL.value
    assert t5 == InternalLabel.TEAM_A.value and pid5 == 9


def test_possession_engine_aggregation_zero_detections_and_exports(tmp_path: Path) -> None:
    cfg = _make_match_config()
    cfg.possession.min_control_frames = 2

    # 1. Zero-detection segment test
    empty_engine = PossessionEngine(cfg, fps=25.0, frame_stride=2, total_video_duration_seconds=120.0)
    for i in range(5):
        empty_engine.step_frame(i * 2, i * 0.08, [], [], frame_shape=(1080, 1920))
    empty_sum = empty_engine.summarize()
    assert empty_sum["attributed_possession_share"]["team_a_pct"] is None
    assert empty_sum["total_analysed_time_distribution_pct"]["UNKNOWN"] == 100.0
    assert empty_sum["segment_scope"] == "partial_video_or_highlight_segment"

    # 2. Controlled sequence: 4 frames Team A (0.32s), 3 frames Team B (within valid movement gate)
    engine = PossessionEngine(cfg, fps=25.0, frame_stride=2, total_video_duration_seconds=60.0)

    for i in range(4):
        fno = i * 2
        ts = round(i * 0.08, 3)
        b = [DetectionRecord(fno, ts, 32, "ball", (195, 290, 205, 300), 0.85)]
        o = [_make_player_obs(fno, ts, 10, InternalLabel.TEAM_A.value, (180, 200, 220, 300))]
        engine.step_frame(fno, ts, o, b, frame_shape=(1080, 1920))

    for i in range(4, 8):
        fno = i * 2
        ts = round(i * 0.08, 3)
        b = [DetectionRecord(fno, ts, 32, "ball", (295, 290, 305, 300), 0.85)]
        o = [_make_player_obs(fno, ts, 22, InternalLabel.TEAM_B.value, (280, 200, 320, 300))]
        engine.step_frame(fno, ts, o, b, frame_shape=(1080, 1920))

    summary = engine.summarize()
    # Team A controlled frames 2, 4, 6 (3 frames = 0.24s); Team B controlled frames 10, 12, 14 (or 12, 14 = 0.16s+)
    assert summary["durations_seconds"]["team_a_controlled"] == pytest.approx(0.24, abs=0.01)
    assert summary["durations_seconds"]["team_b_controlled"] >= 0.16
    assert 50.0 <= summary["attributed_possession_share"]["team_a_pct"] <= 65.0
    assert 35.0 <= summary["attributed_possession_share"]["team_b_pct"] <= 50.0

    # Export CSVs and Markdown/JSON report
    b_csv = export_ball_trajectory_records(tmp_path / "ball.csv", engine.ball_trajectory)
    o_csv = export_player_ownership_records(tmp_path / "own.csv", engine.player_proximities)
    t_csv = export_possession_timeline_records(tmp_path / "time.csv", engine.timeline)
    assert b_csv.is_file() and o_csv.is_file() and t_csv.is_file()

    # Validate against manual possession intervals
    store = PossessionAnnotationStore(tmp_path / "poss_ann", "test_match")
    store.set_manual_intervals(
        [
            PossessionIntervalAnnotation("test_match", 0, 6, 0.0, 0.24, PossessionState.TEAM_A_CONTROL.value, False, True, "manual"),
            PossessionIntervalAnnotation("test_match", 8, 14, 0.32, 0.56, PossessionState.TEAM_B_CONTROL.value, False, True, "manual"),
        ]
    )
    val_res = evaluate_possession_predictions(engine.timeline, store.get_verified_intervals(), cfg)
    assert val_res["metrics_available"] is True
    assert val_res["evaluated_frames"] == 8
    assert val_res["state_accuracy"] >= 0.60

    rep_files = generate_possession_report(tmp_path / "out", summary, val_res)
    assert rep_files["possession_summary_json"].is_file()
    assert rep_files["possession_report_md"].is_file()


def test_static_isolated_false_positive_ball_rejection_and_blacklisting() -> None:
    """
    Regression test for Sprint 2 Chelsea vs Sunderland failure mode:
    A stationary touchline object (cone/spare ball/logo) far from all players must be
    detected as a static clutter lock, broken within 6 frames, and blacklisted.
    """
    tracker = KalmanBallTracker("chelsea_sunderland", BallTrackingConfig(max_coast_frames=3))
    far_players = [
        DetectionRecord(i, i * 0.08, 0, "player", (600, 300, 640, 400), 0.88, tracking_id=5)
        for i in range(1, 9)
    ]

    last_rec = None
    for i in range(1, 8):
        static_fp = [DetectionRecord(i, i * 0.08, 32, "ball", (157, 313, 173, 325), 0.25)]
        last_rec = tracker.step(
            frame_number=i,
            timestamp=i * 0.08,
            ball_candidates=static_fp,
            player_detections=far_players,
        )

    assert last_rec is not None
    assert last_rec.observation_mode == BallObservationMode.LOST.value
    summary = tracker.summarize()
    assert summary["false_track_associations"] >= 1
    assert summary["blacklisted_static_zones_count"] >= 1

    # Subsequent candidate at the same static coordinate is rejected by filter_ball_candidates
    filtered = filter_ball_candidates(
        [DetectionRecord(9, 0.72, 32, "ball", (157, 313, 173, 325), 0.30)],
        far_players,
        frame_shape=(720, 1280),
        blacklisted_zones=tracker.blacklisted_static_zones,
    )
    assert len(filtered) == 0


def test_closeup_camera_cut_and_non_tactical_frame_rejection() -> None:
    """
    Regression test for Sprint 2 Norway vs Wales frame 1640 failure mode:
    A cut from wide-angle play (player height ~100px) to a full-screen player close-up
    (player height ~705px in 720p frame) must trigger a camera cut and classify ownership as UNKNOWN.
    """
    frm_wide = np.full((720, 1280, 3), (40, 160, 50), dtype=np.uint8)
    # Same colour histogram but 6.7x jump in median player height -> camera cut detected
    assert (
        detect_camera_cut(
            frm_wide,
            frm_wide,
            prev_median_player_h=105.0,
            curr_median_player_h=705.0,
        )
        is True
    )

    # Close-up player observation occupying almost full frame height (705px of 720px)
    closeup_obs = _make_player_obs(
        1640, 65.6, 29, InternalLabel.TEAM_B.value, (248, 4, 1037, 709)
    )
    tracker = KalmanBallTracker("norway_wales")
    ball_rec = tracker.step(
        1640,
        65.6,
        [DetectionRecord(1640, 65.6, 32, "ball", (600, 680, 620, 700), 0.45)],
    )
    _, raw_info = evaluate_frame_ownership_candidates(
        [closeup_obs],
        ball_rec,
        config=PossessionConfig(),
        frame_shape=(720, 1280),
    )
    assert raw_info["raw_state"] == PossessionState.UNKNOWN.value
    assert raw_info["controlling_team_id"] is None
    assert raw_info["reason"] == "non_tactical_closeup_frame"


def test_overlapping_possession_intervals_rejected_and_annotator_exports(tmp_path: Path) -> None:
    """Verify overlapping possession intervals raise ValueError and HTML annotators generate cleanly."""
    store = PossessionAnnotationStore(tmp_path / "poss_ann", "norway_wales")
    overlapping = [
        PossessionIntervalAnnotation("norway_wales", 0, 20, 0.0, 0.8, PossessionState.TEAM_A_CONTROL.value, False, True, "manual"),
        PossessionIntervalAnnotation("norway_wales", 20, 40, 0.8, 1.6, PossessionState.TEAM_B_CONTROL.value, False, True, "manual"),
    ]
    with pytest.raises(ValueError, match="Overlapping possession interval"):
        validate_no_overlapping_intervals(overlapping)

    with pytest.raises(ValueError, match="Overlapping possession interval"):
        store.set_manual_intervals(overlapping)

    html_poss = store.generate_html_annotator(match_config=_make_match_config(), fps=25.0)
    assert html_poss.is_file()
    assert "Possession Interval Ground-Truth Annotator" in html_poss.read_text(encoding="utf-8")

    sampled_fnos = sample_representative_frame_numbers(2953, num_samples=45, mandatory_frames=[400, 500])
    assert len(sampled_fnos) >= 45
    assert 400 in sampled_fnos and 500 in sampled_fnos

    det_store = DetectionAnnotationStore(tmp_path / "det_ann", "norway_wales")
    html_det = det_store.generate_html_annotator()
    assert html_det.is_file()
    assert "Detection Ground-Truth Annotator" in html_det.read_text(encoding="utf-8")

