"""Comprehensive Sprint 3.1 unit and regression tests for team discovery, track-level
aggregation, ball false-positive filtering, set-piece safety, Kalman confirmation/decay,
within-shot tracking metrics, possession reliability, and evaluation harness accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    TrackAppearanceBuffer,
    collect_track_appearances,
    evaluate_discovered_labels,
    filter_discovery_player_indices,
    is_tactical_discovery_frame,
)
from src.config import BallTrackingConfig, MatchConfig, PossessionConfig, TeamDiscoveryConfig, TrackingConfig
from src.detection.annotation_workflow import AnnotatedBox, FrameAnnotation
from src.evaluation.detection_eval import evaluate_detections_against_ground_truth
from src.evaluation.possession_eval import PossessionIntervalAnnotation, evaluate_possession_predictions
from src.evaluation.tracking_eval import (
    evaluate_ball_tracking_trajectory,
    evaluate_within_shot_player_tracking,
)
from src.possession.ball_tracker import (
    KalmanBallTracker,
    estimate_pitch_horizon_y,
    filter_ball_candidates,
)
from src.possession.ownership import evaluate_frame_ownership_candidates
from src.possession.possession_engine import PossessionEngine
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    ClassificationPrediction,
    DetectionRecord,
    FrameObservationRecord,
    InternalLabel,
    PossessionState,
)
from src.tracking.stabilizer import TemporalTeamStabilizer
from tests.conftest import make_synthetic_kit_crop


def _obs(tid: int, frame: int, crop: np.ndarray, shot_id: int = 0) -> TeamDiscoveryObservation:
    return TeamDiscoveryObservation(
        tracking_id=tid,
        frame_number=frame,
        timestamp=frame / 25.0,
        crop_bgr=crop,
        blur_score=50.0,
        grass_ratio=0.08,
        occlusion_ratio=0.05,
        shot_id=shot_id,
    )


@pytest.mark.parametrize("clustering_method", ["kmeans", "gmm", "agglomerative", "dbscan", "hdbscan"])
def test_unsupervised_clustering_algorithms_and_multi_outlier_rejection(clustering_method: str) -> None:
    """All supported clustering backends must separate two teams and reject referees + goalkeepers."""
    observations = []
    # Team A: Red shirts (10 tracks)
    for tid in range(1, 11):
        crop = make_synthetic_kit_crop((20, 20, 215), (240, 240, 240), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 6, 12, 18))
    # Team B: White shirts (10 tracks)
    for tid in range(101, 111):
        crop = make_synthetic_kit_crop((235, 235, 235), (200, 200, 200), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 6, 12, 18))
    # Referee: Black shirts (3 tracks)
    for tid in range(201, 204):
        crop = make_synthetic_kit_crop((22, 22, 22), (35, 35, 35), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 6, 12, 18))
    # Goalkeeper: Bright Yellow shirts (2 tracks)
    for tid in range(301, 303):
        crop = make_synthetic_kit_crop((20, 230, 230), (20, 20, 20), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 6, 12, 18))

    cfg = TeamDiscoveryConfig(
        enabled=True,
        feature_method="hsv_lab",
        clustering_method=clustering_method,
        candidate_clusters=3,
        auto_select_k=True,
        max_candidate_clusters=4,
        min_track_observations=3,
        random_state=42,
    )
    apps = collect_track_appearances(observations, cfg)
    disc = AutomaticTeamDiscoverer(cfg).fit(apps)

    gt = [
        InternalLabel.TEAM_A.value if a.tracking_id < 100
        else InternalLabel.TEAM_B.value if a.tracking_id < 200
        else InternalLabel.REFEREE.value if a.tracking_id < 300
        else InternalLabel.UNKNOWN.value
        for a in apps
    ]
    preds = [disc.predict_track(a.tracking_id)[0] for a in apps]
    ev = evaluate_discovered_labels(preds, gt)

    assert ev["macro_f1_on_covered"] >= 0.95
    assert ev["team_coverage"] >= 0.90
    assert ev["referee_misclassification_rate"] == 0.0
    assert ev["unknown_misclassification_rate"] == 0.0

    # Verify full ClassificationPrediction telemetry (real second_nearest_distance & margin)
    sample_pred = disc.predict_classification(make_synthetic_kit_crop((20, 20, 215), (240, 240, 240), seed=999))
    assert not sample_pred.is_unknown
    assert sample_pred.second_nearest_distance > sample_pred.nearest_distance
    assert sample_pred.margin > 0.05
    assert sample_pred.confidence_score > 0.5


def test_similar_kit_colours_and_ambiguous_crop_rejection() -> None:
    """Similar red vs burgundy kits must separate cleanly while intermediate ambiguous crops become Unknown."""
    observations = []
    for tid in range(1, 9):
        crop = make_synthetic_kit_crop((20, 20, 225), (30, 30, 195), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 8, 16))
    for tid in range(51, 59):
        crop = make_synthetic_kit_crop((110, 25, 130), (95, 20, 110), seed=tid)
        observations.extend(_obs(tid, f, crop) for f in (0, 8, 16))

    cfg = TeamDiscoveryConfig(enabled=True, feature_method="hsv_lab", clustering_method="kmeans", random_state=42)
    apps = collect_track_appearances(observations, cfg)
    disc = AutomaticTeamDiscoverer(cfg).fit(apps)

    lbl_a, _ = disc.predict_track(1)
    lbl_b, _ = disc.predict_track(51)
    assert {lbl_a, lbl_b} == {InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value}

    # Unrelated bright cyan goalkeeper crop must be rejected as Unknown
    cyan_gk = make_synthetic_kit_crop((235, 220, 20), (20, 20, 20), seed=777)
    pred_gk = disc.predict_classification(cyan_gk)
    assert pred_gk.predicted_label == InternalLabel.UNKNOWN.value
    assert pred_gk.is_unknown is True
    assert pred_gk.rejection_reason is not None


def test_track_appearance_buffer_resists_single_noisy_crop_and_resets_on_camera_cut() -> None:
    """Track-level online aggregation prevents one bad crop from flipping team and isolates shots."""
    red = make_synthetic_kit_crop((20, 20, 220), (240, 240, 240), seed=1)
    blue = make_synthetic_kit_crop((220, 50, 20), (240, 240, 240), seed=2)
    observations = [
        *[_obs(t, f, make_synthetic_kit_crop((20, 20, 220), (240, 240, 240), seed=t)) for t in range(1, 6) for f in (0, 4, 8)],
        *[_obs(t, f, make_synthetic_kit_crop((220, 50, 20), (240, 240, 240), seed=t)) for t in range(11, 16) for f in (0, 4, 8)],
    ]
    cfg = TeamDiscoveryConfig(enabled=True, feature_method="hsv_lab", min_track_observations=3, random_state=42)
    disc = AutomaticTeamDiscoverer(cfg).fit(collect_track_appearances(observations, cfg))
    red_label, _ = disc.predict_track(1)
    blue_label, _ = disc.predict_track(11)
    assert red_label != blue_label

    # Independent Pass-2 track ID 999 (does not exist in Pass 1!) accumulates 4 red crops, then 1 noisy blue crop
    buf = TrackAppearanceBuffer(disc, max_buffer_samples=8)
    bbox = (100, 100, 140, 200)
    for f in (2, 4, 6, 8):
        p = buf.classify_observation(999, f, bbox, red)
        assert p.predicted_label == red_label

    # One contaminated/occluded crop of the opposing colour must NOT flip the track median
    p_noisy = buf.classify_observation(999, 10, bbox, blue)
    assert p_noisy.predicted_label == red_label

    # After a camera cut, track ID 999 starts fresh with zero history leakage from the prior shot
    buf.on_camera_cut()
    p_after_cut = buf.classify_observation(999, 12, bbox, blue)
    assert p_after_cut.predicted_label == blue_label


def test_shot_segmented_track_collection_and_tactical_frame_filtering() -> None:
    """Shot IDs separate reused track IDs across cuts, and close-up/crowd filters reject garbage."""
    red = make_synthetic_kit_crop((20, 20, 220), seed=1)
    blue = make_synthetic_kit_crop((220, 50, 20), seed=2)
    obs = [
        *[_obs(7, f, red, shot_id=0) for f in (0, 4, 8)],
        *[_obs(7, f, blue, shot_id=1) for f in (20, 24, 28)],
    ]
    apps = collect_track_appearances(obs, TeamDiscoveryConfig(min_track_observations=3))
    assert len(apps) == 2
    assert {a.shot_id for a in apps} == {0, 1}

    # Non-tactical close-up frame rejected; tactical wide frame accepted & crowd outlier filtered
    assert is_tactical_discovery_frame([(200, 10, 900, 700)], (720, 1280)) is False
    wide_boxes = [
        (100, 250, 140, 340),  # normal player h=90
        (300, 260, 340, 350),  # normal player h=90
        (500, 255, 540, 345),  # normal player h=90
        (700, 270, 740, 360),  # normal player h=90
        (200, 20, 225, 80),    # spectator in upper stands (y2=80 < 0.14*720=100.8)
        (900, 100, 1020, 420), # foreground giant outlier h=320 > 2.4*90
    ]
    assert is_tactical_discovery_frame(wide_boxes, (720, 1280)) is True
    kept_idxs = filter_discovery_player_indices(wide_boxes, (720, 1280))
    assert kept_idxs == [0, 1, 2, 3]


def test_reproducible_unseen_match_discovery_determinism() -> None:
    """Two runs with the same random_state produce identical labels and distances."""
    observations = [
        *[_obs(t, f, make_synthetic_kit_crop((20, 20, 220), seed=t)) for t in range(1, 7) for f in (0, 4, 8)],
        *[_obs(t, f, make_synthetic_kit_crop((235, 235, 235), seed=t)) for t in range(21, 27) for f in (0, 4, 8)],
    ]
    cfg = MatchConfig(
        match_id="unseen_match",
        match_name="Unseen Match",
        video_path="dummy.mp4",
        team_discovery=TeamDiscoveryConfig(enabled=True, random_state=42),
    )
    apps = collect_track_appearances(observations, cfg.team_discovery)
    d1 = AutomaticTeamDiscoverer(cfg.team_discovery).fit(apps)
    d2 = AutomaticTeamDiscoverer(cfg.team_discovery).fit(apps)
    assert d1.track_labels_ == d2.track_labels_
    assert d1.track_distances_ == d2.track_distances_


def test_ball_false_positive_regression_crowd_steward_pure_turf_and_white_sock() -> None:
    """Regression tests for Sprint 3 Man Utd vs Aston Villa real ball failure modes."""
    # Construct synthetic elevated pitch frame (720x1280): top 0..170 = dark crowd, 175..720 = green pitch
    frm = np.full((720, 1280, 3), (35, 35, 45), dtype=np.uint8)
    frm[175:, :] = (40, 160, 55)  # green pitch

    # Draw a neon-yellow high-vis steward blob in the crowd at (883, 122, 897, 134)
    frm[120:136, 881:899] = (20, 240, 235)
    # Draw a crisp white football on the pitch at (431, 316, 443, 328)
    cv2.circle(frm, (437, 322), 6, (245, 245, 245), -1)

    horizon = estimate_pitch_horizon_y(frm)
    assert horizon is not None and 155 <= horizon <= 200

    players = [
        DetectionRecord(10, 0.4, 0, "player", (500, 260, 535, 350), 0.90, 1),  # h=90
        DetectionRecord(10, 0.4, 0, "player", (700, 260, 735, 350), 0.90, 2),
    ]

    steward_fp = DetectionRecord(10, 0.4, 32, "ball", (883, 122, 897, 134), 0.74)
    pure_turf_fp = DetectionRecord(10, 0.4, 32, "ball", (944, 374, 954, 384), 0.21)
    white_sock_fp = DetectionRecord(10, 0.4, 32, "ball", (512, 325, 522, 335), 0.20)  # inside player #1 shin
    true_ball = DetectionRecord(10, 0.4, 32, "ball", (431, 316, 443, 328), 0.55)

    filtered = filter_ball_candidates(
        [steward_fp, pure_turf_fp, white_sock_fp, true_ball],
        players,
        frm.shape[:2],
        frame_bgr=frm,
        predicted_pos=None,
        filter_crowd_horizon=True,
    )
    assert len(filtered) == 1
    assert filtered[0].bbox == true_ball.bbox

    # A real lofted white aerial ball above the horizon connected to an active Kalman prediction IS kept
    frm[110:124, 600:614] = (245, 245, 245)
    aerial_ball = DetectionRecord(12, 0.48, 32, "ball", (601, 111, 613, 123), 0.52)
    kept_aerial = filter_ball_candidates(
        [aerial_ball],
        players,
        frm.shape[:2],
        frame_bgr=frm,
        predicted_pos=(605.0, 135.0),
        filter_crowd_horizon=True,
    )
    assert len(kept_aerial) == 1


def test_kalman_low_conf_init_confirmation_set_piece_release_and_cut_cooldown() -> None:
    """Verify:
    1. Single-frame weak FP cannot hijack a LOST tracker (`init_confirm_frames=2`).
    2. Static clutter blacklist releases immediately when a player approaches for a set piece.
    3. Consecutive broadcast wipe frames within cooldown do not double-count camera cuts.
    """
    cfg = BallTrackingConfig(
        max_coast_frames=3,
        init_confirm_frames=2,
        init_high_conf_threshold=0.45,
        camera_cut_cooldown_frames=6,
    )
    tracker = KalmanBallTracker("m1", cfg)

    # 1. Single weak phantom (conf=0.20) must remain LOST until confirmed on next frame
    r1 = tracker.step(2, 0.08, [DetectionRecord(2, 0.08, 32, "ball", (900, 400, 912, 412), 0.20)])
    assert r1.observation_mode == BallObservationMode.LOST.value
    # Second consistent weak observation confirms the real ball
    r2 = tracker.step(4, 0.16, [DetectionRecord(4, 0.16, 32, "ball", (910, 402, 922, 414), 0.22)])
    assert r2.observation_mode == BallObservationMode.OBSERVED.value

    # 2. Set-piece safety: blacklist a static isolated spot, then move a player up to take the free kick
    far_player = [DetectionRecord(10, 0.4, 0, "player", (100, 200, 140, 300), 0.90, 1)]
    for i in range(6):
        fno = 10 + i * 2
        tracker.step(
            fno,
            fno / 25.0,
            [DetectionRecord(fno, fno / 25.0, 32, "ball", (600, 400, 614, 414), 0.55)],
            player_detections=far_player,
        )
    assert len(tracker.blacklisted_static_zones) == 1

    # Free-kick taker walks up right next to (607, 407) -> zone must be released so set piece is tracked!
    taker = [DetectionRecord(30, 1.2, 0, "player", (585, 310, 625, 410), 0.92, 9)]
    r_setpiece = tracker.step(
        30,
        1.2,
        [DetectionRecord(30, 1.2, 32, "ball", (600, 400, 614, 414), 0.65)],
        player_detections=taker,
    )
    assert len(tracker.blacklisted_static_zones) == 0
    assert r_setpiece.observation_mode == BallObservationMode.OBSERVED.value

    # 3. Camera cut wipe cooldown: consecutive abrupt frames 2 frames apart count as ONE cut
    f_a = np.full((180, 320, 3), (40, 170, 50), dtype=np.uint8)
    f_wipe = np.full((180, 320, 3), (230, 30, 30), dtype=np.uint8)
    f_b = np.full((180, 320, 3), (30, 30, 230), dtype=np.uint8)
    tracker.step(40, 1.60, [], frame_bgr=f_a)
    rc1 = tracker.step(42, 1.68, [], frame_bgr=f_wipe)
    rc2 = tracker.step(44, 1.76, [], frame_bgr=f_b)
    assert rc1.camera_cut_reset is True
    assert rc2.camera_cut_reset is False


def test_stabilizer_camera_cut_reset_and_within_shot_fragmentation_metrics() -> None:
    """Verify TemporalTeamStabilizer resets on cuts and within-shot metrics separate cuts from leaks."""
    stab = TemporalTeamStabilizer(TrackingConfig(min_votes=2, lock_min_observations=3))
    pred_a = ClassificationPrediction("Team_A", "Team_A", 0.08, 0.55, 0.47, 0.90, False)
    for f in (2, 4, 6):
        lbl, _, locked = stab.update(5, f, (100, 200, 140, 300), pred_a)
    assert lbl == "Team_A" and locked is True

    stab.on_camera_cut()
    pred_b = ClassificationPrediction("Team_B", "Team_B", 0.09, 0.52, 0.43, 0.88, False)
    lbl_after, _, locked_after = stab.update(5, 8, (100, 200, 140, 300), pred_b)
    assert lbl_after == "Unknown" and locked_after is False  # cleared history requires min_votes again

    # Within-shot tracking evaluation across 2 shots with 1 deliberate cross-cut leak (track 99)
    obs = []
    for f in range(0, 20, 2):  # Shot 0 (10 frames)
        obs.append(FrameObservationRecord("m", f, f / 25.0, 1, "Team_A", "A", "Team_A", (10, 10, 40, 90), 0.9, 0.9, 0.1, 0.9, True))
        obs.append(FrameObservationRecord("m", f, f / 25.0, 99, "Team_B", "B", "Team_B", (80, 10, 110, 90), 0.9, 0.9, 0.1, 0.9, True))
    for f in range(20, 40, 2):  # Shot 1 (10 frames, cut at frame 20)
        obs.append(FrameObservationRecord("m", f, f / 25.0, 2, "Team_A", "A", "Team_A", (10, 10, 40, 90), 0.9, 0.9, 0.1, 0.9, True))
        obs.append(FrameObservationRecord("m", f, f / 25.0, 99, "Team_B", "B", "Team_B", (80, 10, 110, 90), 0.9, 0.9, 0.1, 0.9, True))

    ws = evaluate_within_shot_player_tracking(obs, camera_cut_frames=[20], min_shot_frames=10)
    assert ws["total_shots"] == 2
    assert ws["cross_cut_id_leaks"] == 1
    assert ws["within_shot_mean_ids_per_player_slot"] == pytest.approx(1.0, abs=0.01)
    assert ws["within_shot_team_label_consistency"] == 1.0
    assert ws["formal_identity_metrics"]["metrics_available"] is False


def test_ownership_does_not_override_stabilizer_rejected_unknown_and_harness_metrics() -> None:
    """Ensure ownership never overrides a stabilizer-rejected Unknown and harness computes all Section 8 metrics."""
    ball_rec = BallTrajectoryRecord(
        "m1", 10, 0.4, BallObservationMode.OBSERVED.value, True, False,
        (194, 290, 206, 302), 200.0, 296.0, 0.0, 0.0, 0.0, 0.85, 5.0, 0
    )
    # Track has non-zero track_team_confidence (i.e., evaluated by stabilizer and rejected as ambiguous/Unknown)
    ambiguous_obs = FrameObservationRecord(
        "m1", 10, 0.4, 7, InternalLabel.UNKNOWN.value, "Unknown", InternalLabel.TEAM_A.value,
        (180, 200, 220, 300), 0.92, 0.75, 0.18, 0.45, False
    )
    _, raw = evaluate_frame_ownership_candidates([ambiguous_obs], ball_rec, PossessionConfig())
    assert raw["raw_state"] == PossessionState.UNKNOWN.value
    assert raw["controlling_team_id"] is None

    # Verify enhanced detection evaluation (center error, FP rate, miss rate)
    ann = FrameAnnotation(
        "m1", 10, 0.4, "f.jpg", verified=True, source="manual",
        boxes=[AnnotatedBox("ball", [100, 100, 116, 116])]
    )
    preds = {
        10: [
            DetectionRecord(10, 0.4, 32, "ball", (102, 100, 118, 116), 0.90),  # 2px center error
            DetectionRecord(10, 0.4, 32, "ball", (500, 500, 516, 516), 0.30),  # FP
        ]
    }
    det_res = evaluate_detections_against_ground_truth(preds, [ann], iou_threshold=0.50)
    assert det_res["ball"]["mean_center_error_px"] == pytest.approx(2.0, abs=0.01)
    assert det_res["ball"]["false_positive_rate"] == pytest.approx(0.5, abs=0.01)
    assert det_res["ball"]["miss_rate"] == 0.0

    # Verify ball trajectory evaluation against GT boxes
    traj = [
        BallTrajectoryRecord("m1", 10, 0.4, "observed", True, False, (102, 100, 118, 116), 110.0, 108.0, 0, 0, 0, 0.9, 5, 0),
    ]
    bt_res = evaluate_ball_tracking_trajectory(traj, verified_annotations=[ann], match_distance_px=15.0)
    assert bt_res["metrics_available"] is True
    assert bt_res["ground_truth_metrics"]["observed_mean_error_px"] == pytest.approx(2.0, abs=0.01)
    assert bt_res["ground_truth_metrics"]["false_associations"] == 0

    # Verify possession evaluation computes incorrect_confident_attribution_rate & swap count
    cfg = MatchConfig("m1", "M1", "d.mp4")
    engine = PossessionEngine(cfg, fps=25.0, frame_stride=2)
    for i in range(4):
        fno = i * 2
        b = [DetectionRecord(fno, fno / 25.0, 32, "ball", (194, 290, 206, 302), 0.85)]
        o = [FrameObservationRecord("m1", fno, fno / 25.0, 1, "Team_A", "Team A", "Team_A", (180, 200, 220, 300), 0.9, 0.9, 0.1, 0.9, True)]
        engine.step_frame(fno, fno / 25.0, o, b, frame_shape=(720, 1280))
    gt_ivs = [
        PossessionIntervalAnnotation("m1", 0, 6, 0.0, 0.24, PossessionState.TEAM_B_CONTROL.value, False, True, "manual"),
    ]
    pe = evaluate_possession_predictions(engine.timeline, gt_ivs, cfg)
    assert pe["metrics_available"] is True
    assert pe["incorrect_confident_attribution_rate"] == 1.0
    assert pe["team_ownership_swap_frames"] >= 2

