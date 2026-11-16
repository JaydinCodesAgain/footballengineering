"""Integration test covering dataset loading, classifier training, tracking, possession engine, and report generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.classification.classifier import TeamClassifier
from src.config import load_match_config
from src.evaluation.report import generate_consolidated_report, generate_possession_report
from src.possession.possession_engine import PossessionEngine
from src.schemas import DetectionRecord, FrameObservationRecord, InternalLabel
from tests.conftest import make_synthetic_kit_crop


def test_end_to_end_integration_classifier_tracking_possession_and_reports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = load_match_config(root / "configs" / "norway_wales.yaml")

    # 1. Train classifier on synthetic crops
    crops_a = [make_synthetic_kit_crop((20, 20, 220), (240, 240, 240), seed=i) for i in range(6)]
    crops_b = [make_synthetic_kit_crop((15, 220, 240), (30, 30, 30), seed=20 + i) for i in range(6)]
    crops_ref = [make_synthetic_kit_crop((230, 200, 20), (20, 20, 20), seed=40 + i) for i in range(4)]

    clf = TeamClassifier(cfg.match_id, cfg.classification)
    clf.fit(
        crops_a + crops_b + crops_ref,
        [InternalLabel.TEAM_A.value] * 6
        + [InternalLabel.TEAM_B.value] * 6
        + [InternalLabel.REFEREE.value] * 4,
    )
    assert clf.is_fitted

    # 2. Run PossessionEngine across a 12-frame sequence with observed, occluded (interpolated), and contested states
    engine = PossessionEngine(cfg, fps=25.0, frame_stride=2, total_video_duration_seconds=90.0)

    for step_idx in range(12):
        fno = step_idx * 2
        ts = round(step_idx * 0.08, 3)

        p_a_pred = clf.predict_crop(crops_a[step_idx % 6], match_id=cfg.match_id)
        p_b_pred = clf.predict_crop(crops_b[step_idx % 6], match_id=cfg.match_id)

        if step_idx < 5:
            # Team A dribbling rightwards
            bx = 200 + step_idx * 10
            ball_dets = [DetectionRecord(fno, ts, 32, "ball", (bx - 6, 290, bx + 6, 302), 0.86)]
            obs = [
                FrameObservationRecord(
                    cfg.match_id, fno, ts, 9, p_a_pred.predicted_label, "Norway",
                    p_a_pred.predicted_label, (bx - 20, 200, bx + 20, 300), 0.92,
                    p_a_pred.confidence_score, p_a_pred.nearest_distance, 0.90, True,
                ),
                FrameObservationRecord(
                    cfg.match_id, fno, ts, 14, p_b_pred.predicted_label, "Wales",
                    p_b_pred.predicted_label, (700, 200, 740, 300), 0.91,
                    p_b_pred.confidence_score, p_b_pred.nearest_distance, 0.89, True,
                ),
            ]
        elif step_idx == 5:
            # 1-frame occlusion (no ball detection -> Kalman interpolation)
            bx = 250
            ball_dets = []
            obs = [
                FrameObservationRecord(
                    cfg.match_id, fno, ts, 9, p_a_pred.predicted_label, "Norway",
                    p_a_pred.predicted_label, (bx - 20, 200, bx + 20, 300), 0.92,
                    p_a_pred.confidence_score, p_a_pred.nearest_distance, 0.90, True,
                )
            ]
        else:
            # Team B controls at x=310
            bx = 310
            ball_dets = [DetectionRecord(fno, ts, 32, "ball", (bx - 6, 290, bx + 6, 302), 0.84)]
            obs = [
                FrameObservationRecord(
                    cfg.match_id, fno, ts, 14, p_b_pred.predicted_label, "Wales",
                    p_b_pred.predicted_label, (bx - 20, 200, bx + 20, 300), 0.91,
                    p_b_pred.confidence_score, p_b_pred.nearest_distance, 0.89, True,
                )
            ]

        engine.step_frame(fno, ts, obs, ball_dets, frame_shape=(1080, 1920))

    poss_summary = engine.summarize()
    assert poss_summary["ball_tracking"]["observed_frames"] == 11
    assert poss_summary["ball_tracking"]["interpolated_frames"] == 1
    assert poss_summary["durations_seconds"]["team_a_controlled"] > 0.0
    assert poss_summary["durations_seconds"]["team_b_controlled"] > 0.0
    assert poss_summary["attributed_possession_share"]["team_a_pct"] is not None

    poss_files = generate_possession_report(tmp_path, poss_summary)
    assert poss_files["possession_summary_json"].is_file()
    assert poss_files["possession_report_md"].is_file()

    rep_files = generate_consolidated_report(
        output_dir=tmp_path,
        match_id=cfg.match_id,
        match_name=cfg.match_name,
        detection_summary={"total_frames_processed": 12, "avg_players_per_frame": 1.4, "frames_with_ball": 11, "ball_coverage_pct": 91.67},
        classification_summary={"feature_method": "foreground_masked", "classifier_type": "knn", "overall_accuracy": 0.9333, "macro_f1": 0.9186, "coverage": 0.7333, "unknown_rejection_rate": 0.2667, "per_class": {}},
        tracking_summary={"distinct_tracking_ids": 2, "short_tracks_under_5_obs": 0, "mean_track_observations": 8.5, "total_player_observations": 17, "raw_team_label_changes": 0, "smoothed_team_label_changes": 0, "stable_label_coverage": 1.0},
        possession_summary=poss_summary,
    )
    assert rep_files["report_json"].is_file()
    assert rep_files["report_md"].is_file()

