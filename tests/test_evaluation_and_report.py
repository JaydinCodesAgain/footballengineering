"""Tests for IoU matching, AP/mAP calculation, detection provenance checks, and report generation."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.detection.annotation_workflow import AnnotatedBox, DetectionAnnotationStore, FrameAnnotation
from src.evaluation.detection_eval import (
    compute_average_precision,
    compute_iou,
    evaluate_detections_against_ground_truth,
    match_detections_one_to_one,
)
from src.evaluation.report import export_possession_foundation_records, generate_consolidated_report
from src.schemas import DetectionRecord, FrameObservationRecord


def test_compute_iou_and_one_to_one_matching() -> None:
    assert pytest.approx(compute_iou([0, 0, 10, 10], [0, 0, 10, 10])) == 1.0
    assert pytest.approx(compute_iou([0, 0, 10, 10], [20, 20, 30, 30])) == 0.0
    # 5x10 intersection = 50; union = 100 + 100 - 50 = 150 -> IoU = 1/3
    assert pytest.approx(compute_iou([0, 0, 10, 10], [5, 0, 15, 10])) == 1.0 / 3.0

    # Two overlapping predictions for a single ground-truth box: only the higher-confidence one matches!
    gt_boxes = [[10, 10, 50, 90]]
    pred_boxes = [[10, 10, 48, 88], [11, 11, 50, 90], [200, 200, 240, 280]]
    confs = [0.65, 0.92, 0.80]

    matches, fp_idxs, fn_idxs = match_detections_one_to_one(
        pred_boxes, confs, gt_boxes, iou_threshold=0.50
    )
    assert len(matches) == 1
    assert matches[0][0] == 1  # Index 1 has higher confidence (0.92)
    assert matches[0][1] == 0
    assert sorted(fp_idxs) == [0, 2]
    assert fn_idxs == []


def test_detection_evaluation_provenance_and_metrics(tmp_path: Path) -> None:
    # Unverified annotations must report metrics_available=False
    unverified_ann = FrameAnnotation(
        match_id="m1",
        frame_number=10,
        timestamp=0.4,
        image_rel_path="frames/frame_000010.jpg",
        verified=False,
        source="unlabelled",
        boxes=[],
    )
    res_unavail = evaluate_detections_against_ground_truth({}, [unverified_ann])
    assert res_unavail["metrics_available"] is False

    # Verified manual annotations compute exact precision, recall, F1, and AP
    verified_ann = FrameAnnotation(
        match_id="m1",
        frame_number=10,
        timestamp=0.4,
        image_rel_path="frames/frame_000010.jpg",
        verified=True,
        source="manual",
        boxes=[
            AnnotatedBox(class_name="player", bbox=[10, 10, 50, 100]),
            AnnotatedBox(class_name="player", bbox=[100, 20, 140, 110]),
            AnnotatedBox(class_name="ball", bbox=[300, 300, 320, 320]),
        ],
    )
    preds = {
        10: [
            DetectionRecord(10, 0.4, 0, "player", (11, 10, 50, 99), 0.95, 1),
            DetectionRecord(10, 0.4, 0, "player", (400, 400, 440, 490), 0.60, 2),  # FP player
            DetectionRecord(10, 0.4, 32, "ball", (301, 301, 319, 319), 0.85, None),  # TP ball
        ]
    }
    res = evaluate_detections_against_ground_truth(preds, [verified_ann], iou_threshold=0.50)
    assert res["metrics_available"] is True
    assert res["player"]["tp"] == 1
    assert res["player"]["fp"] == 1
    assert res["player"]["fn"] == 1
    assert pytest.approx(res["player"]["precision"]) == 0.5
    assert pytest.approx(res["player"]["recall"]) == 0.5
    assert res["ball"]["tp"] == 1
    assert res["ball"]["fp"] == 0
    assert res["ball"]["fn"] == 0
    assert pytest.approx(res["ball"]["f1"]) == 1.0

    # Verify annotation store rejects model-predicted provenance
    store = DetectionAnnotationStore(tmp_path / "ann", "m1")
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "frame_number,timestamp,class_name,x1,y1,x2,y2,verified,source\n"
        "10,0.4,player,10,10,50,100,true,model_prediction\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be imported"):
        store.import_boxes_from_csv(bad_csv)


def test_report_and_possession_foundation_export(tmp_path: Path) -> None:
    obs = [
        FrameObservationRecord(
            match_id="norway_wales",
            frame_number=2,
            timestamp=0.08,
            player_id=7,
            team_id="Team_A",
            team_display_name="Norway",
            raw_team_id="Team_A",
            player_bbox=(100, 120, 145, 230),
            player_detection_confidence=0.91,
            classification_confidence=0.84,
            classification_distance=0.14,
            track_team_confidence=0.88,
            is_team_locked=True,
            ball_bbox=(150, 210, 166, 226),
            ball_detection_confidence=0.77,
        )
    ]
    csv_path = export_possession_foundation_records(tmp_path / "obs.csv", obs)
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "Norway" in content
    assert "0.77" in content

    files = generate_consolidated_report(
        output_dir=tmp_path / "report",
        match_id="norway_wales",
        match_name="Norway vs Wales",
        detection_summary={"total_frames_processed": 100, "avg_players_per_frame": 11.2, "frames_with_ball": 72, "ball_coverage_pct": 72.0},
        classification_summary={"feature_method": "spatial_embedding", "classifier_type": "knn", "overall_accuracy": 0.9333, "macro_f1": 0.92, "coverage": 0.9333, "unknown_rejection_rate": 0.0667, "per_class": {}},
        tracking_summary={"distinct_tracking_ids": 28, "short_tracks_under_5_obs": 3, "mean_track_observations": 24.5, "total_player_observations": 686, "raw_team_label_changes": 18, "smoothed_team_label_changes": 1, "stable_label_coverage": 0.91},
    )
    assert files["report_json"].is_file()
    assert files["report_md"].is_file()


def test_report_marks_unavailable_unsupervised_metrics_without_formatting_failure(tmp_path: Path) -> None:
    files = generate_consolidated_report(
        output_dir=tmp_path,
        match_id="unseen_match",
        match_name="Unseen Match",
        detection_summary={"metrics_available": False},
        classification_summary={
            "mode": "unsupervised_per_video_adaptation",
            "feature_method": "spatial_embedding",
            "classifier_type": "kmeans",
            "overall_accuracy": None,
            "selective_accuracy": None,
            "macro_f1": None,
            "positive_macro_f1": None,
            "coverage": None,
            "unknown_rejection_rate": None,
            "per_class": {},
        },
        tracking_summary={},
    )
    report = files["report_md"].read_text(encoding="utf-8")
    assert "Held-Out Validation Accuracy:** Unavailable" in report
    assert "Classification Coverage:** Unavailable" in report

