"""End-to-end pipeline orchestration for dataset preparation, training, tracking, and reporting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import torch

from src.classification.classifier import TeamClassifier
from src.classification.cropper import extract_torso_crop
from src.classification.dataset import JerseyCropDataset, assign_temporal_and_track_splits
from src.classification.diagnostics import audit_dataset_labels
from src.config import MatchConfig, load_match_config
from src.detection.annotation_workflow import DetectionAnnotationStore
from src.detection.detector import FootballDetector
from src.evaluation.classification_eval import (
    evaluate_team_classifier,
    run_classification_benchmark,
)
from src.evaluation.detection_eval import (
    evaluate_detections_against_ground_truth,
    evaluate_legacy_manual_counts,
    run_confidence_threshold_sweep,
)
from src.evaluation.report import (
    export_possession_foundation_records,
    generate_consolidated_report,
    write_csv_rows,
)
from src.evaluation.tracking_eval import compare_tracking_algorithms
from src.pipeline.legacy_migration import (
    LEGACY_NORWAY_WALES_DETECTION_COUNTS,
    migrate_chelsea_sunderland_legacy_dataset,
    migrate_norway_wales_legacy_dataset,
)
from src.schemas import ClassificationPrediction, FrameObservationRecord, InternalLabel
from src.tracking.stabilizer import TemporalTeamStabilizer
from src.tracking.tracker import MultiObjectTracker
from src.visualization.plots import (
    save_confusion_matrix_plot,
    save_crop_reference_gallery,
    save_detections_timeline_plot,
    save_misclassification_gallery,
    save_side_by_side_confusion_matrices,
    save_threshold_calibration_plot,
)
from src.visualization.video_annotator import annotate_match_frame, finalize_mp4_video


def prepare_dataset_for_match(
    match_config: MatchConfig,
    base_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> JerseyCropDataset:
    """
    Prepare or load the jersey crop dataset for a match.
    For the two existing benchmark matches (`norway_wales` and `chelsea_sunderland`),
    migrates the existing labelled samples with full metadata if not already present.
    For any new match, extracts high-quality filtered crops across temporal splits.
    """
    root = base_dir or Path.cwd()
    dataset_dir = root / "data" / "crops" / match_config.match_id
    video_path = match_config.resolve_video_path(root)

    if match_config.match_id == "norway_wales":
        return migrate_norway_wales_legacy_dataset(
            video_path=video_path,
            dataset_dir=dataset_dir,
            match_config=match_config,
            force_rebuild=force_rebuild,
        )
    if match_config.match_id == "chelsea_sunderland":
        return migrate_chelsea_sunderland_legacy_dataset(
            video_path=video_path,
            dataset_dir=dataset_dir,
            match_config=match_config,
            force_rebuild=force_rebuild,
        )

    # Generic extraction pass for any new match
    ds = JerseyCropDataset(dataset_dir, match_config)
    if len(ds.records) > 0 and not force_rebuild:
        return ds

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open match video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    cap.release()

    tracker = MultiObjectTracker(
        detection_config=match_config.detection,
        tracking_config=match_config.tracking,
    )
    for fno, ts, frame_bgr, detections in tracker.track_video(video_path):
        if (fno // max(1, match_config.detection.frame_stride)) % 10 != 0:
            continue
        players = [d for d in detections if d.class_name == "player"]
        all_p_boxes = [d.bbox for d in players]
        for det in players[:4]:
            crop, tbox, qm = extract_torso_crop(
                frame_bgr, det.bbox, config=match_config.cropping, other_player_bboxes=all_p_boxes
            )
            if crop is not None and qm.is_valid:
                ds.add_crop(
                    crop_bgr=crop,
                    frame_number=fno,
                    timestamp=ts,
                    bbox=det.bbox,
                    torso_bbox=tbox,
                    quality=qm,
                    tracking_id=det.tracking_id,
                    split="unassigned",
                )

    split_map = assign_temporal_and_track_splits(
        list(ds.records.values()),
        video_duration_seconds=duration,
        split_config=match_config.splits,
    )
    for cid, sp in split_map.items():
        ds.records[cid].split = sp
    ds.save()
    return ds


def run_match_pipeline(
    config: MatchConfig | str | Path,
    base_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    max_tracking_frames: Optional[int] = 180,
    compare_trackers: bool = False,
    tracker_comparison_frames: int = 90,
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end Football AI pipeline for a configured match:
    1. Dataset loading & automated label audit
    2. Feature & classifier benchmarking + Unknown threshold calibration
    3. Video detection, tracking, jersey classification & temporal stabilization
    4. Annotated video & possession-foundation telemetry export
    5. Detection ground-truth / confidence-sweep evaluation
    6. Consolidated JSON/CSV/Markdown analytics report generation
    """
    root = base_dir or Path.cwd()
    match_cfg = load_match_config(config) if isinstance(config, (str, Path)) else config
    video_path = match_cfg.resolve_video_path(root)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = experiment_name or f"{match_cfg.match_id}_{ts_tag}"
    output_dir = root / "outputs" / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare / load dataset
    dataset = prepare_dataset_for_match(match_cfg, base_dir=root)
    train_samples = dataset.get_labelled_samples(split="train", exclude_unknown=True)
    val_samples = dataset.get_labelled_samples(split="val", exclude_unknown=False)

    # Save reference galleries
    artifacts: Dict[str, str] = {}
    if train_samples:
        p_tr = save_crop_reference_gallery(
            train_samples,
            f"{match_cfg.match_name} — Training Set ({len(train_samples)} samples)",
            output_dir / "gallery_train_samples.png",
        )
        artifacts["gallery_train"] = str(p_tr)
    if val_samples:
        p_val = save_crop_reference_gallery(
            val_samples,
            f"{match_cfg.match_name} — Validation Set ({len(val_samples)} samples)",
            output_dir / "gallery_val_samples.png",
        )
        artifacts["gallery_val"] = str(p_val)

    # 2. Audit labels
    all_labelled = dataset.get_labelled_samples(exclude_unknown=False)
    label_audit = audit_dataset_labels(
        all_labelled, feature_method=match_cfg.classification.feature_method
    )

    # 3. Run feature & classifier benchmark
    benchmark_results = run_classification_benchmark(
        train_samples=train_samples,
        val_samples=val_samples,
        match_config=match_cfg,
    )

    # Train and calibrate the active classifier
    active_clf = TeamClassifier(match_id=match_cfg.match_id, config=match_cfg.classification)
    active_clf.fit(
        [img for _, img, _ in train_samples],
        [lbl for _, _, lbl in train_samples],
    )
    calibration_info = active_clf.calibrate_unknown_threshold(
        [img for _, img, _ in val_samples],
        [lbl for _, _, lbl in val_samples],
    )
    clf_eval = evaluate_team_classifier(active_clf, val_samples, match_config=match_cfg)
    #Also evaluate on held-out test split if present in dataset
    test_samples = dataset.get_labelled_samples(split="test", exclude_unknown=False)
    if test_samples:
        test_eval = evaluate_team_classifier(active_clf, test_samples, match_config=match_cfg)
        clf_eval["held_out_test_evaluation"] = {
            "total_samples": test_eval["total_samples"],
            "overall_accuracy": test_eval["overall_accuracy"],
            "selective_accuracy": test_eval["selective_accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "positive_macro_f1": test_eval["positive_macro_f1"],
            "coverage": test_eval["coverage"],
            "unknown_rejection_rate": test_eval["unknown_rejection_rate"],
            "per_class": test_eval["per_class"],
            "confusion_matrix": test_eval["confusion_matrix"],
        }
    clf_eval["training_size"] = len(train_samples)
    clf_eval["training_class_counts"] = active_clf.class_counts_
    clf_eval["calibration"] = {
        "selected_threshold": calibration_info["selected_threshold"],
        "known_distance_summary": calibration_info["known_distance_summary"],
        "unknown_distance_summary": calibration_info["unknown_distance_summary"],
    }

    model_path = active_clf.save(output_dir / f"{match_cfg.match_id}_team_classifier.joblib")
    artifacts["classifier_checkpoint"] = str(model_path)

    # Save classification diagnostic plots
    cm_path = save_confusion_matrix_plot(
        clf_eval["confusion_matrix"],
        clf_eval["labels_display"],
        f"{match_cfg.match_name} — {match_cfg.classification.feature_method} ({clf_eval['overall_accuracy']:.1%})",
        output_dir / "confusion_matrix_active.png",
    )
    artifacts["confusion_matrix"] = str(cm_path)

    # Side-by-side comparison of legacy baseline vs legacy enhanced vs active model
    side_matrices = []
    for b_item in benchmark_results:
        if b_item["experiment_tag"] in (
            "baseline_hsv__knn3_uncalibrated_t0.45",
            "legacy_enhanced__knn3_uncalibrated_t0.45",
        ):
            side_matrices.append(
                (
                    f"{b_item['feature_method']} (t=0.45)\nAcc: {b_item['overall_accuracy']:.1%}",
                    b_item["confusion_matrix"],
                    b_item["labels_display"],
                )
            )
    side_matrices.append(
        (
            f"Active: {clf_eval['feature_method']}\nAcc: {clf_eval['overall_accuracy']:.1%}",
            clf_eval["confusion_matrix"],
            clf_eval["labels_display"],
        )
    )
    sbs_path = save_side_by_side_confusion_matrices(
        side_matrices, output_dir / "confusion_matrix_comparison.png"
    )
    artifacts["confusion_matrix_comparison"] = str(sbs_path)

    cal_plot_path = save_threshold_calibration_plot(
        calibration_info, output_dir / "unknown_threshold_calibration.png"
    )
    artifacts["threshold_calibration_plot"] = str(cal_plot_path)

    misc_path = save_misclassification_gallery(
        val_samples,
        clf_eval["misclassified_samples"],
        output_dir / "misclassification_gallery.png",
    )
    artifacts["misclassification_gallery"] = str(misc_path)

    # 4. Run video tracking, classification, and temporal stabilization
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_fps = fps / float(max(1, match_cfg.detection.frame_stride))
    raw_video_path = output_dir / f"{match_cfg.match_id}_raw.mp4"
    final_video_path = output_dir / f"{match_cfg.match_id}_analyzed.mp4"

    writer = cv2.VideoWriter(
        str(raw_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    tracker = MultiObjectTracker(
        detection_config=match_cfg.detection,
        tracking_config=match_cfg.tracking,
    )
    stabilizer = TemporalTeamStabilizer(config=match_cfg.tracking)

    people_per_frame: List[int] = []
    balls_per_frame: List[int] = []
    observation_records: List[FrameObservationRecord] = []
    processed_frames = 0

    try:
        for fno, ts, frame_bgr, detections in tracker.track_video(
            video_path, max_frames=max_tracking_frames, fps_override=fps
        ):
            processed_frames += 1
            player_dets = [d for d in detections if d.class_name == "player"]
            ball_dets = [d for d in detections if d.class_name == "ball"]

            people_per_frame.append(len(player_dets))
            balls_per_frame.append(len(ball_dets))

            primary_ball = max(ball_dets, key=lambda b: b.confidence) if ball_dets else None
            all_p_bboxes = [d.bbox for d in player_dets]
            frame_obs: List[FrameObservationRecord] = []

            for p_det in player_dets:
                crop, _, _ = extract_torso_crop(
                    frame_bgr,
                    p_det.bbox,
                    config=match_cfg.cropping,
                    other_player_bboxes=all_p_bboxes,
                )
                if crop is not None:
                    pred = active_clf.predict_crop(crop, match_id=match_cfg.match_id)
                else:
                    pred = ClassificationPrediction(
                        predicted_label=InternalLabel.UNKNOWN.value,
                        raw_nearest_label=InternalLabel.UNKNOWN.value,
                        nearest_distance=1.0,
                        second_nearest_distance=1.0,
                        margin=0.0,
                        confidence_score=0.0,
                        is_unknown=True,
                        rejection_reason="filtered_crop",
                    )

                stable_lbl, trk_conf, is_locked = stabilizer.update(
                    p_det.tracking_id, fno, p_det.bbox, pred
                )
                disp_name = match_cfg.to_display_label(stable_lbl)

                obs_rec = FrameObservationRecord(
                    match_id=match_cfg.match_id,
                    frame_number=fno,
                    timestamp=ts,
                    player_id=p_det.tracking_id,
                    team_id=stable_lbl,
                    team_display_name=disp_name,
                    raw_team_id=pred.predicted_label,
                    player_bbox=p_det.bbox,
                    player_detection_confidence=p_det.confidence,
                    classification_confidence=pred.confidence_score,
                    classification_distance=pred.nearest_distance,
                    track_team_confidence=trk_conf,
                    is_team_locked=is_locked,
                    ball_bbox=primary_ball.bbox if primary_ball else None,
                    ball_detection_confidence=primary_ball.confidence if primary_ball else None,
                )
                frame_obs.append(obs_rec)
                observation_records.append(obs_rec)

            annotated_frame = annotate_match_frame(frame_bgr, frame_obs, ball_dets, match_cfg)
            writer.write(annotated_frame)
    finally:
        writer.release()

    finalized_video = finalize_mp4_video(raw_video_path, final_video_path)
    artifacts["annotated_video"] = str(finalized_video)

    timeline_path = save_detections_timeline_plot(
        people_per_frame,
        balls_per_frame,
        output_fps=output_fps,
        output_path=output_dir / "detections_timeline.png",
    )
    artifacts["detections_timeline"] = str(timeline_path)

    obs_csv_path = export_possession_foundation_records(
        output_dir / "frame_observations.csv", observation_records
    )
    artifacts["frame_observations_csv"] = str(obs_csv_path)

    tracking_summary = stabilizer.summarize_diagnostics(processed_frames=processed_frames)
    tracking_summary["tracker_config"] = match_cfg.tracking.tracker_config

    # 5. Optional ByteTrack vs BoT-SORT comparison
    tracker_comparison: Dict[str, Dict[str, Any]] = {}
    if compare_trackers:
        tracker_comparison = compare_tracking_algorithms(
            video_path=video_path,
            match_config=match_cfg,
            classifier=active_clf,
            tracker_configs=("bytetrack.yaml", "botsort.yaml"),
            max_frames=tracker_comparison_frames,
        )

    # 6. Detection evaluation & confidence sweep on sampled frames
    ann_store = DetectionAnnotationStore(
        root / "data" / "annotations" / match_cfg.match_id,
        match_id=match_cfg.match_id,
    )
    sample_frame_ids = sorted(LEGACY_NORWAY_WALES_DETECTION_COUNTS.keys()) if match_cfg.match_id == "norway_wales" else [
        int(f) for f in (total_video_frames * 0.55 + i * 50 for i in range(6)) if f < total_video_frames
    ]
    if not ann_store.frames:
        ann_store.export_frames_for_annotation(
            video_path=video_path,
            frame_numbers=sample_frame_ids,
            legacy_count_records=LEGACY_NORWAY_WALES_DETECTION_COUNTS if match_cfg.match_id == "norway_wales" else None,
        )

    detector = FootballDetector(config=match_cfg.detection)
    raw_sample_preds = {}
    pred_counts_at_default = {}
    cap_eval = cv2.VideoCapture(str(video_path))
    try:
        for fno in sample_frame_ids:
            cap_eval.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frm = cap_eval.read()
            if not ok or frm is None:
                continue
            # Run with low threshold (0.15) to support full confidence sweep
            raw_dets = detector.detect_frame(
                frm,
                frame_number=fno,
                timestamp=round(fno / fps, 3),
                person_conf_override=0.15,
                ball_conf_override=0.15,
            )
            raw_sample_preds[fno] = raw_dets
            p_def = sum(
                1 for d in raw_dets if d.class_name == "player" and d.confidence >= match_cfg.detection.person_conf
            )
            b_def = sum(
                1 for d in raw_dets if d.class_name == "ball" and d.confidence >= match_cfg.detection.ball_conf
            )
            pred_counts_at_default[fno] = {"people": p_def, "balls": b_def}
    finally:
        cap_eval.release()

    verified_gt = ann_store.get_verified_ground_truth()
    default_filtered_preds = {
        fno: [
            d
            for d in dets
            if (d.class_name == "player" and d.confidence >= match_cfg.detection.person_conf)
            or (d.class_name == "ball" and d.confidence >= match_cfg.detection.ball_conf)
        ]
        for fno, dets in raw_sample_preds.items()
    }
    iou_evaluation = evaluate_detections_against_ground_truth(
        default_filtered_preds,
        verified_gt,
        iou_threshold=match_cfg.detection.iou_threshold,
    )
    conf_sweep = run_confidence_threshold_sweep(
        raw_sample_preds,
        thresholds=(0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
        verified_annotations=verified_gt,
    )
    sweep_csv = write_csv_rows(output_dir / "detection_confidence_sweep.csv", conf_sweep)
    artifacts["detection_confidence_sweep_csv"] = str(sweep_csv)

    legacy_count_eval = None
    if match_cfg.match_id == "norway_wales":
        legacy_count_eval = evaluate_legacy_manual_counts(
            pred_counts_at_default, LEGACY_NORWAY_WALES_DETECTION_COUNTS
        )

    frames_with_ball = sum(1 for c in balls_per_frame if c > 0)
    detection_summary = {
        "total_frames_processed": processed_frames,
        "avg_players_per_frame": round(
            sum(people_per_frame) / float(max(1, processed_frames)), 2
        ),
        "total_ball_detections": sum(balls_per_frame),
        "frames_with_ball": frames_with_ball,
        "ball_coverage_pct": round(frames_with_ball / float(max(1, processed_frames)) * 100.0, 2),
        "person_conf_threshold": match_cfg.detection.person_conf,
        "ball_conf_threshold": match_cfg.detection.ball_conf,
        "iou_evaluation": iou_evaluation,
        "legacy_count_evaluation": legacy_count_eval,
        "confidence_sweep": conf_sweep,
    }

    report_files = generate_consolidated_report(
        output_dir=output_dir,
        match_id=match_cfg.match_id,
        match_name=match_cfg.match_name,
        detection_summary=detection_summary,
        classification_summary=clf_eval,
        tracking_summary=tracking_summary,
        tracker_comparison=tracker_comparison,
        feature_benchmark=benchmark_results,
        label_audit=label_audit,
        artifacts_generated=artifacts,
    )

    return {
        "experiment_id": exp_id,
        "output_dir": str(output_dir),
        "detection_summary": detection_summary,
        "classification_summary": clf_eval,
        "tracking_summary": tracking_summary,
        "tracker_comparison": tracker_comparison,
        "label_audit": label_audit,
        "report_files": {k: str(v) for k, v in report_files.items()},
        "artifacts": artifacts,
    }

