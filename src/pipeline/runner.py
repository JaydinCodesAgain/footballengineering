"""End-to-end pipeline orchestration for dataset preparation, training, tracking, possession, and reporting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2

from src.classification.classifier import TeamClassifier
from src.classification.cropper import extract_torso_crop
from src.classification.dataset import JerseyCropDataset, assign_temporal_and_track_splits
from src.classification.diagnostics import audit_dataset_labels
from src.config import MatchConfig, load_match_config
from src.detection.annotation_workflow import (
    DetectionAnnotationStore,
    sample_representative_frame_numbers,
)
from src.detection.detector import (
    FootballDetector,
    _nms_ball_detections,
    benchmark_ball_detection_strategies,
)
from src.evaluation.classification_eval import (
    evaluate_team_classifier,
    run_classification_benchmark,
)
from src.evaluation.detection_eval import (
    evaluate_detections_against_ground_truth,
    evaluate_legacy_manual_counts,
    run_confidence_threshold_sweep,
)
from src.evaluation.possession_eval import (
    PossessionAnnotationStore,
    evaluate_possession_predictions,
)
from src.evaluation.report import (
    export_ball_trajectory_records,
    export_player_ownership_records,
    export_possession_foundation_records,
    export_possession_timeline_records,
    generate_consolidated_report,
    generate_possession_report,
    write_csv_rows,
)
from src.evaluation.tracking_eval import compare_tracking_algorithms
from src.pipeline.legacy_migration import (
    LEGACY_NORWAY_WALES_DETECTION_COUNTS,
    migrate_chelsea_sunderland_legacy_dataset,
    migrate_norway_wales_legacy_dataset,
)
from src.possession.ball_tracker import filter_ball_candidates
from src.possession.possession_engine import PossessionEngine
from src.schemas import ClassificationPrediction, FrameObservationRecord, InternalLabel
from src.tracking.stabilizer import TemporalTeamStabilizer
from src.tracking.tracker import MultiObjectTracker
from src.visualization.plots import (
    save_ball_detection_diagnostic_gallery,
    save_confusion_matrix_plot,
    save_crop_reference_gallery,
    save_detections_timeline_plot,
    save_misclassification_gallery,
    save_possession_timeline_chart,
    save_side_by_side_confusion_matrices,
    save_threshold_calibration_plot,
)
from src.visualization.video_annotator import annotate_match_frame, finalize_mp4_video
import numpy as np

def prepare_dataset_for_match(
    match_config: MatchConfig,
    base_dir: Optional[Path] = None,
    force_rebuild: bool = False,
) -> JerseyCropDataset:
    """
    Prepare or load the jersey crop dataset for a match.
    Validates manifest integrity and deterministically regenerates missing crop PNGs
    from the source video when needed.
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
        ds.regenerate_missing_crops(video_path)
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


def _build_ball_diagnostic_panel_crop(
    frame_bgr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    label_text: str = "Ball",
    pad: int = 110,
) -> Any:
    """Create a zoomed diagnostic patch around a ball detection or full-frame thumbnail if missed."""
    fh, fw = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            canvas,
            label_text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            box_color,
            2,
        )
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cx0 = max(0, cx - pad)
        cy0 = max(0, cy - pad)
        cx1 = min(fw, cx + pad)
        cy1 = min(fh, cy + pad)
        patch = canvas[cy0:cy1, cx0:cx1]
        if patch.size > 0:
            return cv2.resize(patch, (320, 240), interpolation=cv2.INTER_AREA)

    return cv2.resize(canvas, (320, 240), interpolation=cv2.INTER_AREA)


def run_match_pipeline(
    config: MatchConfig | str | Path,
    base_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    max_tracking_frames: Optional[int] = 180,
    compare_trackers: bool = False,
    tracker_comparison_frames: int = 90,
    enable_possession: Optional[bool] = None,
    run_ball_strategy_benchmark: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end Football AI pipeline for a configured match:
    1. Dataset integrity verification & automated label audit
    2. Feature & classifier benchmarking + Unknown threshold calibration
    3. Video detection, player tracking, Kalman ball tracking, jersey classification & possession engine
    4. Annotated video (with live possession scoreboard) & telemetry CSV exports
    5. Detection ground-truth / confidence-sweep / ball-strategy benchmark evaluation
    6. Consolidated JSON/CSV/Markdown analytics & possession report generation
    """
    root = base_dir or Path.cwd()
    match_cfg = load_match_config(config) if isinstance(config, (str, Path)) else config
    if enable_possession is not None:
        match_cfg.possession.enabled = bool(enable_possession)

    video_path = match_cfg.resolve_video_path(root)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = experiment_name or f"{match_cfg.match_id}_{ts_tag}"
    output_dir = root / "outputs" / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare / load dataset with integrity verification
    dataset = prepare_dataset_for_match(match_cfg, base_dir=root)
    train_samples = dataset.get_labelled_samples(split="train", exclude_unknown=True)
    val_samples = dataset.get_labelled_samples(split="val", exclude_unknown=False)

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

    # Train and calibrate the active classifier strictly on VAL
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

    # Evaluate on held-out test split if present in dataset (frozen threshold from VAL)
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

    # 4. Run video tracking, classification, temporal stabilization, and possession analytics
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_duration = total_video_frames / fps
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
    aux_detector = FootballDetector(config=match_cfg.detection)
    stabilizer = TemporalTeamStabilizer(config=match_cfg.tracking)
    possession_engine = (
        PossessionEngine(
            match_config=match_cfg,
            fps=fps,
            frame_stride=match_cfg.detection.frame_stride,
            total_video_duration_seconds=total_video_duration,
        )
        if match_cfg.possession.enabled
        else None
    )

    people_per_frame: List[int] = []
    balls_per_frame: List[int] = []
    observation_records: List[FrameObservationRecord] = []
    ball_diagnostic_items: List[Dict[str, Any]] = []
    processed_frames = 0
    first_frame_seen: Optional[int] = None
    last_frame_seen: Optional[int] = None
    t_track_start = time.perf_counter()

    try:
        for fno, ts, frame_bgr, detections in tracker.track_video(
            video_path, max_frames=max_tracking_frames, fps_override=fps
        ):
            processed_frames += 1
            if first_frame_seen is None:
                first_frame_seen = fno
            last_frame_seen = fno

            player_dets = [d for d in detections if d.class_name == "player"]
            ball_dets = [d for d in detections if d.class_name == "ball"]
            det_source = "full_or_tiled" if ball_dets else "none"

            # If ball still missed on this frame and Kalman ROI search is available, try ROI crop search
            if not ball_dets and possession_engine is not None:
                roi_box = possession_engine.get_predicted_ball_roi((height, width))
                if roi_box is not None:
                    roi_balls = aux_detector.detect_ball_in_roi(
                        frame_bgr,
                        roi_box,
                        frame_number=fno,
                        timestamp=ts,
                        conf_threshold=match_cfg.detection.roi_conf_threshold,
                    )
                    if roi_balls:
                        ball_dets = _nms_ball_detections(roi_balls)
                        det_source = "kalman_roi_recovery"

            # Collect diagnostic gallery examples across failure & recovery modes
            if len(ball_diagnostic_items) < 12:
                filtered_check = filter_ball_candidates(
                    ball_dets,
                    player_dets,
                    (height, width),
                    min_size=match_cfg.detection.min_ball_box_size_px,
                    max_size=match_cfg.detection.max_ball_box_size_px,
                )
                if len(ball_dets) > len(filtered_check):
                    rejected_b = [b for b in ball_dets if b not in filtered_check][0]
                    ball_diagnostic_items.append(
                        {
                            "category": "Rejected False Positive",
                            "frame_number": fno,
                            "note": f"conf={rejected_b.confidence:.2f} filtered (size/torso/overlay)",
                            "title_color": "darkred",
                            "crop_or_frame_bgr": _build_ball_diagnostic_panel_crop(
                                frame_bgr, rejected_b.bbox, (0, 0, 255), f"FP {rejected_b.confidence:.2f}"
                            ),
                        }
                    )
                elif filtered_check:
                    top_b = max(filtered_check, key=lambda b: b.confidence)
                    if top_b.confidence < 0.30 and sum(1 for x in ball_diagnostic_items if x["category"] == "Low-Conf Recovery") < 4:
                        ball_diagnostic_items.append(
                            {
                                "category": "Low-Conf Recovery",
                                "frame_number": fno,
                                "note": f"conf={top_b.confidence:.2f} ({det_source})",
                                "title_color": "darkorange",
                                "crop_or_frame_bgr": _build_ball_diagnostic_panel_crop(
                                    frame_bgr, top_b.bbox, (0, 180, 255), f"LowConf {top_b.confidence:.2f}"
                                ),
                            }
                        )
                    elif top_b.confidence >= 0.35 and sum(1 for x in ball_diagnostic_items if x["category"] == "High-Conf Hit") < 4:
                        ball_diagnostic_items.append(
                            {
                                "category": "High-Conf Hit",
                                "frame_number": fno,
                                "note": f"conf={top_b.confidence:.2f} ({det_source})",
                                "title_color": "darkgreen",
                                "crop_or_frame_bgr": _build_ball_diagnostic_panel_crop(
                                    frame_bgr, top_b.bbox, (0, 220, 0), f"Ball {top_b.confidence:.2f}"
                                ),
                            }
                        )
                elif sum(1 for x in ball_diagnostic_items if x["category"] == "Missed / Occluded") < 3:
                    ball_diagnostic_items.append(
                        {
                            "category": "Missed / Occluded",
                            "frame_number": fno,
                            "note": "No ball above threshold (blur/occlusion)",
                            "title_color": "dimgray",
                            "crop_or_frame_bgr": _build_ball_diagnostic_panel_crop(frame_bgr, None),
                        }
                    )

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

                primary_raw_ball = max(ball_dets, key=lambda b: b.confidence) if ball_dets else None
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
                    ball_bbox=primary_raw_ball.bbox if primary_raw_ball else None,
                    ball_detection_confidence=primary_raw_ball.confidence if primary_raw_ball else None,
                )
                frame_obs.append(obs_rec)
                observation_records.append(obs_rec)

            ball_traj_rec = None
            poss_frame_rec = None
            if possession_engine is not None:
                ball_traj_rec, poss_frame_rec, _ = possession_engine.step_frame(
                    frame_number=fno,
                    timestamp=ts,
                    player_observations=frame_obs,
                    raw_ball_detections=ball_dets,
                    player_detections=player_dets,
                    frame_bgr=frame_bgr,
                    frame_shape=(height, width),
                    detection_source=det_source,
                )
                people_per_frame.append(len(player_dets))
                balls_per_frame.append(1 if ball_traj_rec.is_observed else 0)
            else:
                people_per_frame.append(len(player_dets))
                balls_per_frame.append(len(ball_dets))

            annotated_frame = annotate_match_frame(
                frame_bgr,
                frame_obs,
                ball_dets,
                match_cfg,
                ball_trajectory_rec=ball_traj_rec,
                possession_rec=poss_frame_rec,
            )
            writer.write(annotated_frame)
    finally:
        writer.release()

    processing_wall_time_sec = round(time.perf_counter() - t_track_start, 2)
    avg_proc_fps = round(processed_frames / max(0.01, processing_wall_time_sec), 2)

    experiment_metadata: Dict[str, Any] = {
        "experiment_id": exp_id,
        "match_id": match_cfg.match_id,
        "match_name": match_cfg.match_name,
        "source_video_filename": video_path.name,
        "source_video_path": str(video_path),
        "video_resolution": f"{width}x{height}",
        "video_fps": round(fps, 2),
        "total_video_frames": total_video_frames,
        "total_video_duration_seconds": round(total_video_duration, 2),
        "start_frame": first_frame_seen if first_frame_seen is not None else match_cfg.tracking.start_frame,
        "end_frame": last_frame_seen if last_frame_seen is not None else match_cfg.tracking.start_frame,
        "frame_stride": match_cfg.detection.frame_stride,
        "processed_frames": processed_frames,
        "analysed_duration_seconds": round(processed_frames * (match_cfg.detection.frame_stride / fps), 2),
        "model_weights": match_cfg.detection.model_name,
        "imgsz": match_cfg.detection.imgsz,
        "person_conf_threshold": match_cfg.detection.person_conf,
        "ball_conf_threshold": match_cfg.detection.ball_conf,
        "tile_conf_threshold": match_cfg.detection.tile_conf_threshold,
        "roi_conf_threshold": match_cfg.detection.roi_conf_threshold,
        "enable_tiled_ball_search": match_cfg.detection.enable_tiled_ball_search,
        "enable_roi_ball_search": match_cfg.detection.enable_roi_ball_search,
        "tracker_config": match_cfg.tracking.tracker_config,
        "processing_wall_time_seconds": processing_wall_time_sec,
        "average_processing_fps": avg_proc_fps,
    }

    finalized_video = finalize_mp4_video(raw_video_path, final_video_path)
    artifacts["annotated_video"] = str(finalized_video)

    if ball_diagnostic_items:
        b_diag_path = save_ball_detection_diagnostic_gallery(
            ball_diagnostic_items, output_dir / "ball_detection_diagnostics.png"
        )
        artifacts["ball_detection_diagnostics"] = str(b_diag_path)

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

    # Export Possession & Ball Trajectory outputs
    possession_summary: Dict[str, Any] = {}
    possession_validation: Dict[str, Any] = {}
    if possession_engine is not None:
        possession_summary = possession_engine.summarize()
        possession_summary["experiment_metadata"] = experiment_metadata
        b_csv = export_ball_trajectory_records(
            output_dir / "ball_trajectory.csv", possession_engine.ball_trajectory
        )
        o_csv = export_player_ownership_records(
            output_dir / "player_ownership.csv", possession_engine.player_proximities
        )
        t_csv = export_possession_timeline_records(
            output_dir / "possession_timeline.csv", possession_engine.timeline
        )
        artifacts["ball_trajectory_csv"] = str(b_csv)
        artifacts["player_ownership_csv"] = str(o_csv)
        artifacts["possession_timeline_csv"] = str(t_csv)

        p_chart = save_possession_timeline_chart(
            possession_engine.timeline,
            match_cfg,
            possession_summary,
            output_dir / "possession_timeline_chart.png",
        )
        artifacts["possession_timeline_chart"] = str(p_chart)

        # Initialize / evaluate manual possession annotations
        poss_ann_store = PossessionAnnotationStore(
            root / "data" / "annotations" / match_cfg.match_id,
            match_id=match_cfg.match_id,
        )
        if first_frame_seen is not None and last_frame_seen is not None and not poss_ann_store.intervals:
            poss_ann_store.create_template_intervals(
                start_frame=first_frame_seen,
                end_frame=last_frame_seen,
                fps=fps,
                interval_frames=30,
            )
        poss_html = poss_ann_store.generate_html_annotator(match_config=match_cfg, fps=fps)
        artifacts["possession_annotator_html"] = str(poss_html)
        verified_poss_ivs = poss_ann_store.get_verified_intervals()
        possession_validation = evaluate_possession_predictions(
            possession_engine.timeline,
            verified_poss_ivs,
            match_config=match_cfg,
        )
        if possession_validation.get("metrics_available"):
            pcm_path = save_confusion_matrix_plot(
                possession_validation["confusion_matrix"],
                possession_validation["state_labels"],
                f"{match_cfg.match_name} — Possession State Validation",
                output_dir / "possession_confusion_matrix.png",
            )
            artifacts["possession_confusion_matrix"] = str(pcm_path)

        poss_rep_files = generate_possession_report(
            output_dir=output_dir,
            possession_summary=possession_summary,
            possession_validation=possession_validation,
        )
        for k, v in poss_rep_files.items():
            artifacts[k] = str(v)

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

    # 6. Detection evaluation, confidence sweep & ball strategy benchmark on sampled frames
    ann_store = DetectionAnnotationStore(
        root / "data" / "annotations" / match_cfg.match_id,
        match_id=match_cfg.match_id,
    )
    legacy_fnos = (
        sorted(LEGACY_NORWAY_WALES_DETECTION_COUNTS.keys())
        if match_cfg.match_id == "norway_wales"
        else []
    )
    representative_frame_ids = sample_representative_frame_numbers(
        total_frames=total_video_frames,
        num_samples=45,
        mandatory_frames=legacy_fnos,
    )
    if len(ann_store.frames) < len(representative_frame_ids):
        ann_store.export_frames_for_annotation(
            video_path=video_path,
            frame_numbers=representative_frame_ids,
            legacy_count_records=LEGACY_NORWAY_WALES_DETECTION_COUNTS if match_cfg.match_id == "norway_wales" else None,
        )
    det_html = ann_store.generate_html_annotator()
    artifacts["detection_annotator_html"] = str(det_html)

    sample_frame_ids = (
        legacy_fnos
        if match_cfg.match_id == "norway_wales"
        else representative_frame_ids[::4][:12]
    )

    raw_sample_preds = {}
    pred_counts_at_default = {}
    sample_frames_list: List[Tuple[int, float, Any]] = []
    cap_eval = cv2.VideoCapture(str(video_path))
    try:
        for fno in sample_frame_ids:
            cap_eval.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frm = cap_eval.read()
            if not ok or frm is None:
                continue
            ts_f = round(fno / fps, 3)
            sample_frames_list.append((int(fno), ts_f, frm))
            raw_dets = aux_detector.detect_frame(
                frm,
                frame_number=fno,
                timestamp=ts_f,
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

    ball_strategy_benchmark: List[Dict[str, Any]] = []
    if run_ball_strategy_benchmark and sample_frames_list:
        ball_strategy_benchmark = benchmark_ball_detection_strategies(
            sample_frames_list[:6],
            verified_annotations=verified_gt,
            iou_threshold=match_cfg.detection.iou_threshold,
        )

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
        "ball_strategy_benchmark": ball_strategy_benchmark,
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
        possession_summary=possession_summary,
        possession_validation=possession_validation,
        ball_detection_benchmark=ball_strategy_benchmark,
        artifacts_generated=artifacts,
        experiment_metadata=experiment_metadata,
    )

    return {
        "experiment_id": exp_id,
        "output_dir": str(output_dir),
        "experiment_metadata": experiment_metadata,
        "detection_summary": detection_summary,
        "classification_summary": clf_eval,
        "tracking_summary": tracking_summary,
        "tracker_comparison": tracker_comparison,
        "possession_summary": possession_summary,
        "possession_validation": possession_validation,
        "ball_detection_benchmark": ball_strategy_benchmark,
        "label_audit": label_audit,
        "report_files": {k: str(v) for k, v in report_files.items()},
        "artifacts": artifacts,
    }
