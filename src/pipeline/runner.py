"""End-to-end pipeline orchestration for dataset preparation, training, tracking, possession, and reporting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2

from src.classification.classifier import TeamClassifier
from src.classification.cropper import extract_torso_crop
from src.classification.dataset import JerseyCropDataset, assign_temporal_and_track_splits
from src.classification.diagnostics import audit_dataset_labels
from src.classification.features import get_feature_extractor
from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    TrackAppearanceBuffer,
    collect_track_appearances,
    filter_discovery_player_indices,
    is_tactical_discovery_frame,
)
from src.config import MatchConfig, load_match_config
from src.detection.motion_ball import FixedCameraMotionBallDetector, MotionBallConfig
from src.geometry.pitch import load_pitch_calibration
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter
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
from src.possession.ball_tracker import detect_camera_cut, filter_ball_candidates
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


def _build_pitch_region_filter(match_config: MatchConfig) -> Optional[PitchRegionFilter]:
    """Build optional geometric pitch-region filter when a fixed-camera calibration is supplied."""
    if not match_config.pitch.calibration_path or not match_config.pitch.enable_region_filter:
        return None
    pr_cfg = PitchRegionConfig(
        pitch_length_m=float(match_config.pitch.pitch_length_m),
        pitch_width_m=float(match_config.pitch.pitch_width_m),
        player_margin_m=float(match_config.pitch.player_margin_m),
        ball_margin_m=float(match_config.pitch.ball_margin_m),
    )
    return PitchRegionFilter.from_calibration_path(
        match_config.pitch.calibration_path,
        coordinate_space=str(match_config.pitch.coordinate_space),
        config=pr_cfg,
    )


def _discover_teams_with_frame_cache(
    video_path: Path,
    match_config: MatchConfig,
    max_frames: Optional[int] = 300,
    fps_override: Optional[float] = None,
    pitch_region_filter: Optional[PitchRegionFilter] = None,
) -> Tuple[
    AutomaticTeamDiscoverer,
    List[Tuple[Any, np.ndarray, str]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """Run one detection/tracking pass, cache lightweight per-frame tracks & features,
    and fit per-video unsupervised team appearances on tactical, shot-segmented tracks.
    """
    tracker = MultiObjectTracker(
        detection_config=match_config.detection,
        tracking_config=match_config.tracking,
        pitch_region_filter=pitch_region_filter,
        fixed_camera=bool(match_config.ball_tracking.fixed_camera),
        use_pitch_roi_for_tiling=bool(match_config.pitch.use_pitch_roi_for_tiling),
        filter_players_by_pitch=bool(match_config.pitch.filter_players_by_pitch),
        filter_balls_by_pitch=bool(match_config.pitch.filter_balls_by_pitch),
        motion_ball_detector=None,
    )
    extractor = get_feature_extractor(match_config.team_discovery.feature_method)
    tactical_observations: List[TeamDiscoveryObservation] = []
    fallback_observations: List[TeamDiscoveryObservation] = []
    cached_frames: List[Dict[str, Any]] = []

    prev_small: Optional[np.ndarray] = None
    prev_med_h: Optional[float] = None
    last_cut_frame: Optional[int] = None
    shot_id = 0
    cooldown = max(0, int(match_config.ball_tracking.camera_cut_cooldown_frames))

    for fno, ts, frame_bgr, detections in tracker.track_video(
        video_path, max_frames=max_frames, fps_override=fps_override
    ):
        fh, fw = frame_bgr.shape[:2]
        players = [d for d in detections if d.class_name == "player"]
        balls = [d for d in detections if d.class_name == "ball"]
        player_boxes = [d.bbox for d in players]
        heights = [float(b[3] - b[1]) for b in player_boxes if (b[3] - b[1]) > 15]
        med_h = float(np.median(heights)) if heights else None

        small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
        is_cut = False
        if (
            not match_config.ball_tracking.fixed_camera
            and prev_small is not None
            and detect_camera_cut(
                prev_small,
                small_curr,
                corr_threshold=match_config.ball_tracking.camera_cut_hist_threshold,
                prev_median_player_h=prev_med_h,
                curr_median_player_h=med_h,
            )
        ):
            if last_cut_frame is None or (fno - last_cut_frame) > cooldown:
                shot_id += 1
                is_cut = True
                last_cut_frame = fno
        prev_small = small_curr
        if med_h is not None:
            prev_med_h = med_h

        if pitch_region_filter is not None and match_config.pitch.filter_players_by_pitch:
            tactical = len(player_boxes) >= 4
            valid_disc_indices = set(range(len(player_boxes))) if tactical else set()
        else:
            tactical = is_tactical_discovery_frame(player_boxes, (fh, fw))
            valid_disc_indices = set(filter_discovery_player_indices(player_boxes, (fh, fw))) if tactical else set()

        per_player_features: List[Optional[np.ndarray]] = []
        per_player_weights: List[float] = []
        for p_idx, player in enumerate(players):
            crop, _, quality = extract_torso_crop(
                frame_bgr, player.bbox, config=match_config.cropping, other_player_bboxes=player_boxes
            )
            if crop is None or not quality.is_valid:
                per_player_features.append(None)
                per_player_weights.append(0.0)
                continue

            feat = extractor(crop)
            q_weight = (1.0 - quality.grass_ratio) * (1.0 - quality.occlusion_ratio)
            per_player_features.append(feat)
            per_player_weights.append(max(0.05, float(q_weight)))

            obs = TeamDiscoveryObservation(
                tracking_id=player.tracking_id,
                frame_number=fno,
                timestamp=ts,
                crop_bgr=crop,
                blur_score=quality.blur_score,
                grass_ratio=quality.grass_ratio,
                occlusion_ratio=quality.occlusion_ratio,
                shot_id=shot_id,
                bbox=player.bbox,
                feature=feat,
            )
            fallback_observations.append(obs)
            if tactical and p_idx in valid_disc_indices:
                tactical_observations.append(obs)

        cached_frames.append(
            {
                "frame_number": fno,
                "timestamp": ts,
                "shot_id": shot_id,
                "camera_cut": is_cut,
                "players": players,
                "balls": balls,
                "player_features": per_player_features,
                "player_weights": per_player_weights,
            }
        )

    appearances = collect_track_appearances(tactical_observations, match_config.team_discovery)
    if len(appearances) < 4:
        appearances = collect_track_appearances(fallback_observations, match_config.team_discovery)
    discoverer = AutomaticTeamDiscoverer(match_config.team_discovery).fit(appearances)

    from types import SimpleNamespace

    gallery_samples: List[Tuple[Any, np.ndarray, str]] = []
    source_obs = tactical_observations if len(tactical_observations) >= len(appearances) else fallback_observations
    by_seg: Dict[Tuple[int, int], TeamDiscoveryObservation] = {}
    for item in source_obs:
        if item.tracking_id is not None:
            seg_key = (int(item.shot_id), int(item.tracking_id))
            prev_item = by_seg.get(seg_key)
            if prev_item is None or item.blur_score > prev_item.blur_score:
                by_seg[seg_key] = item
    for appearance in appearances:
        item = by_seg.get(appearance.segment_key)
        if item is None:
            continue
        label, _ = discoverer.predict_track(appearance.tracking_id, shot_id=appearance.shot_id)
        gallery_samples.append(
            (
                SimpleNamespace(legacy_gallery_index=None, raw_label=None, frame_number=item.frame_number),
                item.crop_bgr,
                label,
            )
        )
    return discoverer, gallery_samples, discoverer.report(appearances), cached_frames


def discover_teams_from_video(
    video_path: Path,
    match_config: MatchConfig,
    max_frames: Optional[int] = 300,
) -> Tuple[AutomaticTeamDiscoverer, List[Tuple[Any, np.ndarray, str]], Dict[str, Any]]:
    """Run a tracking pass and fit per-video, unsupervised team appearances.

    This pass is intentionally independent of any crop manifest or team-name
    mapping.  Its output is a track-level model consumed by the tracking and
    possession pass.
    """
    discoverer, gallery_samples, discovery_report, _ = _discover_teams_with_frame_cache(
        video_path=video_path,
        match_config=match_config,
        max_frames=max_frames,
    )
    return discoverer, gallery_samples, discovery_report


def _run_auto_team_pipeline(
    match_cfg: MatchConfig, video_path: Path, output_dir: Path, exp_id: str, max_frames: Optional[int]
) -> Dict[str, Any]:
    """Unattended single-detection-pass inference: discover track appearances, then
    classify tracks using verified track-level descriptors + temporal stabilization
    and estimate confidence-aware possession with Kalman ROI ball recovery.
    """
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_duration = total_video_frames / max(1.0, fps)
    cap.release()

    pitch_region_filter = _build_pitch_region_filter(match_cfg)
    discoverer, gallery, discovery, cached_frames = _discover_teams_with_frame_cache(
        video_path,
        match_cfg,
        max_frames=max_frames,
        fps_override=fps,
        pitch_region_filter=pitch_region_filter,
    )
    report_path = output_dir / "team_discovery_report.json"
    report_path.write_text(json.dumps(discovery, indent=2), encoding="utf-8")
    artifacts: Dict[str, str] = {"team_discovery_report": str(report_path)}
    artifacts["team_cluster_gallery"] = str(
        save_crop_reference_gallery(
            gallery,
            f"{match_cfg.match_name} — Discovered Teams",
            output_dir / "team_cluster_gallery.png",
        )
    )
    artifacts["team_clustering_metrics_csv"] = str(
        write_csv_rows(
            output_dir / "team_clustering_metrics.csv",
            [
                {
                    "shot_id": seg_key[0],
                    "tracking_id": seg_key[1],
                    "discovered_label": pred.predicted_label,
                    "raw_nearest_label": pred.raw_nearest_label,
                    "cosine_distance": pred.nearest_distance,
                    "second_nearest_distance": pred.second_nearest_distance,
                    "margin": pred.margin,
                    "confidence_score": pred.confidence_score,
                    "rejection_reason": pred.rejection_reason or "",
                }
                for seg_key, pred in sorted(discoverer.segment_predictions_.items())
            ],
        )
    )

    writer = cv2.VideoWriter(
        str(output_dir / f"{match_cfg.match_id}_raw.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, match_cfg.detection.frame_stride),
        (width, height),
    )
    aux_detector = FootballDetector(config=match_cfg.detection)
    track_buffer = TrackAppearanceBuffer(
        discoverer=discoverer,
        max_buffer_samples=max(8, match_cfg.team_discovery.max_samples_per_track),
        max_missing_frames=match_cfg.tracking.max_missing_frames_before_reset,
        max_spatial_jump_ratio=match_cfg.tracking.max_spatial_jump_ratio,
    )
    stabilizer = TemporalTeamStabilizer(match_cfg.tracking)
    engine = PossessionEngine(
        match_cfg,
        fps=fps,
        frame_stride=match_cfg.detection.frame_stride,
        total_video_duration_seconds=total_video_duration,
        pitch_region_filter=pitch_region_filter,
    )
    motion_ball_det = (
        FixedCameraMotionBallDetector(
            config=MotionBallConfig(
                enabled=True,
                verify_with_upscaled_yolo=bool(match_cfg.ball_tracking.motion_ball_verify_with_yolo),
            ),
            pitch_region_filter=pitch_region_filter,
        )
        if match_cfg.ball_tracking.enable_motion_ball_recovery
        else None
    )
    observations: List[FrameObservationRecord] = []
    processed = 0

    cap_read = cv2.VideoCapture(str(video_path))
    current_pos = 0
    try:
        for item in cached_frames:
            fno = int(item["frame_number"])
            ts = float(item["timestamp"])
            shot_id = int(item["shot_id"])
            is_cut = bool(item["camera_cut"])
            players = item["players"]
            balls = list(item["balls"])
            per_player_features = item["player_features"]
            per_player_weights = item["player_weights"]

            # Read exact frame sequentially without slow random seeks whenever possible
            if fno < current_pos or (fno - current_pos) > 30:
                cap_read.set(cv2.CAP_PROP_POS_FRAMES, fno)
                current_pos = fno
            frame = None
            while current_pos <= fno:
                ok, frm = cap_read.read()
                current_pos += 1
                if not ok:
                    break
                if current_pos - 1 == fno:
                    frame = frm
            if frame is None:
                continue

            if is_cut:
                track_buffer.on_camera_cut()
                stabilizer.on_camera_cut()

            # Check whether cached detections contain a plausible on-pitch ball; if not, run Kalman ROI & motion recovery
            pred_ball_pos = engine.ball_tracker.get_predicted_position()
            active_blacklist = engine.ball_tracker.blacklisted_static_zones
            plausible_balls = (
                filter_ball_candidates(
                    balls,
                    players,
                    (height, width),
                    min_size=match_cfg.detection.min_ball_box_size_px,
                    max_size=match_cfg.detection.max_ball_box_size_px,
                    blacklisted_zones=active_blacklist,
                    frame_bgr=frame,
                    predicted_pos=pred_ball_pos,
                    filter_crowd_horizon=match_cfg.ball_tracking.filter_crowd_horizon,
                    pitch_region_filter=pitch_region_filter if match_cfg.pitch.filter_balls_by_pitch else None,
                )
                if match_cfg.detection.filter_false_positive_balls and balls
                else balls
            )
            det_source = (
                "tiled_yolo"
                if (plausible_balls and aux_detector.should_use_tiled((height, width)))
                else ("full_frame" if plausible_balls else "none")
            )
            active_balls = list(plausible_balls)

            # Always step fixed-camera motion background model so temporal state stays current
            motion_balls: List[Any] = []
            if motion_ball_det is not None:
                motion_balls = motion_ball_det.detect_candidates(
                    frame_bgr=frame,
                    frame_number=fno,
                    timestamp=ts,
                    player_detections=players,
                    predicted_ball_pos=pred_ball_pos,
                    blacklisted_zones=active_blacklist,
                    yolo_detector=aux_detector if not active_balls else None,
                )

            if not active_balls and match_cfg.detection.enable_roi_ball_search:
                roi_box = engine.get_predicted_ball_roi((height, width))
                if roi_box is not None:
                    roi_balls = aux_detector.detect_ball_in_roi(
                        frame,
                        roi_box,
                        frame_number=fno,
                        timestamp=ts,
                        conf_threshold=match_cfg.detection.roi_conf_threshold,
                        upscale_factor=float(match_cfg.ball_tracking.roi_upscale_factor),
                    )
                    if roi_balls:
                        roi_plausible = (
                            filter_ball_candidates(
                                roi_balls,
                                players,
                                (height, width),
                                min_size=match_cfg.detection.min_ball_box_size_px,
                                max_size=match_cfg.detection.max_ball_box_size_px,
                                blacklisted_zones=active_blacklist,
                                frame_bgr=frame,
                                predicted_pos=pred_ball_pos,
                                filter_crowd_horizon=match_cfg.ball_tracking.filter_crowd_horizon,
                                pitch_region_filter=pitch_region_filter if match_cfg.pitch.filter_balls_by_pitch else None,
                            )
                            if match_cfg.detection.filter_false_positive_balls
                            else roi_balls
                        )
                        if roi_plausible:
                            active_balls = _nms_ball_detections(roi_plausible)
                            det_source = "kalman_roi_recovery"

            if not active_balls and motion_balls:
                active_balls = _nms_ball_detections(motion_balls)
                det_source = "motion_assisted"

            frame_obs: List[FrameObservationRecord] = []
            for player, feat, q_weight in zip(players, per_player_features, per_player_weights):
                # Always update online track buffer (keyed by shot_id + tracking_id)
                buf_pred = track_buffer.classify_observation(
                    player.tracking_id,
                    fno,
                    player.bbox,
                    None,
                    feature=feat,
                    quality_weight=q_weight,
                )
                # When this exact verified track segment qualified in Pass 1, use its
                # full-track aggregated descriptor classification; otherwise use online buffer.
                seg_key = (shot_id, int(player.tracking_id)) if player.tracking_id is not None else None
                if seg_key is not None and seg_key in discoverer.segment_predictions_:
                    prediction = discoverer.segment_predictions_[seg_key]
                else:
                    prediction = buf_pred

                label, stability, locked = stabilizer.update(player.tracking_id, fno, player.bbox, prediction)
                raw_ball = max(active_balls, key=lambda b: b.confidence) if active_balls else None
                record = FrameObservationRecord(
                    match_cfg.match_id,
                    fno,
                    ts,
                    player.tracking_id,
                    label,
                    match_cfg.to_display_label(label),
                    prediction.predicted_label,
                    player.bbox,
                    player.confidence,
                    prediction.confidence_score,
                    prediction.nearest_distance,
                    stability,
                    locked,
                    raw_ball.bbox if raw_ball else None,
                    raw_ball.confidence if raw_ball else None,
                )
                frame_obs.append(record)
                observations.append(record)

            ball_rec, poss_rec, _ = engine.step_frame(
                fno, ts, frame_obs, active_balls, players, frame, (height, width), detection_source=det_source
            )
            writer.write(annotate_match_frame(frame, frame_obs, active_balls, match_cfg, ball_rec, poss_rec))
            processed += 1
    finally:
        cap_read.release()
        writer.release()

    wall_time = round(time.perf_counter() - t_start, 2)
    final_video = finalize_mp4_video(output_dir / f"{match_cfg.match_id}_raw.mp4", output_dir / "annotated_match.mp4")
    artifacts["annotated_video"] = str(final_video)
    artifacts["frame_observations_csv"] = str(export_possession_foundation_records(output_dir / "frame_observations.csv", observations))
    if match_cfg.pitch.calibration_path:
        pitch_rows = []
        planar_cal = None
        if pitch_region_filter is None:
            planar_cal = load_pitch_calibration(match_cfg.pitch.calibration_path)
        for observation in observations:
            x1, _, x2, y2 = observation.player_bbox
            if pitch_region_filter is not None:
                position = pitch_region_filter.project_pixel_to_pitch_m((x1 + x2) / 2.0, y2)
            elif planar_cal is not None:
                position = planar_cal.project_pixel((x1 + x2) / 2.0, y2)
            else:
                position = None
            if position is not None:
                pitch_rows.append(
                    {
                        "frame_number": observation.frame_number,
                        "timestamp": observation.timestamp,
                        "tracking_id": observation.player_id,
                        "team_id": observation.team_id,
                        "pitch_x_m": round(float(position[0]), 3),
                        "pitch_y_m": round(float(position[1]), 3),
                    }
                )
        artifacts["pitch_positions_csv"] = str(write_csv_rows(output_dir / "pitch_positions.csv", pitch_rows))
    artifacts["ball_tracking_diagnostics_csv"] = str(export_ball_trajectory_records(output_dir / "ball_tracking_diagnostics.csv", engine.ball_trajectory))
    artifacts["possession_timeline_csv"] = str(export_possession_timeline_records(output_dir / "possession_timeline.csv", engine.timeline))
    summary = engine.summarize()
    possession_files = generate_possession_report(
        output_dir,
        summary,
        {"metrics_available": False, "validation_blocker": "missing_verified_manual_possession_annotations"},
    )
    artifacts.update({key: str(value) for key, value in possession_files.items()})
    ball_summary = engine.ball_tracker.summarize()
    ball_metrics = {
        "metrics_available": False,
        "validation_blocker": "missing_verified_manual_detection_annotations",
        "total_frames_processed": processed,
        "avg_players_per_frame": round(len(observations) / float(max(1, processed)), 2),
        "frames_with_ball": ball_summary["observed_frames"],
        "ball_coverage_pct": ball_summary["observed_coverage_pct"],
        "observed_coverage_pct": ball_summary["observed_coverage_pct"],
    }
    ball_metrics_path = output_dir / "ball_detection_metrics.json"
    ball_metrics_path.write_text(json.dumps(ball_metrics, indent=2), encoding="utf-8")
    artifacts["ball_detection_metrics"] = str(ball_metrics_path)
    write_csv_rows(output_dir / "tracking_comparison.csv", [])
    classification = {
        "mode": "unsupervised_per_video_adaptation",
        "feature_method": match_cfg.team_discovery.feature_method,
        "classifier_type": match_cfg.team_discovery.clustering_method,
        "metrics_available": False,
        "validation_blocker": "No independent labels supplied.",
        "team_discovery": discovery,
        "overall_accuracy": None,
        "selective_accuracy": None,
        "macro_f1": None,
        "positive_macro_f1": None,
        "coverage": None,
        "unknown_rejection_rate": None,
        "per_class": {},
    }
    tracking = stabilizer.summarize_diagnostics(processed)
    tracking["tracker_config"] = match_cfg.tracking.tracker_config
    experiment_metadata = {
        "experiment_id": exp_id,
        "match_id": match_cfg.match_id,
        "match_name": match_cfg.match_name,
        "mode": "auto_teams_unsupervised",
        "video_resolution": f"{width}x{height}",
        "video_fps": round(fps, 2),
        "frame_stride": match_cfg.detection.frame_stride,
        "processed_frames": processed,
        "tracker_config": match_cfg.tracking.tracker_config,
        "detection_mode": match_cfg.detection.inference_mode,
        "detector_model": match_cfg.detection.model_name,
        "person_conf": match_cfg.detection.person_conf,
        "tile_size": f"{match_cfg.detection.tile_width}x{match_cfg.detection.tile_height}",
        "tile_imgsz": match_cfg.detection.tile_imgsz,
        "fixed_camera": bool(match_cfg.ball_tracking.fixed_camera),
        "pitch_region_filter": pitch_region_filter.source_description if pitch_region_filter else None,
        "motion_ball_recovery": bool(match_cfg.ball_tracking.enable_motion_ball_recovery),
        "feature_method": match_cfg.team_discovery.feature_method,
        "clustering_method": match_cfg.team_discovery.clustering_method,
        "processing_wall_time_seconds": wall_time,
        "average_processing_fps": round(processed / max(0.01, wall_time), 2),
    }
    report_files = generate_consolidated_report(
        output_dir,
        match_cfg.match_id,
        match_cfg.match_name,
        ball_metrics,
        classification,
        tracking,
        possession_summary=summary,
        artifacts_generated=artifacts,
        experiment_metadata=experiment_metadata,
    )
    return {
        "experiment_id": exp_id,
        "output_dir": str(output_dir),
        "experiment_metadata": experiment_metadata,
        "classification_summary": classification,
        "detection_summary": ball_metrics,
        "tracking_summary": tracking,
        "possession_summary": summary,
        "report_files": {key: str(value) for key, value in report_files.items()},
        "artifacts": artifacts,
    }


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

    if match_cfg.team_discovery.enabled:
        return _run_auto_team_pipeline(match_cfg, video_path, output_dir, exp_id, max_tracking_frames)

    artifacts: Dict[str, str] = {}
    # Supervised mode remains available for the existing labelled workflows.
    dataset = prepare_dataset_for_match(match_cfg, base_dir=root)
    train_samples = dataset.get_labelled_samples(split="train", exclude_unknown=True, verified_only=True)
    val_samples = dataset.get_labelled_samples(split="val", exclude_unknown=False, verified_only=True)
    p_tr = save_crop_reference_gallery(
            train_samples,
            f"{match_cfg.match_name} — Training Set ({len(train_samples)} samples)",
            output_dir / "gallery_train_samples.png",
        )
    artifacts["gallery_train"] = str(p_tr)
    p_val = save_crop_reference_gallery(
            val_samples,
            f"{match_cfg.match_name} — Validation Set ({len(val_samples)} samples)",
            output_dir / "gallery_val_samples.png",
        )
    artifacts["gallery_val"] = str(p_val)

    # 2. Audit labels
    all_labelled = dataset.get_labelled_samples(exclude_unknown=False, verified_only=True)
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

            # If no plausible ball found on this frame and Kalman ROI search is available, try ROI crop search
            plausible_balls = (
                filter_ball_candidates(
                    ball_dets,
                    player_dets,
                    (height, width),
                    min_size=match_cfg.detection.min_ball_box_size_px,
                    max_size=match_cfg.detection.max_ball_box_size_px,
                    blacklisted_zones=possession_engine.ball_tracker.blacklisted_static_zones
                    if possession_engine is not None
                    else None,
                    frame_bgr=frame_bgr,
                    predicted_pos=possession_engine.ball_tracker.get_predicted_position()
                    if possession_engine is not None
                    else None,
                    filter_crowd_horizon=match_cfg.ball_tracking.filter_crowd_horizon,
                )
                if match_cfg.detection.filter_false_positive_balls and ball_dets
                else ball_dets
            )
            if not plausible_balls and possession_engine is not None:
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
