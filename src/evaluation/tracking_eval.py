"""Tracking evaluation, within-shot fragmentation analysis, ball trajectory evaluation,
manual identity continuity auditing, and ByteTrack vs BoT-SORT comparison.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np

from src.classification.classifier import TeamClassifier
from src.classification.cropper import extract_torso_crop
from src.classification.team_discovery import AutomaticTeamDiscoverer, TrackAppearanceBuffer
from src.config import MatchConfig, TrackingConfig
from src.detection.annotation_workflow import FrameAnnotation
from src.possession.ball_tracker import detect_camera_cut
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    ClassificationPrediction,
    FrameObservationRecord,
    InternalLabel,
)
from src.tracking.stabilizer import TemporalTeamStabilizer
from src.tracking.tracker import MultiObjectTracker


def evaluate_manual_identity_continuity(
    player_id_sequences: Dict[str, Sequence[Optional[int]]],
) -> Dict[str, Any]:
    """
    Evaluate manual player identity continuity sequences (`PLAYER_ID_AUDIT`).
    Computes ID switches, transition stability, and per-target dominant-ID purity (IDF1 proxy).
    Clearly marks formal identity accuracy as unavailable when no sequences are provided.
    """
    if not player_id_sequences:
        return {
            "identity_ground_truth_available": False,
            "status": "unavailable_missing_manual_identity_sequences",
            "message": (
                "No manual ground-truth identity sequences were provided. "
                "Automatic tracking diagnostics report track lifetimes and within-shot continuity only; "
                "formal IDF1 and HOTA require independently verified identity annotations."
            ),
            "idf1": None,
            "hota": None,
            "total_transitions": 0,
            "total_id_switches": 0,
            "id_switch_rate": None,
            "mean_id_purity_idf1_proxy": None,
            "per_player": {},
        }

    total_transitions = 0
    total_switches = 0
    id_purities: List[float] = []
    per_player: Dict[str, Any] = {}

    for player_name, seq in player_id_sequences.items():
        visible_ids = [int(x) for x in seq if x is not None]
        switches = 0
        transitions = 0
        for prev_id, curr_id in zip(seq, seq[1:]):
            if prev_id is None or curr_id is None:
                continue
            transitions += 1
            if int(prev_id) != int(curr_id):
                switches += 1

        if visible_ids:
            dominant_id, dom_count = Counter(visible_ids).most_common(1)[0]
            purity = float(dom_count) / float(len(visible_ids))
            idf1_proxy = (2.0 * dom_count) / float(2 * len(visible_ids))
            id_purities.append(purity)
        else:
            dominant_id = None
            purity = 0.0
            idf1_proxy = 0.0

        total_transitions += transitions
        total_switches += switches

        per_player[player_name] = {
            "visible_observations": len(visible_ids),
            "distinct_assigned_ids": len(set(visible_ids)),
            "dominant_track_id": dominant_id,
            "comparable_transitions": transitions,
            "id_switches": switches,
            "id_switch_rate": round(switches / float(transitions), 4) if transitions > 0 else 0.0,
            "dominant_id_purity": round(purity, 4),
            "idf1_proxy": round(idf1_proxy, 4),
        }

    return {
        "identity_ground_truth_available": True,
        "status": "verified_manual_identity_audit",
        "total_transitions": total_transitions,
        "total_id_switches": total_switches,
        "id_switch_rate": round(total_switches / float(total_transitions), 4)
        if total_transitions > 0
        else 0.0,
        "mean_id_purity_idf1_proxy": round(float(np.mean(id_purities)), 4) if id_purities else 0.0,
        "per_player": per_player,
    }


def evaluate_within_shot_player_tracking(
    observations: Sequence[FrameObservationRecord],
    camera_cut_frames: Sequence[int] = (),
    min_shot_frames: int = 10,
) -> Dict[str, Any]:
    """Measure player tracking continuity and fragmentation strictly within uninterrupted
    camera shots, separating legitimate broadcast camera-cut resets from actual tracker
    fragmentation and detecting cross-cut ID leaks.
    """
    if not observations:
        return {
            "total_observations": 0,
            "distinct_track_ids": 0,
            "camera_cuts_count": len(camera_cut_frames),
            "cross_cut_id_leaks": 0,
            "continuous_shots_analyzed": 0,
            "within_shot_mean_ids_per_player_slot": None,
            "within_shot_short_track_fraction": None,
            "within_shot_mean_track_lifetime": None,
            "within_shot_team_label_consistency": None,
            "formal_identity_metrics": {
                "metrics_available": False,
                "idf1": None,
                "hota": None,
                "validation_blocker": "missing_verified_manual_identity_annotations",
            },
            "per_shot": [],
        }

    cut_set: Set[int] = {int(f) for f in camera_cut_frames}
    frame_numbers_sorted = sorted({int(o.frame_number) for o in observations})
    shot_by_frame: Dict[int, int] = {}
    cur_shot = 0
    for idx, fno in enumerate(frame_numbers_sorted):
        if idx > 0 and fno in cut_set:
            cur_shot += 1
        shot_by_frame[fno] = cur_shot

    shots_by_tid: Dict[int, Set[int]] = defaultdict(set)
    obs_by_shot: Dict[int, List[FrameObservationRecord]] = defaultdict(list)
    frames_by_shot: Dict[int, Set[int]] = defaultdict(set)

    for obs in observations:
        sh = shot_by_frame[int(obs.frame_number)]
        obs_by_shot[sh].append(obs)
        frames_by_shot[sh].add(int(obs.frame_number))
        if obs.player_id is not None:
            shots_by_tid[int(obs.player_id)].add(sh)

    cross_cut_leaks = sum(1 for shs in shots_by_tid.values() if len(shs) > 1)

    per_shot_rows: List[Dict[str, Any]] = []
    weighted_frag_num = 0.0
    weighted_frag_den = 0.0
    total_ws_tracks = 0
    total_ws_short = 0
    ws_lifetimes: List[int] = []
    consistent_obs = 0
    total_known_obs = 0

    for sh in sorted(obs_by_shot.keys()):
        sh_obs = obs_by_shot[sh]
        n_frames = len(frames_by_shot[sh])
        by_tid: Dict[int, List[FrameObservationRecord]] = defaultdict(list)
        for o in sh_obs:
            if o.player_id is not None:
                by_tid[int(o.player_id)].append(o)

        distinct_ids = len(by_tid)
        lengths = [len(v) for v in by_tid.values()]
        short_ids = sum(1 for l in lengths if l < 5)
        avg_players = sum(lengths) / float(max(1, n_frames))
        ids_per_slot = distinct_ids / float(max(1.0, avg_players))

        # Track-level team label consistency within this shot
        sh_consistent = 0
        sh_known = 0
        sh_team_switches = 0
        for tid_obs in by_tid.values():
            known_lbls = [
                o.team_id
                for o in tid_obs
                if o.team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
            ]
            if known_lbls:
                dom_cnt = Counter(known_lbls).most_common(1)[0][1]
                sh_consistent += dom_cnt
                sh_known += len(known_lbls)
                for a, b in zip(known_lbls, known_lbls[1:]):
                    if a != b:
                        sh_team_switches += 1

        if n_frames >= min_shot_frames and distinct_ids > 0:
            weighted_frag_num += ids_per_slot * n_frames
            weighted_frag_den += n_frames
            total_ws_tracks += distinct_ids
            total_ws_short += short_ids
            ws_lifetimes.extend(lengths)
            consistent_obs += sh_consistent
            total_known_obs += sh_known

        f_list = sorted(frames_by_shot[sh])
        per_shot_rows.append(
            {
                "shot_id": sh,
                "start_frame": f_list[0],
                "end_frame": f_list[-1],
                "frames_count": n_frames,
                "distinct_track_ids": distinct_ids,
                "avg_players_per_frame": round(avg_players, 2),
                "ids_per_player_slot": round(ids_per_slot, 3),
                "short_tracks_under_5_obs": short_ids,
                "short_track_fraction": round(short_ids / float(max(1, distinct_ids)), 4),
                "mean_track_lifetime": round(float(np.mean(lengths)), 2) if lengths else 0.0,
                "max_track_lifetime": max(lengths) if lengths else 0,
                "within_shot_team_switches": sh_team_switches,
                "within_shot_team_consistency": round(sh_consistent / float(sh_known), 4)
                if sh_known > 0
                else None,
            }
        )

    qualifying_shots = [r for r in per_shot_rows if r["frames_count"] >= min_shot_frames]
    return {
        "total_observations": len(observations),
        "distinct_track_ids": len(shots_by_tid),
        "camera_cuts_count": len(cut_set),
        "total_shots": len(per_shot_rows),
        "continuous_shots_analyzed": len(qualifying_shots),
        "min_shot_frames_threshold": min_shot_frames,
        "cross_cut_id_leaks": int(cross_cut_leaks),
        "within_shot_mean_ids_per_player_slot": round(weighted_frag_num / weighted_frag_den, 3)
        if weighted_frag_den > 0
        else None,
        "within_shot_short_track_fraction": round(total_ws_short / float(total_ws_tracks), 4)
        if total_ws_tracks > 0
        else None,
        "within_shot_mean_track_lifetime": round(float(np.mean(ws_lifetimes)), 2)
        if ws_lifetimes
        else None,
        "within_shot_team_label_consistency": round(consistent_obs / float(total_known_obs), 4)
        if total_known_obs > 0
        else None,
        "formal_identity_metrics": {
            "metrics_available": False,
            "idf1": None,
            "hota": None,
            "validation_blocker": "missing_verified_manual_identity_annotations",
        },
        "per_shot": per_shot_rows,
    }


def evaluate_ball_tracking_trajectory(
    trajectory_records: Sequence[BallTrajectoryRecord],
    verified_annotations: Optional[Sequence[FrameAnnotation]] = None,
    match_distance_px: float = 30.0,
) -> Dict[str, Any]:
    """Evaluate ball tracking provenance, continuity, and (when verified GT boxes exist)
    localisation accuracy, track breaks, reacquisition correctness, and false associations.
    """
    total = len(trajectory_records)
    obs_recs = [r for r in trajectory_records if r.observation_mode == BallObservationMode.OBSERVED.value]
    interp_recs = [r for r in trajectory_records if r.observation_mode == BallObservationMode.INTERPOLATED.value]
    lost_recs = [r for r in trajectory_records if r.observation_mode == BallObservationMode.LOST.value]

    # Within-shot continuity & jump diagnostics
    seg_lengths: List[int] = []
    cur_len = 0
    jumps_px: List[float] = []
    prev_xy: Optional[Tuple[float, float]] = None
    for r in trajectory_records:
        if r.camera_cut_reset:
            if cur_len > 0:
                seg_lengths.append(cur_len)
                cur_len = 0
            prev_xy = None
        if r.is_observed or r.is_interpolated:
            cur_len += 1
            if r.is_observed and r.center_x is not None and r.center_y is not None:
                if prev_xy is not None:
                    jumps_px.append(math.hypot(r.center_x - prev_xy[0], r.center_y - prev_xy[1]))
                prev_xy = (float(r.center_x), float(r.center_y))
        else:
            if cur_len > 0:
                seg_lengths.append(cur_len)
                cur_len = 0
            prev_xy = None
    if cur_len > 0:
        seg_lengths.append(cur_len)

    provenance = {
        "total_frames": total,
        "observed_frames": len(obs_recs),
        "interpolated_frames": len(interp_recs),
        "lost_frames": len(lost_recs),
        "observed_rate_pct": round(len(obs_recs) / float(max(1, total)) * 100.0, 2),
        "interpolated_rate_pct": round(len(interp_recs) / float(max(1, total)) * 100.0, 2),
        "lost_rate_pct": round(len(lost_recs) / float(max(1, total)) * 100.0, 2),
        "active_rate_pct": round((len(obs_recs) + len(interp_recs)) / float(max(1, total)) * 100.0, 2),
        "camera_cut_resets": sum(1 for r in trajectory_records if r.camera_cut_reset),
        "mean_continuous_track_frames": round(float(np.mean(seg_lengths)), 2) if seg_lengths else 0.0,
        "max_continuous_track_frames": max(seg_lengths) if seg_lengths else 0,
        "large_observed_jumps_over_150px": sum(1 for j in jumps_px if j > 150.0),
        "median_observed_step_px": round(float(np.median(jumps_px)), 2) if jumps_px else 0.0,
        "p95_observed_step_px": round(float(np.percentile(jumps_px, 95)), 2) if jumps_px else 0.0,
    }

    verified = [
        a for a in (verified_annotations or []) if a.verified and a.source == "manual"
    ]
    if not verified:
        return {
            "metrics_available": False,
            "validation_blocker": "missing_verified_manual_detection_annotations",
            "provenance": provenance,
            "ground_truth_metrics": None,
        }

    rec_by_frame = {int(r.frame_number): r for r in trajectory_records}
    obs_errors: List[float] = []
    interp_errors: List[float] = []
    all_errors: List[float] = []
    gt_visible_frames = 0
    matched_active = 0
    false_associations = 0
    track_breaks = 0
    reacquisitions_total = 0
    reacquisitions_correct = 0
    prev_was_matched = False
    prev_was_lost = False

    for ann in sorted(verified, key=lambda a: a.frame_number):
        fno = int(ann.frame_number)
        rec = rec_by_frame.get(fno)
        if rec is None:
            continue
        gt_balls = [b.bbox for b in ann.boxes if b.class_name == "ball"]
        pred_active = (rec.is_observed or rec.is_interpolated) and (
            rec.center_x is not None and rec.center_y is not None
        )

        if not gt_balls:
            if pred_active:
                false_associations += 1
            prev_was_matched = False
            prev_was_lost = not pred_active
            continue

        gt_visible_frames += 1
        gb = gt_balls[0]
        gcx, gcy = (gb[0] + gb[2]) / 2.0, (gb[1] + gb[3]) / 2.0

        if not pred_active:
            if prev_was_matched and not rec.camera_cut_reset:
                track_breaks += 1
            prev_was_matched = False
            prev_was_lost = True
            continue

        assert rec.center_x is not None and rec.center_y is not None
        err = math.hypot(float(rec.center_x) - gcx, float(rec.center_y) - gcy)
        is_match = err <= match_distance_px
        if is_match:
            matched_active += 1
            all_errors.append(err)
            if rec.is_observed:
                obs_errors.append(err)
            else:
                interp_errors.append(err)
        else:
            false_associations += 1
            if prev_was_matched and not rec.camera_cut_reset:
                track_breaks += 1

        if prev_was_lost and rec.is_observed:
            reacquisitions_total += 1
            if is_match:
                reacquisitions_correct += 1

        prev_was_matched = is_match
        prev_was_lost = False

    return {
        "metrics_available": True,
        "validation_blocker": None,
        "provenance": provenance,
        "ground_truth_metrics": {
            "evaluated_gt_frames": len(verified),
            "gt_ball_visible_frames": gt_visible_frames,
            "match_distance_threshold_px": match_distance_px,
            "trajectory_recall_on_visible": round(matched_active / float(max(1, gt_visible_frames)), 4),
            "mean_localisation_error_px": round(float(np.mean(all_errors)), 2) if all_errors else None,
            "observed_mean_error_px": round(float(np.mean(obs_errors)), 2) if obs_errors else None,
            "interpolated_mean_error_px": round(float(np.mean(interp_errors)), 2) if interp_errors else None,
            "track_breaks": int(track_breaks),
            "false_associations": int(false_associations),
            "reacquisitions_evaluated": int(reacquisitions_total),
            "reacquisition_accuracy": round(reacquisitions_correct / float(reacquisitions_total), 4)
            if reacquisitions_total > 0
            else None,
        },
    }


def compare_tracking_algorithms(
    video_path: str | Path,
    match_config: MatchConfig,
    classifier: Union[TeamClassifier, AutomaticTeamDiscoverer],
    tracker_configs: Sequence[str] = ("bytetrack.yaml", "botsort.yaml"),
    max_frames: int = 120,
) -> Dict[str, Dict[str, Any]]:
    """
    Run an apples-to-apples comparison between tracking algorithms (e.g. ByteTrack vs BoT-SORT)
    on the exact same video segment, detector parameters, and team classifier/discoverer.
    Separates camera-cut resets from within-shot fragmentation and records runtime.
    """
    comparison: Dict[str, Dict[str, Any]] = {}
    is_discoverer = isinstance(classifier, AutomaticTeamDiscoverer)

    for tracker_name in tracker_configs:
        trk_cfg = TrackingConfig(
            tracker_config=tracker_name,
            start_frame=match_config.tracking.start_frame,
            history_length=match_config.tracking.history_length,
            min_votes=match_config.tracking.min_votes,
            min_agreement=match_config.tracking.min_agreement,
            confidence_weighted=match_config.tracking.confidence_weighted,
            lock_confidence_threshold=match_config.tracking.lock_confidence_threshold,
            lock_min_observations=match_config.tracking.lock_min_observations,
            reclassify_contradiction_streak=match_config.tracking.reclassify_contradiction_streak,
            max_missing_frames_before_reset=match_config.tracking.max_missing_frames_before_reset,
            max_spatial_jump_ratio=match_config.tracking.max_spatial_jump_ratio,
            reset_on_camera_cut=match_config.tracking.reset_on_camera_cut,
        )
        tracker = MultiObjectTracker(
            detection_config=match_config.detection,
            tracking_config=trk_cfg,
        )
        stabilizer = TemporalTeamStabilizer(config=trk_cfg)
        track_buf = (
            TrackAppearanceBuffer(
                discoverer=classifier,
                max_buffer_samples=max(8, match_config.team_discovery.max_samples_per_track),
                max_missing_frames=trk_cfg.max_missing_frames_before_reset,
                max_spatial_jump_ratio=trk_cfg.max_spatial_jump_ratio,
            )
            if is_discoverer
            else None
        )

        processed = 0
        ball_frames = 0
        obs_records: List[FrameObservationRecord] = []
        cut_frames: List[int] = []
        prev_small: Optional[np.ndarray] = None
        prev_med_h: Optional[float] = None
        last_cut_frame: Optional[int] = None
        cooldown = max(0, int(match_config.ball_tracking.camera_cut_cooldown_frames))

        t0 = time.perf_counter()
        for fno, ts, frame_bgr, detections in tracker.track_video(video_path, max_frames=max_frames):
            processed += 1
            player_dets = [d for d in detections if d.class_name == "player"]
            ball_dets = [d for d in detections if d.class_name == "ball"]
            if ball_dets:
                ball_frames += 1

            heights = [float(d.bbox[3] - d.bbox[1]) for d in player_dets if (d.bbox[3] - d.bbox[1]) > 15]
            med_h = float(np.median(heights)) if heights else None
            small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
            if prev_small is not None and detect_camera_cut(
                prev_small,
                small_curr,
                corr_threshold=match_config.ball_tracking.camera_cut_hist_threshold,
                prev_median_player_h=prev_med_h,
                curr_median_player_h=med_h,
            ):
                if last_cut_frame is None or (fno - last_cut_frame) > cooldown:
                    cut_frames.append(int(fno))
                    last_cut_frame = int(fno)
                    stabilizer.on_camera_cut()
                    if track_buf is not None:
                        track_buf.on_camera_cut()
            prev_small = small_curr
            if med_h is not None:
                prev_med_h = med_h

            all_p_bboxes = [d.bbox for d in player_dets]
            for det in player_dets:
                crop, _, quality = extract_torso_crop(
                    frame_bgr,
                    det.bbox,
                    config=match_config.cropping,
                    other_player_bboxes=all_p_bboxes,
                )
                if is_discoverer and track_buf is not None:
                    q_w = (1.0 - quality.grass_ratio) * (1.0 - quality.occlusion_ratio) if quality.is_valid else 0.0
                    pred = track_buf.classify_observation(
                        det.tracking_id, fno, det.bbox, crop, quality_weight=q_w
                    )
                elif crop is not None:
                    pred = classifier.predict_crop(crop, match_id=match_config.match_id)  # type: ignore[union-attr]
                else:
                    pred = ClassificationPrediction(
                        predicted_label=InternalLabel.UNKNOWN.value,
                        raw_nearest_label=InternalLabel.UNKNOWN.value,
                        nearest_distance=1.0,
                        second_nearest_distance=1.0,
                        margin=0.0,
                        confidence_score=0.0,
                        is_unknown=True,
                        rejection_reason="invalid_or_occluded_crop",
                    )
                stable_lbl, trk_conf, is_locked = stabilizer.update(det.tracking_id, fno, det.bbox, pred)
                obs_records.append(
                    FrameObservationRecord(
                        match_id=match_config.match_id,
                        frame_number=fno,
                        timestamp=ts,
                        player_id=det.tracking_id,
                        team_id=stable_lbl,
                        team_display_name=match_config.to_display_label(stable_lbl),
                        raw_team_id=pred.predicted_label,
                        player_bbox=det.bbox,
                        player_detection_confidence=det.confidence,
                        classification_confidence=pred.confidence_score,
                        classification_distance=pred.nearest_distance,
                        track_team_confidence=trk_conf,
                        is_team_locked=is_locked,
                    )
                )

        wall_s = time.perf_counter() - t0
        diag = stabilizer.summarize_diagnostics(processed_frames=processed)
        within_shot = evaluate_within_shot_player_tracking(obs_records, camera_cut_frames=cut_frames)
        diag["tracker_config"] = tracker_name
        diag["frames_with_ball"] = ball_frames
        diag["ball_frame_coverage"] = round(ball_frames / float(max(1, processed)), 4)
        diag["processing_wall_time_seconds"] = round(wall_s, 2)
        diag["ms_per_frame"] = round((wall_s / float(max(1, processed))) * 1000.0, 1)
        diag["camera_cuts_detected"] = len(cut_frames)
        diag["cross_cut_id_leaks"] = within_shot["cross_cut_id_leaks"]
        diag["within_shot_mean_ids_per_player_slot"] = within_shot["within_shot_mean_ids_per_player_slot"]
        diag["within_shot_short_track_fraction"] = within_shot["within_shot_short_track_fraction"]
        diag["within_shot_mean_track_lifetime"] = within_shot["within_shot_mean_track_lifetime"]
        diag["within_shot_team_label_consistency"] = within_shot["within_shot_team_label_consistency"]
        diag["formal_identity_metrics"] = within_shot["formal_identity_metrics"]
        comparison[tracker_name] = diag

    return comparison
