"""Tracking evaluation, manual identity continuity auditing, and ByteTrack vs BoT-SORT comparison."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.classification.classifier import TeamClassifier
from src.classification.cropper import extract_torso_crop
from src.config import MatchConfig, TrackingConfig
from src.schemas import ClassificationPrediction, InternalLabel
from src.tracking.stabilizer import TemporalTeamStabilizer
from src.tracking.tracker import MultiObjectTracker


def evaluate_manual_identity_continuity(
    player_id_sequences: Dict[str, Sequence[Optional[int]]],
) -> Dict[str, Any]:
    """
    Evaluate manual player identity continuity sequences (Cell 17 `PLAYER_ID_AUDIT`).
    Computes ID switches, transition stability, and per-target dominant-ID purity (IDF1 proxy).
    Clearly marks formal identity accuracy as unavailable when no sequences are provided.
    """
    if not player_id_sequences:
        return {
            "identity_ground_truth_available": False,
            "status": "unavailable_missing_manual_identity_sequences",
            "message": (
                "No manual ground-truth identity sequences were provided. "
                "Automatic tracking diagnostics report track lifetimes and label stability only."
            ),
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


def compare_tracking_algorithms(
    video_path: str | Path,
    match_config: MatchConfig,
    classifier: TeamClassifier,
    tracker_configs: Sequence[str] = ("bytetrack.yaml", "botsort.yaml"),
    max_frames: int = 120,
) -> Dict[str, Dict[str, Any]]:
    """
    Run an apples-to-apples comparison between tracking algorithms (e.g. ByteTrack vs BoT-SORT)
    on the exact same video segment, detector parameters, and team classifier.
    """
    comparison: Dict[str, Dict[str, Any]] = {}

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
        )
        tracker = MultiObjectTracker(
            detection_config=match_config.detection,
            tracking_config=trk_cfg,
        )
        stabilizer = TemporalTeamStabilizer(config=trk_cfg)

        processed = 0
        ball_frames = 0
        for fno, _, frame_bgr, detections in tracker.track_video(video_path, max_frames=max_frames):
            processed += 1
            player_dets = [d for d in detections if d.class_name == "player"]
            ball_dets = [d for d in detections if d.class_name == "ball"]
            if ball_dets:
                ball_frames += 1

            all_p_bboxes = [d.bbox for d in player_dets]
            for det in player_dets:
                crop, _, _ = extract_torso_crop(
                    frame_bgr,
                    det.bbox,
                    config=match_config.cropping,
                    other_player_bboxes=all_p_bboxes,
                )
                if crop is not None:
                    pred = classifier.predict_crop(crop, match_id=match_config.match_id)
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
                stabilizer.update(det.tracking_id, fno, det.bbox, pred)

        diag = stabilizer.summarize_diagnostics(processed_frames=processed)
        diag["tracker_config"] = tracker_name
        diag["frames_with_ball"] = ball_frames
        diag["ball_frame_coverage"] = round(ball_frames / float(max(1, processed)), 4)
        comparison[tracker_name] = diag

    return comparison

