"""Accuracy-focused evaluation harness for the Football Vision Engine (Sprint 3.1).

Evaluates unsupervised team discovery, ball detection, ball tracking, within-shot
player tracking, and confidence-aware possession against independently verified
reference data.  Never treats model outputs as verified ground truth; when
independent annotations are absent, emits explicit validation blockers and
generates reviewable sample templates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.classification.dataset import JerseyCropDataset
from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    collect_track_appearances,
    evaluate_discovered_labels,
)
from src.config import MatchConfig, TeamDiscoveryConfig, load_match_config
from src.detection.annotation_workflow import DetectionAnnotationStore
from src.evaluation.detection_eval import evaluate_detections_against_ground_truth
from src.evaluation.possession_eval import (
    PossessionAnnotationStore,
    evaluate_possession_predictions,
)
from src.evaluation.tracking_eval import (
    evaluate_ball_tracking_trajectory,
    evaluate_within_shot_player_tracking,
)
from src.schemas import (
    BallTrajectoryRecord,
    DetectionRecord,
    FrameObservationRecord,
    PossessionFrameRecord,
)


def evaluate_team_discovery_on_verified_crops(
    match_config: MatchConfig,
    crops_root: str | Path,
    discovery_config: Optional[TeamDiscoveryConfig] = None,
    seeds: Sequence[int] = (0, 7, 21, 42, 99),
) -> Dict[str, Any]:
    """Evaluate unsupervised team discovery against an independently human-verified crop dataset.

    Ground-truth labels are NEVER supplied to `AutomaticTeamDiscoverer.fit`; they are
    compared only after fitting via best Team_A/Team_B permutation alignment.
    """
    ds = JerseyCropDataset(Path(crops_root), match_config)
    samples = ds.get_labelled_samples(exclude_unknown=False, verified_only=True)
    if len(samples) < 4:
        return {
            "match_id": match_config.match_id,
            "metrics_available": False,
            "validation_blocker": "insufficient_verified_crop_labels",
            "total_verified_samples": len(samples),
        }

    base_cfg = discovery_config or match_config.team_discovery
    gt_labels = [lbl for _, _, lbl in samples]

    from src.classification.features import get_feature_extractor

    extractor = get_feature_extractor(base_cfg.feature_method)
    precomputed_features = [extractor(crop) for _, crop, _ in samples]

    seed_results: List[Dict[str, Any]] = []
    primary_eval: Optional[Dict[str, Any]] = None
    primary_report: Optional[Dict[str, Any]] = None

    for seed in seeds:
        cfg = TeamDiscoveryConfig(
            enabled=True,
            feature_method=base_cfg.feature_method,
            clustering_method=base_cfg.clustering_method,
            candidate_clusters=base_cfg.candidate_clusters,
            auto_select_k=base_cfg.auto_select_k,
            max_candidate_clusters=base_cfg.max_candidate_clusters,
            min_track_observations=1,
            min_cluster_tracks=base_cfg.min_cluster_tracks,
            max_samples_per_track=base_cfg.max_samples_per_track,
            assignment_distance_multiplier=base_cfg.assignment_distance_multiplier,
            radius_mad_multiplier=base_cfg.radius_mad_multiplier,
            min_team_margin=base_cfg.min_team_margin,
            max_distance_ratio=base_cfg.max_distance_ratio,
            outlier_repulsion_ratio=base_cfg.outlier_repulsion_ratio,
            random_state=int(seed),
        )
        obs = [
            TeamDiscoveryObservation(
                tracking_id=idx,
                frame_number=meta.frame_number,
                timestamp=meta.timestamp,
                crop_bgr=crop,
                blur_score=max(1.0, float(meta.quality.blur_score)),
                grass_ratio=min(0.79, float(meta.quality.grass_ratio)),
                occlusion_ratio=min(0.60, float(meta.quality.occlusion_ratio)),
                feature=precomputed_features[idx],
            )
            for idx, (meta, crop, _) in enumerate(samples)
        ]
        apps = collect_track_appearances(obs, cfg)
        disc = AutomaticTeamDiscoverer(cfg).fit(apps)
        preds = [disc.predict_track(i)[0] for i in range(len(samples))]
        ev = evaluate_discovered_labels(preds, gt_labels)
        seed_results.append(
            {
                "seed": int(seed),
                "macro_f1_on_covered": ev["macro_f1_on_covered"],
                "macro_f1_including_unknown": ev["macro_f1_including_unknown"],
                "overall_accuracy_including_rejected": ev["overall_accuracy_including_rejected"],
                "classification_coverage": ev["classification_coverage"],
                "referee_misclassification_rate": ev["referee_misclassification_rate"],
                "unknown_misclassification_rate": ev["unknown_misclassification_rate"],
                "adjusted_rand_index": ev["adjusted_rand_index"],
            }
        )
        if int(seed) == int(base_cfg.random_state) or primary_eval is None:
            primary_eval = ev
            primary_report = disc.report(apps)

    # Temporal subset stability (chronological half 1 vs half 2)
    ordered_idx = sorted(range(len(samples)), key=lambda i: samples[i][0].frame_number)
    mid = len(ordered_idx) // 2
    temporal_stability: Dict[str, Any] = {}
    for half_name, idxs in (("first_half", ordered_idx[:mid]), ("second_half", ordered_idx[mid:])):
        if len(idxs) < 6:
            continue
        sub_cfg = TeamDiscoveryConfig(
            enabled=True,
            feature_method=base_cfg.feature_method,
            clustering_method=base_cfg.clustering_method,
            candidate_clusters=base_cfg.candidate_clusters,
            auto_select_k=base_cfg.auto_select_k,
            max_candidate_clusters=base_cfg.max_candidate_clusters,
            min_track_observations=1,
            min_cluster_tracks=base_cfg.min_cluster_tracks,
            assignment_distance_multiplier=base_cfg.assignment_distance_multiplier,
            radius_mad_multiplier=base_cfg.radius_mad_multiplier,
            min_team_margin=base_cfg.min_team_margin,
            max_distance_ratio=base_cfg.max_distance_ratio,
            outlier_repulsion_ratio=base_cfg.outlier_repulsion_ratio,
            random_state=int(base_cfg.random_state),
        )
        sub_obs = [
            TeamDiscoveryObservation(
                tracking_id=k,
                frame_number=samples[i][0].frame_number,
                timestamp=samples[i][0].timestamp,
                crop_bgr=samples[i][1],
                blur_score=max(1.0, float(samples[i][0].quality.blur_score)),
                grass_ratio=min(0.79, float(samples[i][0].quality.grass_ratio)),
                occlusion_ratio=min(0.60, float(samples[i][0].quality.occlusion_ratio)),
            )
            for k, i in enumerate(idxs)
        ]
        sub_apps = collect_track_appearances(sub_obs, sub_cfg)
        sub_disc = AutomaticTeamDiscoverer(sub_cfg).fit(sub_apps)
        sub_preds = [sub_disc.predict_track(k)[0] for k in range(len(idxs))]
        sub_gt = [samples[i][2] for i in idxs]
        temporal_stability[half_name] = evaluate_discovered_labels(sub_preds, sub_gt)

    f1_vals = [float(r["macro_f1_on_covered"] or 0.0) for r in seed_results]
    f1_3_vals = [float(r["macro_f1_including_unknown"] or 0.0) for r in seed_results]
    return {
        "match_id": match_config.match_id,
        "feature_method": base_cfg.feature_method,
        "clustering_method": base_cfg.clustering_method,
        "metrics_available": True,
        "discovery_report": primary_report,
        "evaluation": primary_eval,
        "seed_stability": {
            "seeds_tested": list(seeds),
            "macro_f1_on_covered_mean": round(float(np.mean(f1_vals)), 4),
            "macro_f1_on_covered_std": round(float(np.std(f1_vals)), 4),
            "macro_f1_3way_mean": round(float(np.mean(f1_3_vals)), 4),
            "macro_f1_3way_std": round(float(np.std(f1_3_vals)), 4),
            "per_seed": seed_results,
        },
        "temporal_subset_stability": temporal_stability,
    }


def evaluate_run_directory(
    run_dir: str | Path,
    match_id: str,
    annotations_root: str | Path = "data/annotations",
    match_config: Optional[MatchConfig] = None,
) -> Dict[str, Any]:
    """Evaluate all telemetry artifacts exported by a pipeline run against independent GT (if present)."""
    rdir = Path(run_dir)
    ann_dir = Path(annotations_root) / match_id

    # 1. Load frame observations for within-shot tracking evaluation
    obs_csv = rdir / "frame_observations.csv"
    ball_csv = (
        rdir / "ball_tracking_diagnostics.csv"
        if (rdir / "ball_tracking_diagnostics.csv").is_file()
        else rdir / "ball_trajectory.csv"
    )
    poss_csv = rdir / "possession_timeline.csv"

    cut_frames: List[int] = []
    ball_records: List[BallTrajectoryRecord] = []
    if ball_csv.is_file():
        with ball_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bbox_val = None
                if row.get("bbox"):
                    parts = [int(x) for x in row["bbox"].strip("[]").split(",") if x.strip()]
                    if len(parts) == 4:
                        bbox_val = (parts[0], parts[1], parts[2], parts[3])
                is_cut = str(row.get("camera_cut_reset", "")).lower() == "true"
                fno = int(row["frame_number"])
                if is_cut:
                    cut_frames.append(fno)
                ball_records.append(
                    BallTrajectoryRecord(
                        match_id=row.get("match_id", match_id),
                        frame_number=fno,
                        timestamp=float(row["timestamp"]),
                        observation_mode=row["observation_mode"],
                        is_observed=str(row.get("is_observed", "")).lower() == "true",
                        is_interpolated=str(row.get("is_interpolated", "")).lower() == "true",
                        bbox=bbox_val,
                        center_x=float(row["center_x"]) if row.get("center_x") not in (None, "", "None") else None,
                        center_y=float(row["center_y"]) if row.get("center_y") not in (None, "", "None") else None,
                        velocity_x=float(row.get("velocity_x") or 0.0),
                        velocity_y=float(row.get("velocity_y") or 0.0),
                        speed_px_per_sec=float(row.get("speed_px_per_sec") or 0.0),
                        confidence=float(row.get("confidence") or 0.0),
                        uncertainty_radius_px=float(row.get("uncertainty_radius_px") or 999.0),
                        frames_since_observed=int(row.get("frames_since_observed") or 0),
                        camera_cut_reset=is_cut,
                        rejected_outliers_count=int(row.get("rejected_outliers_count") or 0),
                        detection_source=row.get("detection_source", "none"),
                    )
                )

    obs_records: List[FrameObservationRecord] = []
    frame_predictions: Dict[int, List[DetectionRecord]] = {}
    if obs_csv.is_file():
        with obs_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fno = int(row["frame_number"])
                ts = float(row["timestamp"])
                pbox = tuple(int(x) for x in row["player_bbox"].strip("[]").split(","))
                pid = int(row["player_id"]) if row.get("player_id") not in (None, "", "None") else None
                obs_records.append(
                    FrameObservationRecord(
                        match_id=row.get("match_id", match_id),
                        frame_number=fno,
                        timestamp=ts,
                        player_id=pid,
                        team_id=row["team_id"],
                        team_display_name=row.get("team_display_name", row["team_id"]),
                        raw_team_id=row.get("raw_team_id", row["team_id"]),
                        player_bbox=pbox,  # type: ignore[arg-type]
                        player_detection_confidence=float(row.get("player_detection_confidence") or 0.0),
                        classification_confidence=float(row.get("classification_confidence") or 0.0),
                        classification_distance=float(row.get("classification_distance") or 1.0),
                        track_team_confidence=float(row.get("track_team_confidence") or 0.0),
                        is_team_locked=str(row.get("is_team_locked", "")).lower() == "true",
                    )
                )
                frame_predictions.setdefault(fno, []).append(
                    DetectionRecord(fno, ts, 0, "player", pbox, float(row.get("player_detection_confidence") or 0.0), pid)  # type: ignore[arg-type]
                )
    for b_rec in ball_records:
        if b_rec.is_observed and b_rec.bbox is not None:
            frame_predictions.setdefault(b_rec.frame_number, []).append(
                DetectionRecord(b_rec.frame_number, b_rec.timestamp, 32, "ball", b_rec.bbox, b_rec.confidence, None)
            )

    # 2. Detection & Ball Tracking GT evaluation
    det_gt: List[Any] = []
    if (ann_dir / "detection_annotations.json").is_file():
        det_store = DetectionAnnotationStore(ann_dir, match_id=match_id)
        det_gt = det_store.get_verified_ground_truth()
    detection_eval = evaluate_detections_against_ground_truth(frame_predictions, det_gt, iou_threshold=0.50)
    ball_tracking_eval = evaluate_ball_tracking_trajectory(ball_records, verified_annotations=det_gt)

    # 3. Within-shot player tracking evaluation
    player_tracking_eval = evaluate_within_shot_player_tracking(obs_records, camera_cut_frames=cut_frames)

    # 4. Possession GT evaluation
    poss_records: List[PossessionFrameRecord] = []
    if poss_csv.is_file():
        with poss_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                poss_records.append(
                    PossessionFrameRecord(
                        match_id=row.get("match_id", match_id),
                        frame_number=int(row["frame_number"]),
                        timestamp=float(row["timestamp"]),
                        dt_seconds=float(row["dt_seconds"]),
                        ball_observation_mode=row["ball_observation_mode"],
                        ball_visibility=row["ball_visibility"],
                        possession_state=row["possession_state"],
                        raw_interaction_state=row["raw_interaction_state"],
                        controlling_player_id=int(row["controlling_player_id"])
                        if row.get("controlling_player_id") not in (None, "", "None")
                        else None,
                        controlling_team_id=row.get("controlling_team_id") or None,
                        controlling_team_display=row.get("controlling_team_display", "UNKNOWN"),
                        possession_confidence=float(row.get("possession_confidence") or 0.0),
                        closest_player_id=int(row["closest_player_id"])
                        if row.get("closest_player_id") not in (None, "", "None")
                        else None,
                        closest_team_id=row.get("closest_team_id") or None,
                        closest_normalized_distance=float(row["closest_normalized_distance"])
                        if row.get("closest_normalized_distance") not in (None, "", "None")
                        else None,
                        second_closest_team_id=row.get("second_closest_team_id") or None,
                        second_closest_normalized_distance=float(row["second_closest_normalized_distance"])
                        if row.get("second_closest_normalized_distance") not in (None, "", "None")
                        else None,
                        running_team_a_share_pct=float(row.get("running_team_a_share_pct") or 50.0),
                        running_team_b_share_pct=float(row.get("running_team_b_share_pct") or 50.0),
                        camera_cut=str(row.get("camera_cut", "")).lower() == "true",
                    )
                )
    poss_ivs = []
    if (ann_dir / "possession_annotations.json").is_file():
        poss_store = PossessionAnnotationStore(ann_dir, match_id=match_id)
        poss_ivs = poss_store.get_verified_intervals()
    possession_eval = evaluate_possession_predictions(poss_records, poss_ivs, match_config=match_config)

    return {
        "match_id": match_id,
        "run_dir": str(rdir),
        "detection_evaluation": detection_eval,
        "ball_tracking_evaluation": ball_tracking_eval,
        "player_tracking_evaluation": player_tracking_eval,
        "possession_evaluation": possession_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Football Vision Engine — Accuracy Evaluation Harness")
    parser.add_argument("--run-dir", type=str, required=True, help="Pipeline output directory to evaluate.")
    parser.add_argument("--match-id", type=str, required=True, help="Match identifier.")
    parser.add_argument("--config", type=str, default=None, help="Optional match YAML config path.")
    parser.add_argument("--annotations-root", type=str, default="data/annotations")
    parser.add_argument("--crops-dir", type=str, default=None, help="Optional verified crop dataset directory.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON report path.")
    args = parser.parse_args()

    cfg = load_match_config(args.config) if args.config else None
    result = evaluate_run_directory(
        run_dir=args.run_dir,
        match_id=args.match_id,
        annotations_root=args.annotations_root,
        match_config=cfg,
    )
    if args.crops_dir and cfg is not None:
        result["team_discovery_verified_crops_evaluation"] = evaluate_team_discovery_on_verified_crops(
            match_config=cfg, crops_root=args.crops_dir
        )

    payload = json.dumps(result, indent=2)
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
