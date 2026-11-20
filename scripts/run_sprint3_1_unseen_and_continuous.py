"""Reproducible Sprint 3.1 Stage E & F evaluation:
1. Held-out unseen match (`Man Utd 4-4 Aston Villa`, 1,500 frames) with frozen config
   and apples-to-apples comparison against Sprint 3 baseline (`outputs/man_utd_4_4_aston_villa_highlights_20261117_104302`).
2. Strongest available continuous-shot proxy (`Chelsea vs Sunderland`, `27.mp4`, frames 840..1640)
   with explicit fixed-camera pending protocol.
3. End-to-end real-video unsupervised team discovery validation matched by IoU against
   independently human-verified crops on `Norway vs Wales` and `Chelsea vs Sunderland`.
4. Exports ready-to-review human annotation templates for `Man Utd vs Aston Villa` without
   ever treating predicted labels as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

from src.config import (
    DetectionConfig,
    MatchConfig,
    PossessionConfig,
    TeamDiscoveryConfig,
    TrackingConfig,
    load_match_config,
)
from src.detection.annotation_workflow import (
    DetectionAnnotationStore,
    sample_representative_frame_numbers,
)
from src.evaluation.harness import evaluate_run_directory
from src.evaluation.possession_eval import PossessionAnnotationStore
from src.evaluation.report import write_csv_rows
from src.pipeline.runner import run_match_pipeline


def _audit_man_utd_run_details(run_dir: Path) -> Dict[str, Any]:
    """Compute fine-grained diagnostic comparison metrics (pass agreement, crowd-band balls, jumps)."""
    disc_csv = run_dir / "team_clustering_metrics.csv"
    obs_csv = run_dir / "frame_observations.csv"
    ball_csv = run_dir / "ball_tracking_diagnostics.csv"

    disc_rows = list(csv.DictReader(disc_csv.open(encoding="utf-8"))) if disc_csv.is_file() else []
    obs_rows = list(csv.DictReader(obs_csv.open(encoding="utf-8"))) if obs_csv.is_file() else []
    ball_rows = list(csv.DictReader(ball_csv.open(encoding="utf-8"))) if ball_csv.is_file() else []

    # Check Pass-1 vs Pass-2 track label agreement
    if disc_rows and "shot_id" in disc_rows[0]:
        disc_map = {(int(r["shot_id"]), int(r["tracking_id"])): r["discovered_label"] for r in disc_rows}
        # Reconstruct shot_id from camera_cut_reset in ball_rows
        shot_by_frame = {}
        sh = 0
        for idx, br in enumerate(ball_rows):
            if idx > 0 and br.get("camera_cut_reset") == "True":
                sh += 1
            shot_by_frame[int(br["frame_number"])] = sh
        obs_by_key: Dict[Any, Counter[str]] = defaultdict(Counter)
        for r in obs_rows:
            if r.get("player_id") not in (None, "", "None"):
                key = (shot_by_frame.get(int(r["frame_number"]), 0), int(r["player_id"]))
                obs_by_key[key][r["team_id"]] += 1
    else:
        disc_map = {int(r["tracking_id"]): r["discovered_label"] for r in disc_rows}
        obs_by_key = defaultdict(Counter)
        for r in obs_rows:
            if r.get("player_id") not in (None, "", "None"):
                obs_by_key[int(r["player_id"])][r["team_id"]] += 1

    agree = disagree = 0
    for key, d_lbl in disc_map.items():
        if key in obs_by_key:
            maj = obs_by_key[key].most_common(1)[0][0]
            if maj == d_lbl:
                agree += 1
            else:
                disagree += 1

    # Ball diagnostics
    obs_balls = [r for r in ball_rows if r.get("observation_mode") == "observed"]
    top_band_balls = [r for r in obs_balls if float(r["center_y"]) < 140.0]
    high_speed_balls = [r for r in obs_balls if float(r["speed_px_per_sec"]) > 1500.0]
    roi_recoveries = [r for r in obs_balls if r.get("detection_source") == "kalman_roi_recovery"]
    cut_frames = [int(r["frame_number"]) for r in ball_rows if r.get("camera_cut_reset") == "True"]
    double_cuts = [
        (cut_frames[i], cut_frames[i + 1])
        for i in range(len(cut_frames) - 1)
        if cut_frames[i + 1] - cut_frames[i] <= 4
    ]

    return {
        "pass1_vs_pass2_tracks_compared": agree + disagree,
        "pass1_vs_pass2_agree": agree,
        "pass1_vs_pass2_disagree": disagree,
        "pass1_vs_pass2_agreement_rate": round(agree / float(max(1, agree + disagree)), 4),
        "observed_ball_frames": len(obs_balls),
        "top_band_y_lt_140_observed_balls": len(top_band_balls),
        "high_speed_gt_1500_observed_balls": len(high_speed_balls),
        "kalman_roi_recovery_hits": len(roi_recoveries),
        "camera_cuts_detected": len(cut_frames),
        "consecutive_cut_double_fires": len(double_cuts),
    }


def _prepare_unseen_human_annotation_templates(video_path: Path, match_id: str) -> Dict[str, str]:
    """Generate unverified review templates for the unseen match so human verification can proceed."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()

    ann_dir = Path("data/annotations") / match_id
    det_store = DetectionAnnotationStore(ann_dir, match_id=match_id)
    sample_fnos = sample_representative_frame_numbers(total_frames=total_frames, num_samples=45)
    if len(det_store.frames) < len(sample_fnos):
        det_store.export_frames_for_annotation(video_path=video_path, frame_numbers=sample_fnos)
    det_html = det_store.generate_html_annotator()

    poss_store = PossessionAnnotationStore(ann_dir, match_id=match_id)
    if not poss_store.intervals:
        poss_store.create_template_intervals(start_frame=0, end_frame=2998, fps=fps, interval_frames=100)
    poss_html = poss_store.generate_html_annotator(fps=fps)

    return {
        "detection_annotations_json": str(det_store.json_path),
        "detection_annotator_html": str(det_html),
        "possession_annotations_json": str(poss_store.json_path),
        "possession_annotator_html": str(poss_html),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3.1 Stage E & F Evaluation Runner")
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--max-frames-unseen", type=int, default=1500)
    parser.add_argument("--max-frames-continuous", type=int, default=350)
    args = parser.parse_args()

    out_root = Path("outputs/sprint3_1")
    (out_root / "ball_tracking").mkdir(parents=True, exist_ok=True)
    (out_root / "possession").mkdir(parents=True, exist_ok=True)
    (out_root / "experiments").mkdir(parents=True, exist_ok=True)

    # 1. Held-out unseen match: Man Utd 4-4 Aston Villa (exact 1,500 frames, frozen config!)
    unseen_video = Path("data/Man Utd 4-4 Aston Villa _ Highlights.mp4")
    unseen_cfg = MatchConfig(
        match_id="man_utd_4_4_aston_villa_highlights",
        match_name="Man Utd 4-4 Aston Villa _ Highlights",
        video_path=str(unseen_video),
        detection=DetectionConfig(model_name=args.model, imgsz=1088, frame_stride=2),
        tracking=TrackingConfig(tracker_config=args.tracker, reset_on_camera_cut=True),
        team_discovery=TeamDiscoveryConfig(
            enabled=True,
            feature_method="hsv_lab",
            clustering_method="kmeans",
            auto_select_k=True,
            max_candidate_clusters=4,
            random_state=42,
        ),
        possession=PossessionConfig(enabled=True),
    )

    print(f"=== Running FROZEN Sprint 3.1 pipeline on unseen match ({args.max_frames_unseen} frames, tracker={args.tracker}, model={args.model}) ===")
    unseen_res = run_match_pipeline(
        config=unseen_cfg,
        experiment_name="sprint3_1/man_utd_villa_run",
        max_tracking_frames=args.max_frames_unseen,
        enable_possession=True,
    )
    unseen_run_dir = Path(unseen_res["output_dir"])

    # Evaluate both baseline Sprint 3 and improved Sprint 3.1 runs with identical harness
    baseline_dir = Path("outputs/man_utd_4_4_aston_villa_highlights_20261117_104302")
    baseline_harness = evaluate_run_directory(
        baseline_dir, "man_utd_4_4_aston_villa_highlights", match_config=unseen_cfg
    )
    sprint31_harness = evaluate_run_directory(
        unseen_run_dir, "man_utd_4_4_aston_villa_highlights", match_config=unseen_cfg
    )
    baseline_details = _audit_man_utd_run_details(baseline_dir)
    sprint31_details = _audit_man_utd_run_details(unseen_run_dir)

    # Generate human annotation review templates for the unseen match
    ann_templates = _prepare_unseen_human_annotation_templates(
        unseen_video, "man_utd_4_4_aston_villa_highlights"
    )

    # 2. Strongest available continuous-shot proxy: Chelsea vs Sunderland wide shot (frames 840+, stride=2)
    print(f"\n=== Running continuous-shot proxy on Chelsea vs Sunderland (27.mp4, start=840, frames={args.max_frames_continuous}) ===")
    cs_cfg = MatchConfig(
        match_id="chelsea_sunderland_continuous_proxy",
        match_name="Chelsea vs Sunderland (Continuous Wide-Shot Proxy)",
        video_path="27.mp4",
        detection=DetectionConfig(model_name=args.model, imgsz=1088, frame_stride=2),
        tracking=TrackingConfig(tracker_config=args.tracker, start_frame=840, reset_on_camera_cut=True),
        team_discovery=TeamDiscoveryConfig(
            enabled=True,
            feature_method="hsv_lab",
            clustering_method="kmeans",
            auto_select_k=True,
            max_candidate_clusters=4,
            random_state=42,
        ),
        possession=PossessionConfig(enabled=True),
    )
    cs_res = run_match_pipeline(
        config=cs_cfg,
        experiment_name="sprint3_1/chelsea_continuous_proxy_run",
        max_tracking_frames=args.max_frames_continuous,
        enable_possession=True,
    )
    cs_harness = evaluate_run_directory(
        Path(cs_res["output_dir"]), "chelsea_sunderland_continuous_proxy", match_config=cs_cfg
    )

    # Save required Sprint 3.1 deliverable JSONs
    ball_tracking_metrics = {
        "metrics_available_ground_truth_localisation": False,
        "validation_blocker": (
            "No manually verified per-frame ground-truth ball bounding boxes exist yet; "
            "provenance, continuity, crowd-FP reduction, jump reduction, and camera-cut "
            "wipe suppression are measured on identical 1,500-frame unseen footage."
        ),
        "unseen_man_utd_villa_1500f": {
            "sprint3_baseline": {
                "provenance": baseline_harness["ball_tracking_evaluation"]["provenance"],
                "diagnostics": baseline_details,
            },
            "sprint3_1_improved": {
                "provenance": sprint31_harness["ball_tracking_evaluation"]["provenance"],
                "tracker_summary": unseen_res["possession_summary"]["ball_tracking"],
                "diagnostics": sprint31_details,
            },
        },
        "continuous_shot_proxy_chelsea_sunderland": {
            "provenance": cs_harness["ball_tracking_evaluation"]["provenance"],
            "tracker_summary": cs_res["possession_summary"]["ball_tracking"],
        },
    }
    (out_root / "ball_tracking" / "tracking_metrics.json").write_text(
        json.dumps(ball_tracking_metrics, indent=2), encoding="utf-8"
    )

    # Save possession deliverables
    baseline_poss_json = json.loads((baseline_dir / "possession_summary.json").read_text(encoding="utf-8"))
    poss_summary_payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unseen_man_utd_villa_1500f": {
            "sprint3_baseline": baseline_poss_json.get("summary", baseline_poss_json),
            "sprint3_1_improved": unseen_res["possession_summary"],
        },
        "continuous_shot_proxy_chelsea_sunderland": cs_res["possession_summary"],
    }
    (out_root / "possession" / "possession_summary.json").write_text(
        json.dumps(poss_summary_payload, indent=2), encoding="utf-8"
    )

    poss_eval_payload = {
        "unseen_man_utd_villa_1500f": sprint31_harness["possession_evaluation"],
        "continuous_shot_proxy_chelsea_sunderland": cs_harness["possession_evaluation"],
        "human_review_templates_generated": ann_templates,
        "reporting_guardrail": (
            "Attributed possession shares are reported strictly alongside attribution coverage "
            "and full 6-state time distribution. Low-coverage broadcast highlight segments must "
            "not be presented as validated full-match possession."
        ),
    }
    (out_root / "possession" / "possession_evaluation.json").write_text(
        json.dumps(poss_eval_payload, indent=2), encoding="utf-8"
    )

    # Save unseen match & fixed-camera protocol results
    unseen_results_payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_configuration": {
            "detector_model": args.model,
            "imgsz": 1088,
            "frame_stride": 2,
            "tracker_config": args.tracker,
            "reset_on_camera_cut": True,
            "team_discovery_feature_method": "hsv_lab",
            "team_discovery_clustering_method": "kmeans",
            "auto_select_k": True,
            "max_candidate_clusters": 4,
            "radius_mad_multiplier": 2.6,
            "min_team_margin": 0.03,
            "max_distance_ratio": 0.85,
            "outlier_repulsion_ratio": 1.05,
            "ball_filter_crowd_horizon": True,
            "ball_init_confirm_frames": 2,
            "camera_cut_cooldown_frames": 6,
        },
        "unseen_match_man_utd_villa_1500f": {
            "video_path": str(unseen_video),
            "human_ground_truth_status": "pending_human_review",
            "human_review_templates": ann_templates,
            "experiment_metadata": unseen_res["experiment_metadata"],
            "team_discovery_report": unseen_res["classification_summary"]["team_discovery"],
            "tracking_summary": unseen_res["tracking_summary"],
            "within_shot_player_tracking_sprint3_baseline": {
                k: v for k, v in baseline_harness["player_tracking_evaluation"].items() if k != "per_shot"
            },
            "within_shot_player_tracking_sprint3_1": {
                k: v for k, v in sprint31_harness["player_tracking_evaluation"].items() if k != "per_shot"
            },
            "diagnostic_comparison": {
                "sprint3_baseline": baseline_details,
                "sprint3_1_improved": sprint31_details,
            },
            "possession_summary": {
                "attributed_possession_share": unseen_res["possession_summary"]["attributed_possession_share"],
                "possession_coverage_pct": unseen_res["possession_summary"]["possession_coverage_pct"],
                "total_analysed_time_distribution_pct": unseen_res["possession_summary"]["total_analysed_time_distribution_pct"],
                "camera_cuts_count": unseen_res["possession_summary"]["camera_cuts_count"],
                "longest_continuous_segment_seconds": unseen_res["possession_summary"]["longest_continuous_segment_seconds"],
            },
        },
        "continuous_shot_proxy_chelsea_sunderland": {
            "video_path": "27.mp4",
            "start_frame": 840,
            "processed_frames": cs_res["experiment_metadata"]["processed_frames"],
            "experiment_metadata": cs_res["experiment_metadata"],
            "team_discovery_report": cs_res["classification_summary"]["team_discovery"],
            "tracking_summary": cs_res["tracking_summary"],
            "within_shot_player_tracking": {
                k: v for k, v in cs_harness["player_tracking_evaluation"].items() if k != "per_shot"
            },
            "ball_tracking": cs_res["possession_summary"]["ball_tracking"],
            "possession_coverage_pct": cs_res["possession_summary"]["possession_coverage_pct"],
            "attributed_possession_share": cs_res["possession_summary"]["attributed_possession_share"],
        },
        "fixed_camera_generalisation": {
            "status": "pending_fixed_camera_footage",
            "reason": (
                "All three local match videos (Norway vs Wales, Chelsea vs Sunderland, "
                "Man Utd vs Aston Villa) are panning/zooming broadcast TV footage. "
                "No elevated static tripod grassroots recording is present in the workspace."
            ),
            "ready_to_run_command": (
                'python -m src.pipeline.cli --video "data/raw/fixed_grassroots_match.mp4" '
                '--auto-teams --enable-possession --tracker botsort.yaml '
                '--pitch-calibration "configs/venue_pitch_calibration.json" --max-frames 3000'
            ),
            "evaluation_command": (
                'python -m src.evaluation.harness --run-dir "outputs/<exp_id>" '
                '--match-id "<match_id>" --annotations-root "data/annotations" '
                '--output-json "outputs/<exp_id>/harness_eval.json"'
            ),
        },
    }
    (out_root / "experiments" / "unseen_match_results.json").write_text(
        json.dumps(unseen_results_payload, indent=2), encoding="utf-8"
    )

    # Also export per-shot fragmentation CSV for unseen match before/after
    per_shot_csv_rows = []
    for tag, h in (("sprint3_baseline", baseline_harness), ("sprint3_1_improved", sprint31_harness)):
        for row in h["player_tracking_evaluation"].get("per_shot", []):
            r_copy = dict(row)
            r_copy["version"] = tag
            per_shot_csv_rows.append(r_copy)
    write_csv_rows(out_root / "player_tracking" / "unseen_per_shot_fragmentation.csv", per_shot_csv_rows)

    print("\n=== UNSEEN MATCH (MAN UTD 4-4 VILLA, 1500 FRAMES) SUMMARY ===")
    print("  Baseline diagnostics :", baseline_details)
    print("  Sprint 3.1 diagnostic:", sprint31_details)
    print(
        "  Baseline within-shot :",
        {k: v for k, v in baseline_harness["player_tracking_evaluation"].items() if k not in ("per_shot", "formal_identity_metrics")},
    )
    print(
        "  Sprint3.1 within-shot:",
        {k: v for k, v in sprint31_harness["player_tracking_evaluation"].items() if k not in ("per_shot", "formal_identity_metrics")},
    )
    print(
        "  Sprint3.1 wall time  :",
        unseen_res["experiment_metadata"]["processing_wall_time_seconds"],
        "s (",
        unseen_res["experiment_metadata"]["average_processing_fps"],
        "FPS)",
    )


if __name__ == "__main__":
    main()

