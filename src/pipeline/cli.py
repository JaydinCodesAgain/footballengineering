"""Command-line interface for the Football AI Engineering pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_match_config
from src.pipeline.runner import run_match_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Football AI Engineering Pipeline — Detection, Classification, Tracking & Possession"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/norway_wales.yaml",
        help="Path to the match YAML configuration file.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional name for the output directory under outputs/.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Maximum number of video frames to process during tracking inference.",
    )
    parser.add_argument(
        "--compare-trackers",
        action="store_true",
        help="Run ByteTrack vs BoT-SORT comparison on the match video.",
    )
    parser.add_argument(
        "--tracker-comparison-frames",
        type=int,
        default=90,
        help="Number of frames to use when comparing ByteTrack and BoT-SORT.",
    )
    parser.add_argument(
        "--enable-possession",
        action="store_true",
        help="Enable Kalman ball tracking, player ownership inference, and possession analytics.",
    )
    parser.add_argument(
        "--disable-possession",
        action="store_true",
        help="Disable possession analytics even if enabled in the match YAML config.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=None,
        help="Optional override for ball detection confidence threshold.",
    )
    parser.add_argument(
        "--max-coast-frames",
        type=int,
        default=None,
        help="Optional override for maximum Kalman ball interpolation coast frames.",
    )
    parser.add_argument(
        "--control-distance",
        type=float,
        default=None,
        help="Optional override for normalized player-to-ball control distance threshold.",
    )
    parser.add_argument(
        "--min-control-frames",
        type=int,
        default=None,
        help="Optional override for minimum control hysteresis frames.",
    )

    args = parser.parse_args()
    match_cfg = load_match_config(Path(args.config))

    if args.enable_possession:
        match_cfg.possession.enabled = True
    if args.disable_possession:
        match_cfg.possession.enabled = False
    if args.ball_conf is not None:
        match_cfg.detection.ball_conf = float(args.ball_conf)
    if args.max_coast_frames is not None:
        match_cfg.ball_tracking.max_coast_frames = int(args.max_coast_frames)
    if args.control_distance is not None:
        match_cfg.possession.control_distance_threshold = float(args.control_distance)
    if args.min_control_frames is not None:
        match_cfg.possession.min_control_frames = int(args.min_control_frames)

    result = run_match_pipeline(
        config=match_cfg,
        experiment_name=args.experiment_name,
        max_tracking_frames=args.max_frames,
        compare_trackers=args.compare_trackers,
        tracker_comparison_frames=args.tracker_comparison_frames,
    )

    poss = result.get("possession_summary", {})
    attr = poss.get("attributed_possession_share", {})
    b_trk = poss.get("ball_tracking", {})

    summary = {
        "experiment_id": result["experiment_id"],
        "output_dir": result["output_dir"],
        "classification_accuracy": result["classification_summary"]["overall_accuracy"],
        "classification_macro_f1": result["classification_summary"]["macro_f1"],
        "coverage": result["classification_summary"]["coverage"],
        "tracking_frames": result["tracking_summary"]["processed_frames"],
        "distinct_tracks": result["tracking_summary"]["distinct_tracking_ids"],
        "raw_label_changes": result["tracking_summary"]["raw_team_label_changes"],
        "smoothed_label_changes": result["tracking_summary"]["smoothed_team_label_changes"],
        "stable_label_coverage": result["tracking_summary"]["stable_label_coverage"],
        "ball_observed_coverage_pct": b_trk.get(
            "observed_coverage_pct", result["detection_summary"]["ball_coverage_pct"]
        ),
        "ball_tracked_or_interpolated_pct": b_trk.get("tracked_or_interpolated_pct"),
        "attributed_possession_share": attr,
        "possession_coverage_pct": poss.get("possession_coverage_pct"),
        "report_files": result["report_files"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
