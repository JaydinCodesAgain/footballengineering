"""Command-line interface for the Football AI Engineering pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.runner import run_match_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Football AI Engineering Pipeline — Detection, Classification, and Tracking"
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

    args = parser.parse_args()
    result = run_match_pipeline(
        config=Path(args.config),
        experiment_name=args.experiment_name,
        max_tracking_frames=args.max_frames,
        compare_trackers=args.compare_trackers,
        tracker_comparison_frames=args.tracker_comparison_frames,
    )

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
        "report_files": result["report_files"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

