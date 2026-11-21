"""Command-line interface for the Football AI Engineering pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from src.config import MatchConfig, TeamDiscoveryConfig, load_match_config
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
    parser.add_argument("--video", type=str, default=None, help="Video path for a new, label-free match.")
    parser.add_argument("--auto-teams", action="store_true", help="Discover two visual teams from this video; no crop manifest or kit colours are used.")
    parser.add_argument("--team-clustering", choices=("kmeans", "gmm", "agglomerative", "dbscan", "hdbscan"), default="kmeans")
    parser.add_argument("--tracker", type=str, default=None, help="Optional override for tracker YAML (e.g. bytetrack.yaml, botsort.yaml).")
    parser.add_argument("--pitch-calibration", type=str, default=None, help="JSON file mapping fixed-camera pixels to pitch metres.")
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
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional override for starting video frame index.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Optional override for frame sampling stride.",
    )
    # Sprint 3.2: panoramic multi-tile detection, fixed-camera pitch filtering & motion-ball recovery
    parser.add_argument(
        "--detection-mode",
        choices=("full_frame", "tiled", "auto"),
        default=None,
        help="Detection mode: full_frame (default broadcast), tiled (overlapping native-res tiles), or auto.",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional override for YOLO detector weights (e.g. yolo11n.pt, yolo11s.pt).")
    parser.add_argument("--person-conf", type=float, default=None, help="Optional override for player detection confidence threshold.")
    parser.add_argument("--tile-width", type=int, default=None, help="Optional tile width in pixels for tiled mode.")
    parser.add_argument("--tile-height", type=int, default=None, help="Optional tile height in pixels for tiled mode.")
    parser.add_argument("--tile-overlap", type=float, default=None, help="Optional tile overlap fraction (0.0..0.49).")
    parser.add_argument("--tile-imgsz", type=int, default=None, help="Optional YOLO imgsz per tile.")
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Enable fixed-camera mode (persistent static-distractor mask, disables broadcast camera-cut/GMC, enables motion-assisted small-ball recovery).",
    )
    parser.add_argument(
        "--pitch-coordinate-space",
        choices=("raw_fisheye", "undistorted", "planar"),
        default=None,
        help="Coordinate space of the supplied pitch calibration (default: raw_fisheye).",
    )
    parser.add_argument("--pitch-player-margin", type=float, default=None, help="Pitch boundary margin in metres for players (default: 2.0).")
    parser.add_argument("--pitch-ball-margin", type=float, default=None, help="Pitch boundary margin in metres for balls (default: 2.5).")
    parser.add_argument("--disable-pitch-filter", action="store_true", help="Disable geometric pitch-region filtering even when pitch calibration is supplied.")
    parser.add_argument("--enable-motion-ball", action="store_true", help="Explicitly enable fixed-camera motion-assisted small-ball recovery.")
    parser.add_argument("--disable-motion-ball", action="store_true", help="Disable motion-assisted small-ball recovery.")

    args = parser.parse_args()
    if args.auto_teams and args.video:
        match_id = re.sub(r"[^a-z0-9]+", "_", Path(args.video).stem.lower()).strip("_") or "unseen_match"
        match_cfg = MatchConfig(
            match_id=match_id,
            match_name=Path(args.video).stem,
            video_path=args.video,
            team_discovery=TeamDiscoveryConfig(enabled=True, clustering_method=args.team_clustering),
        )
    else:
        match_cfg = load_match_config(Path(args.config))
        if args.auto_teams:
            match_cfg.team_discovery.enabled = True
            match_cfg.team_discovery.clustering_method = args.team_clustering
        if args.video:
            match_cfg.video_path = args.video

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
    if args.start_frame is not None:
        match_cfg.tracking.start_frame = int(args.start_frame)
    if args.frame_stride is not None:
        match_cfg.detection.frame_stride = int(args.frame_stride)
    if args.pitch_calibration:
        match_cfg.pitch.calibration_path = args.pitch_calibration
    if args.tracker:
        match_cfg.tracking.tracker_config = str(args.tracker)
    if args.detection_mode:
        match_cfg.detection.inference_mode = str(args.detection_mode)
    if args.model:
        match_cfg.detection.model_name = str(args.model)
    if args.person_conf is not None:
        match_cfg.detection.person_conf = float(args.person_conf)
    if args.tile_width is not None:
        match_cfg.detection.tile_width = int(args.tile_width)
    if args.tile_height is not None:
        match_cfg.detection.tile_height = int(args.tile_height)
    if args.tile_overlap is not None:
        match_cfg.detection.tile_overlap = float(args.tile_overlap)
    if args.tile_imgsz is not None:
        match_cfg.detection.tile_imgsz = int(args.tile_imgsz)
    if args.fixed_camera:
        match_cfg.ball_tracking.fixed_camera = True
        match_cfg.ball_tracking.persistent_static_distractors = True
        match_cfg.ball_tracking.enable_motion_ball_recovery = True
        match_cfg.ball_tracking.roi_upscale_factor = 2.0
        match_cfg.ball_tracking.init_high_conf_threshold = 0.30
        match_cfg.detection.roi_size_px = 416
        match_cfg.tracking.reset_on_camera_cut = False
    if args.pitch_coordinate_space:
        match_cfg.pitch.coordinate_space = str(args.pitch_coordinate_space)
    if args.pitch_player_margin is not None:
        match_cfg.pitch.player_margin_m = float(args.pitch_player_margin)
    if args.pitch_ball_margin is not None:
        match_cfg.pitch.ball_margin_m = float(args.pitch_ball_margin)
    if args.disable_pitch_filter:
        match_cfg.pitch.enable_region_filter = False
    if args.enable_motion_ball:
        match_cfg.ball_tracking.enable_motion_ball_recovery = True
    if args.disable_motion_ball:
        match_cfg.ball_tracking.enable_motion_ball_recovery = False

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
