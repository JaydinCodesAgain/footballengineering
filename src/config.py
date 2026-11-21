"""Configuration management for matches, detection, classification, and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from src.schemas import InternalLabel


@dataclass
class DetectionConfig:
    model_name: str = "yolo11n.pt"
    imgsz: int = 1088
    person_conf: float = 0.40
    ball_conf: float = 0.20
    frame_stride: int = 2
    iou_threshold: float = 0.50
    enable_roi_ball_search: bool = True
    roi_size_px: int = 640
    roi_conf_threshold: float = 0.14
    enable_tiled_ball_search: bool = True
    tile_conf_threshold: float = 0.16
    min_ball_box_size_px: int = 5
    max_ball_box_size_px: int = 140
    filter_false_positive_balls: bool = True
    # Sprint 3.2: native-resolution overlapping multi-tile player + ball detection
    inference_mode: str = "full_frame"  # "full_frame" | "tiled" | "auto"
    auto_tile_min_width_px: int = 2400
    tile_width: int = 1280
    tile_height: int = 1280
    tile_overlap: float = 0.22
    tile_imgsz: int = 1280
    tile_batch_size: int = 4
    tile_auto_content_crop: bool = True
    player_nms_iou: float = 0.45
    player_containment_iom: float = 0.72
    ball_nms_iou: float = 0.35
    include_full_frame_in_tiled: bool = False


@dataclass
class CroppingConfig:
    min_player_height: int = 30
    min_player_width: int = 12
    min_crop_width: int = 8
    min_crop_height: int = 10
    left_frac: float = 0.22
    right_frac: float = 0.78
    top_frac: float = 0.18
    bottom_frac: float = 0.55
    min_blur_variance: float = 8.0
    max_grass_ratio: float = 0.80
    max_occlusion_ratio: float = 0.65
    grass_hsv_low: Tuple[int, int, int] = (33, 35, 25)
    grass_hsv_high: Tuple[int, int, int] = (95, 255, 255)


@dataclass
class ClassificationConfig:
    feature_method: str = "spatial_embedding"
    classifier_type: str = "knn"
    n_neighbors: int = 3
    metric: str = "cosine"
    unknown_threshold: float = 0.35
    auto_calibrate_threshold: bool = True
    min_margin: float = 0.02
    balance_class_weights: bool = True
    random_state: int = 42


@dataclass
class TrackingConfig:
    tracker_config: str = "botsort.yaml"
    start_frame: int = 0
    history_length: int = 9
    min_votes: int = 3
    min_agreement: float = 0.60
    confidence_weighted: bool = True
    lock_confidence_threshold: float = 0.78
    lock_min_observations: int = 6
    reclassify_contradiction_streak: int = 5
    max_missing_frames_before_reset: int = 30
    max_spatial_jump_ratio: float = 3.5
    reset_on_camera_cut: bool = True


@dataclass
class SplitConfig:
    train_end_fraction: float = 0.40
    val_start_fraction: float = 0.55
    val_end_fraction: float = 0.80
    test_start_fraction: float = 0.80


@dataclass
class BallTrackingConfig:
    max_coast_frames: int = 5
    max_jump_px_per_frame: float = 165.0
    reacquire_confirm_frames: int = 2
    velocity_damping: float = 0.85
    interpolation_conf_decay: float = 0.78
    camera_cut_hist_threshold: float = 0.52
    camera_cut_cooldown_frames: int = 6
    process_noise_scale: float = 8.0
    measurement_noise_scale: float = 12.0
    init_confirm_frames: int = 2
    init_high_conf_threshold: float = 0.45
    static_zone_ttl_frames: int = 250
    filter_crowd_horizon: bool = True
    # Sprint 3.2: fixed-camera persistent static-background distractor mask & cut control
    fixed_camera: bool = False
    persistent_static_distractors: bool = False
    static_distractor_radius_px: float = 24.0
    static_distractor_min_hits: int = 4
    enable_motion_ball_recovery: bool = False
    motion_ball_verify_with_yolo: bool = False
    roi_upscale_factor: float = 1.0


@dataclass
class PossessionConfig:
    enabled: bool = True
    control_distance_threshold: float = 1.35
    contested_distance_threshold: float = 1.15
    contested_margin_threshold: float = 0.22
    loose_ball_speed_threshold_px_s: float = 520.0
    min_control_frames: int = 2
    max_pass_retention_frames: int = 6
    min_ownership_confidence: float = 0.28
    exclude_referees: bool = True
    attribute_unknown_goalkeepers: bool = False
    pitch_y_min_frac: float = 0.08
    pitch_y_max_frac: float = 0.96
    pitch_x_min_frac: float = 0.02
    pitch_x_max_frac: float = 0.98


@dataclass
class TeamDiscoveryConfig:
    """Configuration for per-video, label-free team appearance discovery."""

    enabled: bool = False
    feature_method: str = "hsv_lab"
    clustering_method: str = "kmeans"
    candidate_clusters: int = 3
    auto_select_k: bool = True
    max_candidate_clusters: int = 4
    min_track_observations: int = 3
    min_cluster_tracks: int = 2
    max_samples_per_track: int = 8
    assignment_distance_multiplier: float = 1.0
    radius_mad_multiplier: float = 2.6
    min_team_margin: float = 0.03
    max_distance_ratio: float = 0.85
    outlier_repulsion_ratio: float = 1.05
    random_state: int = 42


@dataclass
class PitchConfig:
    """Optional fixed-camera pitch calibration & geometric pitch-region filtering."""

    calibration_path: Optional[str] = None
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    enable_region_filter: bool = True
    coordinate_space: str = "raw_fisheye"  # "raw_fisheye" | "undistorted" | "planar"
    player_margin_m: float = 2.0
    ball_margin_m: float = 2.5
    filter_players_by_pitch: bool = True
    filter_balls_by_pitch: bool = True
    use_pitch_roi_for_tiling: bool = True


@dataclass
class MatchConfig:
    """Configuration for a specific football match and pipeline run."""

    match_id: str
    match_name: str
    video_path: str
    fallback_video_paths: List[str] = field(default_factory=list)
    label_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            InternalLabel.TEAM_A.value: "Team A",
            InternalLabel.TEAM_B.value: "Team B",
            InternalLabel.REFEREE.value: "REF",
            InternalLabel.UNKNOWN.value: "Unknown",
        }
    )
    label_aliases: Dict[str, str] = field(default_factory=dict)
    team_colours_bgr: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            InternalLabel.TEAM_A.value: (0, 0, 255),
            InternalLabel.TEAM_B.value: (255, 0, 0),
            InternalLabel.REFEREE.value: (0, 255, 255),
            InternalLabel.UNKNOWN.value: (160, 160, 160),
        }
    )
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cropping: CroppingConfig = field(default_factory=CroppingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    ball_tracking: BallTrackingConfig = field(default_factory=BallTrackingConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    team_discovery: TeamDiscoveryConfig = field(default_factory=TeamDiscoveryConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)

    def resolve_video_path(self, base_dir: Optional[Path] = None) -> Path:
        """Resolve the video path across primary and fallback locations."""
        root = base_dir or Path.cwd()
        candidates = [self.video_path] + list(self.fallback_video_paths)
        for candidate in candidates:
            p = Path(candidate)
            if p.is_file():
                return p
            rel = root / candidate
            if rel.is_file():
                return rel
        return root / self.video_path

    def to_display_label(self, internal_label: str) -> str:
        """Map an internal generic label (e.g. 'Team_A') to the match display name (e.g. 'Norway')."""
        return self.label_mapping.get(internal_label, internal_label)

    def to_internal_label(self, raw_or_display_label: str) -> str:
        """Map a match-specific team name or alias (e.g. 'Norway', 'REF', 'Other') to an InternalLabel."""
        cleaned = str(raw_or_display_label).strip()
        if cleaned in InternalLabel.all_labels():
            return cleaned
        if cleaned in self.label_aliases:
            return self.label_aliases[cleaned]
        for internal_key, display_val in self.label_mapping.items():
            if display_val.lower() == cleaned.lower():
                return internal_key
        lower = cleaned.lower()
        if lower in ("ref", "referee", "official"):
            return InternalLabel.REFEREE.value
        if lower in ("unknown", "other", "ambiguous", "none", "unlabelled"):
            return InternalLabel.UNKNOWN.value
        raise ValueError(
            f"Label '{raw_or_display_label}' is not compatible with match '{self.match_id}'. "
            f"Valid mappings: {self.label_mapping}"
        )

    def get_colour_bgr(self, label: str) -> Tuple[int, int, int]:
        """Return the BGR annotation colour for either an internal or display label."""
        try:
            internal = self.to_internal_label(label)
        except ValueError:
            internal = InternalLabel.UNKNOWN.value
        colour = self.team_colours_bgr.get(internal, (160, 160, 160))
        return (int(colour[0]), int(colour[1]), int(colour[2]))


def load_match_config(config_path: str | Path) -> MatchConfig:
    """Load and validate a YAML match configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    if "match_id" not in raw or "match_name" not in raw:
        raise ValueError(f"Configuration {path} must specify 'match_id' and 'match_name'.")

    det_raw = raw.get("detection", {})
    crop_raw = raw.get("cropping", {})
    if "grass_hsv_low" in crop_raw:
        crop_raw["grass_hsv_low"] = tuple(crop_raw["grass_hsv_low"])
    if "grass_hsv_high" in crop_raw:
        crop_raw["grass_hsv_high"] = tuple(crop_raw["grass_hsv_high"])

    cls_raw = raw.get("classification", {})
    trk_raw = raw.get("tracking", {})
    spl_raw = raw.get("splits", {})
    btrk_raw = raw.get("ball_tracking", {})
    poss_raw = raw.get("possession", {})
    discovery_raw = raw.get("team_discovery", {})
    pitch_raw = raw.get("pitch", {})

    colours_raw = raw.get("team_colours_bgr", {})
    colours_parsed: Dict[str, Tuple[int, int, int]] = {
        InternalLabel.TEAM_A.value: (0, 0, 255),
        InternalLabel.TEAM_B.value: (255, 0, 0),
        InternalLabel.REFEREE.value: (0, 255, 255),
        InternalLabel.UNKNOWN.value: (160, 160, 160),
    }
    for k, v in colours_raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 3:
            colours_parsed[k] = (int(v[0]), int(v[1]), int(v[2]))

    label_mapping = {
        InternalLabel.TEAM_A.value: "Team A",
        InternalLabel.TEAM_B.value: "Team B",
        InternalLabel.REFEREE.value: "REF",
        InternalLabel.UNKNOWN.value: "Unknown",
    }
    label_mapping.update(raw.get("label_mapping", {}))

    return MatchConfig(
        match_id=str(raw["match_id"]),
        match_name=str(raw["match_name"]),
        video_path=str(raw.get("video_path", "")),
        fallback_video_paths=list(raw.get("fallback_video_paths", [])),
        label_mapping=label_mapping,
        label_aliases=dict(raw.get("label_aliases", {})),
        team_colours_bgr=colours_parsed,
        detection=DetectionConfig(**det_raw),
        cropping=CroppingConfig(**crop_raw),
        classification=ClassificationConfig(**cls_raw),
        tracking=TrackingConfig(**trk_raw),
        splits=SplitConfig(**spl_raw),
        ball_tracking=BallTrackingConfig(**btrk_raw),
        possession=PossessionConfig(**poss_raw),
        team_discovery=TeamDiscoveryConfig(**discovery_raw),
        pitch=PitchConfig(**pitch_raw),
    )

