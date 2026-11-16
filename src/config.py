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
    ball_conf: float = 0.25
    frame_stride: int = 2
    iou_threshold: float = 0.50


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
    tracker_config: str = "bytetrack.yaml"
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


@dataclass
class SplitConfig:
    train_end_fraction: float = 0.40
    val_start_fraction: float = 0.55
    val_end_fraction: float = 0.80
    test_start_fraction: float = 0.80


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
    )

