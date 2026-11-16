"""Core data structures and schemas for the Football AI Engineering pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class InternalLabel(str, Enum):
    """Generic internal class labels decoupled from specific club or country names."""

    TEAM_A = "Team_A"
    TEAM_B = "Team_B"
    REFEREE = "Referee"
    UNKNOWN = "Unknown"

    @classmethod
    def valid_training_labels(cls) -> Tuple[str, ...]:
        """Labels valid as supervised positive classes during classifier training."""
        return (cls.TEAM_A.value, cls.TEAM_B.value, cls.REFEREE.value)

    @classmethod
    def all_labels(cls) -> Tuple[str, ...]:
        """All internal labels including Unknown."""
        return (cls.TEAM_A.value, cls.TEAM_B.value, cls.REFEREE.value, cls.UNKNOWN.value)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / float(self.height) if self.height > 0 else 0.0

    def is_valid(self, min_width: int = 1, min_height: int = 1) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1 and self.width >= min_width and self.height >= min_height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def to_list(self) -> List[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    @classmethod
    def from_coords(cls, coords: Any) -> "BoundingBox":
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates for BoundingBox, got {coords}")
        x1, y1, x2, y2 = [int(round(float(c))) for c in coords]
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass
class DetectionRecord:
    """Single object detection or tracked observation in a frame."""

    frame_number: int
    timestamp: float
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    tracking_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CropQualityMetrics:
    """Quality and contamination diagnostics for a torso crop."""

    width: int
    height: int
    blur_score: float
    grass_ratio: float
    occlusion_ratio: float
    edge_clipped: bool
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JerseyCropMetadata:
    """Complete metadata record for an extracted player torso crop."""

    crop_id: str
    match_id: str
    frame_number: int
    timestamp: float
    tracking_id: Optional[int]
    bbox: Tuple[int, int, int, int]
    torso_bbox: Tuple[int, int, int, int]
    ground_truth_label: Optional[str] = None
    raw_label: Optional[str] = None
    is_labelled: bool = False
    label_verified: bool = False
    split: str = "unassigned"
    image_path: str = ""
    quality: CropQualityMetrics = field(
        default_factory=lambda: CropQualityMetrics(
            width=0,
            height=0,
            blur_score=0.0,
            grass_ratio=0.0,
            occlusion_ratio=0.0,
            edge_clipped=False,
            is_valid=True,
        )
    )
    legacy_gallery_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["torso_bbox"] = list(self.torso_bbox)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JerseyCropMetadata":
        q_raw = data.get("quality", {})
        quality = CropQualityMetrics(**q_raw) if isinstance(q_raw, dict) else q_raw
        return cls(
            crop_id=str(data["crop_id"]),
            match_id=str(data["match_id"]),
            frame_number=int(data["frame_number"]),
            timestamp=float(data["timestamp"]),
            tracking_id=int(data["tracking_id"]) if data.get("tracking_id") is not None else None,
            bbox=tuple(int(x) for x in data["bbox"]),  # type: ignore[arg-type]
            torso_bbox=tuple(int(x) for x in data["torso_bbox"]),  # type: ignore[arg-type]
            ground_truth_label=data.get("ground_truth_label"),
            raw_label=data.get("raw_label"),
            is_labelled=bool(data.get("is_labelled", False)),
            label_verified=bool(data.get("label_verified", False)),
            split=str(data.get("split", "unassigned")),
            image_path=str(data.get("image_path", "")),
            quality=quality,
            legacy_gallery_index=int(data["legacy_gallery_index"])
            if data.get("legacy_gallery_index") is not None
            else None,
        )


@dataclass
class ClassificationPrediction:
    """Result of classifying a single jersey crop, including distance/uncertainty."""

    predicted_label: str
    raw_nearest_label: str
    nearest_distance: float
    second_nearest_distance: float
    margin: float
    confidence_score: float
    is_unknown: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameObservationRecord:
    """Structured per-player observation record designed as the foundation for possession & tactical analytics."""

    match_id: str
    frame_number: int
    timestamp: float
    player_id: Optional[int]
    team_id: str
    team_display_name: str
    raw_team_id: str
    player_bbox: Tuple[int, int, int, int]
    player_detection_confidence: float
    classification_confidence: float
    classification_distance: float
    track_team_confidence: float
    is_team_locked: bool
    ball_bbox: Optional[Tuple[int, int, int, int]] = None
    ball_detection_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["player_bbox"] = list(self.player_bbox)
        if self.ball_bbox is not None:
            data["ball_bbox"] = list(self.ball_bbox)
        return data

