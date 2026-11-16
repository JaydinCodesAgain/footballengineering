"""Ground-truth bounding-box annotation workflow for player and ball detection evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import cv2
import numpy as np


@dataclass
class AnnotatedBox:
    """Single ground-truth bounding box annotation."""

    class_name: str  # "player" or "ball"
    bbox: List[int]  # [x1, y1, x2, y2]

    def validate(self) -> None:
        if self.class_name not in ("player", "ball"):
            raise ValueError(f"Invalid class_name '{self.class_name}'. Expected 'player' or 'ball'.")
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError(f"Invalid bounding box coordinates: {self.bbox}")


@dataclass
class FrameAnnotation:
    """Ground-truth annotation state for a single video frame."""

    match_id: str
    frame_number: int
    timestamp: float
    image_rel_path: str
    verified: bool = False
    source: str = "unlabelled"  # MUST be "manual" with verified=True for independent evaluation
    boxes: List[AnnotatedBox] = field(default_factory=list)
    legacy_counts: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameAnnotation":
        boxes = [AnnotatedBox(class_name=b["class_name"], bbox=list(b["bbox"])) for b in data.get("boxes", [])]
        return cls(
            match_id=str(data["match_id"]),
            frame_number=int(data["frame_number"]),
            timestamp=float(data.get("timestamp", 0.0)),
            image_rel_path=str(data.get("image_rel_path", "")),
            verified=bool(data.get("verified", False)),
            source=str(data.get("source", "unlabelled")),
            boxes=boxes,
            legacy_counts=data.get("legacy_counts"),
        )


class DetectionAnnotationStore:
    """
    Manages frame-level bounding-box annotations on disk via JSON and CSV files.
    Enforces strict provenance: model-predicted proposals are NEVER treated as verified ground truth.
    """

    def __init__(self, root_dir: str | Path, match_id: str) -> None:
        self.root_dir = Path(root_dir)
        self.match_id = match_id
        self.frames_dir = self.root_dir / "frames"
        self.json_path = self.root_dir / "detection_annotations.json"
        self.csv_path = self.root_dir / "detection_boxes.csv"
        self.frames: Dict[int, FrameAnnotation] = {}

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.json_path.is_file():
            self.load()

    def export_frames_for_annotation(
        self,
        video_path: str | Path,
        frame_numbers: Sequence[int],
        legacy_count_records: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> List[FrameAnnotation]:
        """
        Extract specific video frames to `frames/frame_<id>.jpg` and initialize empty
        annotation templates ready for manual bounding-box labelling.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video for frame extraction: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        exported: List[FrameAnnotation] = []

        try:
            for fno in sorted(set(int(f) for f in frame_numbers)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rel_path = f"frames/frame_{fno:06d}.jpg"
                cv2.imwrite(str(self.root_dir / rel_path), frame)

                if fno not in self.frames:
                    self.frames[fno] = FrameAnnotation(
                        match_id=self.match_id,
                        frame_number=fno,
                        timestamp=round(fno / fps, 3),
                        image_rel_path=rel_path,
                        verified=False,
                        source="unlabelled",
                        boxes=[],
                        legacy_counts=(legacy_count_records or {}).get(fno),
                    )
                elif legacy_count_records and fno in legacy_count_records:
                    self.frames[fno].legacy_counts = legacy_count_records[fno]

                exported.append(self.frames[fno])
        finally:
            cap.release()

        self.save()
        return exported

    def set_manual_frame_boxes(
        self,
        frame_number: int,
        boxes: Sequence[AnnotatedBox],
        timestamp: float = 0.0,
        image_rel_path: str = "",
        verified: bool = True,
    ) -> FrameAnnotation:
        """Record manually verified ground-truth bounding boxes for a frame."""
        validated_boxes: List[AnnotatedBox] = []
        for b in boxes:
            b.validate()
            validated_boxes.append(b)

        existing = self.frames.get(frame_number)
        ann = FrameAnnotation(
            match_id=self.match_id,
            frame_number=int(frame_number),
            timestamp=existing.timestamp if existing else round(float(timestamp), 3),
            image_rel_path=existing.image_rel_path if existing else image_rel_path,
            verified=bool(verified),
            source="manual",
            boxes=validated_boxes,
            legacy_counts=existing.legacy_counts if existing else None,
        )
        self.frames[int(frame_number)] = ann
        return ann

    def save(self) -> None:
        """Save annotations to JSON and a flat human-editable CSV table."""
        ordered = [self.frames[k] for k in sorted(self.frames.keys())]
        verified_frames = [
            f for f in ordered if f.verified and f.source == "manual"
        ]
        payload = {
            "match_id": self.match_id,
            "total_frames": len(ordered),
            "verified_manual_box_frames": len(verified_frames),
            "evaluation_ready": len(verified_frames) > 0,
            "frames": [f.to_dict() for f in ordered],
        }
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        fieldnames = ["frame_number", "timestamp", "class_name", "x1", "y1", "x2", "y2", "verified", "source"]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for f_ann in ordered:
                if not f_ann.boxes:
                    writer.writerow(
                        {
                            "frame_number": f_ann.frame_number,
                            "timestamp": f_ann.timestamp,
                            "class_name": "",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "verified": f_ann.verified,
                            "source": f_ann.source,
                        }
                    )
                else:
                    for b in f_ann.boxes:
                        writer.writerow(
                            {
                                "frame_number": f_ann.frame_number,
                                "timestamp": f_ann.timestamp,
                                "class_name": b.class_name,
                                "x1": b.bbox[0],
                                "y1": b.bbox[1],
                                "x2": b.bbox[2],
                                "y2": b.bbox[3],
                                "verified": f_ann.verified,
                                "source": f_ann.source,
                            }
                        )

    def load(self) -> None:
        """Load annotations from `detection_annotations.json`."""
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.frames.clear()
        for item in data.get("frames", []):
            ann = FrameAnnotation.from_dict(item)
            self.frames[ann.frame_number] = ann

    def import_boxes_from_csv(self, csv_path: Optional[str | Path] = None) -> int:
        """
        Load manual bounding box annotations from `detection_boxes.csv` (or a specified CSV file).
        Rejects any row whose `source` is `'model_prediction'` or `'auto'` from being marked `verified`.
        """
        target = Path(csv_path) if csv_path else self.csv_path
        if not target.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {target}")

        grouped_boxes: Dict[int, List[AnnotatedBox]] = {}
        frame_meta: Dict[int, Tuple[float, bool, str]] = {}

        with target.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fno_str = row.get("frame_number", "").strip()
                if not fno_str:
                    continue
                fno = int(fno_str)
                ts = float(row.get("timestamp", 0.0) or 0.0)
                ver = row.get("verified", "false").strip().lower() in ("true", "1", "yes", "y")
                src = row.get("source", "manual").strip() or "manual"
                if src.lower() in ("model", "model_prediction", "auto", "predicted", "yolo"):
                    raise ValueError(
                        f"Frame {fno}: Ground-truth boxes cannot be imported with source='{src}'. "
                        "Predictions must never be presented as independent ground truth."
                    )

                frame_meta[fno] = (ts, ver, src)
                grouped_boxes.setdefault(fno, [])

                cls_name = row.get("class_name", "").strip()
                if cls_name:
                    x1 = int(float(row["x1"]))
                    y1 = int(float(row["y1"]))
                    x2 = int(float(row["x2"]))
                    y2 = int(float(row["y2"]))
                    box = AnnotatedBox(class_name=cls_name, bbox=[x1, y1, x2, y2])
                    box.validate()
                    grouped_boxes[fno].append(box)

        updated = 0
        for fno, boxes in grouped_boxes.items():
            ts, ver, src = frame_meta[fno]
            existing = self.frames.get(fno)
            self.frames[fno] = FrameAnnotation(
                match_id=self.match_id,
                frame_number=fno,
                timestamp=existing.timestamp if existing else ts,
                image_rel_path=existing.image_rel_path if existing else f"frames/frame_{fno:06d}.jpg",
                verified=bool(ver and src == "manual"),
                source=src,
                boxes=boxes,
                legacy_counts=existing.legacy_counts if existing else None,
            )
            updated += 1

        self.save()
        return updated

    def get_verified_ground_truth(self) -> List[FrameAnnotation]:
        """
        Return ONLY frames that have been explicitly verified (`verified=True`)
        with independent manual provenance (`source='manual'`).
        """
        return [
            self.frames[k]
            for k in sorted(self.frames.keys())
            if self.frames[k].verified and self.frames[k].source == "manual"
        ]

