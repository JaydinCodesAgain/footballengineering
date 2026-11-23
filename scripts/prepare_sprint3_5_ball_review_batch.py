"""Prepare, but never auto-approve, the Sprint 3.5 football-label review batch.

The script deliberately reuses :class:`DetectionAnnotationStore`.  It extracts
360 match-disjoint frames and saves low-threshold YOLO ball suggestions in a
separate field.  Suggestions are dashed overlays in the existing HTML reviewer
and cannot become ground truth unless a human redraws and marks the frame
verified.  The generated images and mutable annotation files are gitignored;
the compact manifest is the reproducible record of what was prepared.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DetectionConfig
from src.detection.annotation_workflow import AnnotatedBox, DetectionAnnotationStore
from src.detection.detector import FootballDetector


@dataclass(frozen=True)
class ReviewSource:
    match_id: str
    video_path: str
    split: str


REVIEW_SOURCES = (
    ReviewSource("chelsea_sunderland", "27.mp4", "train"),
    ReviewSource("norway_wales", "Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4", "train"),
    ReviewSource("man_utd_4_4_aston_villa_highlights", "data/Man Utd 4-4 Aston Villa _ Highlights.mp4", "val"),
)


def select_sparse_frame_numbers(total_frames: int, count: int) -> list[int]:
    """Choose deterministic, non-adjacent samples spanning usable match footage."""
    if total_frames <= 0:
        raise ValueError("Video reports no frames.")
    if count <= 0:
        raise ValueError("count must be positive.")
    if total_frames < count:
        raise ValueError(f"Requested {count} frames from a {total_frames}-frame video.")

    first = int(round((total_frames - 1) * 0.05))
    last = int(round((total_frames - 1) * 0.95))
    chosen = np.linspace(first, last, num=count, dtype=np.int64).tolist()
    selected = sorted(set(int(x) for x in chosen))
    if len(selected) != count:
        # linspace can duplicate only on very short sources; retain deterministic
        # full-range indices rather than silently returning too few review images.
        selected = np.linspace(0, total_frames - 1, num=count, dtype=np.int64).tolist()
    if len(set(selected)) != count:
        raise ValueError(f"Could not select {count} distinct frames from {total_frames} frames.")
    return [int(x) for x in selected]


def _load_ball_proposals(
    video_path: Path,
    frame_numbers: Iterable[int],
    detector: FootballDetector,
    confidence: float,
    max_per_frame: int,
) -> dict[int, list[AnnotatedBox]]:
    """Collect model *suggestions*, never labels, at a deliberately permissive threshold."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for proposals: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    proposals: dict[int, list[AnnotatedBox]] = {}
    try:
        for frame_number in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = cap.read()
            if not ok or frame is None:
                proposals[int(frame_number)] = []
                continue
            records = detector.detect_frame(
                frame,
                frame_number=int(frame_number),
                timestamp=float(frame_number) / fps,
                person_conf_override=0.99,
                ball_conf_override=confidence,
                imgsz_override=640,
            )
            balls = sorted(
                (record for record in records if record.class_name == "ball"),
                key=lambda record: record.confidence,
                reverse=True,
            )[:max_per_frame]
            proposals[int(frame_number)] = [
                AnnotatedBox(class_name="ball", bbox=[int(v) for v in ball.bbox]) for ball in balls
            ]
    finally:
        cap.release()
    return proposals


def prepare_review_batch(
    *,
    annotation_root: Path,
    manifest_path: Path,
    samples_per_match: int,
    with_proposals: bool,
    proposal_model: str,
    proposal_confidence: float,
    max_proposals_per_frame: int,
) -> dict[str, Any]:
    """Extract the review batch and write an honest, non-training manifest."""
    annotation_root = _resolve_project_path(annotation_root)
    manifest_path = _resolve_project_path(manifest_path)
    if samples_per_match < 1:
        raise ValueError("samples_per_match must be positive.")

    detector = (
        FootballDetector(DetectionConfig(model_name=proposal_model, imgsz=640)) if with_proposals else None
    )
    source_reports: list[dict[str, Any]] = []

    for source in REVIEW_SOURCES:
        video_path = ROOT / source.video_path
        if not video_path.is_file():
            raise FileNotFoundError(f"Required review source is missing: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        frames = select_sparse_frame_numbers(total_frames, samples_per_match)
        source_root = annotation_root / source.match_id
        store = DetectionAnnotationStore(source_root, match_id=source.match_id)
        exported = store.export_frames_for_annotation(video_path, frames)

        proposal_count = 0
        if detector is not None:
            proposal_map = _load_ball_proposals(
                video_path,
                frames,
                detector,
                proposal_confidence,
                max_proposals_per_frame,
            )
            for frame_number, proposals in proposal_map.items():
                store.frames[frame_number].model_proposals = proposals
                proposal_count += len(proposals)
            store.save()
            store.generate_html_annotator()

        source_reports.append(
            {
                **asdict(source),
                "video_path": source.video_path,
                "video_metadata": {"frame_count": total_frames, "fps": fps, "width": width, "height": height},
                "annotation_root": str(source_root.relative_to(ROOT)).replace("\\", "/"),
                "review_frames": len(exported),
                "manual_verified_frames": 0,
                "manual_verified_ball_boxes": 0,
                "review_status": "pending_human_review",
                "model_proposals": proposal_count,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "sprint3_5_first_football_specific_detector_training_review_batch",
        "review_status": "unreviewed_not_eligible_for_training",
        "sampling": {
            "method": "deterministic_uniform_5_to_95_percent_per_source_match",
            "samples_per_match": samples_per_match,
            "total_review_frames": sum(item["review_frames"] for item in source_reports),
            "adjacent_frame_leakage": "none_across_splits; sparse samples only within each source",
        },
        "label_policy": {
            "ground_truth": "manual boxes only; reviewer must inspect every frame including negatives",
            "model_proposals": "dashed UI suggestions only; excluded from CSV ground truth and training export",
            "soccertrack_reconstructed_ball_coordinates": "excluded; not observed bounding boxes",
        },
        "splits": source_reports,
        "final_test_split": {
            "status": "not_assigned",
            "reason": "No locally available, uninspected source has independently reviewed boxes. A future match must be reserved before model selection.",
        },
        "training_gate": {
            "eligible": False,
            "blockers": [
                "0 manually verified review frames",
                "0 manually verified football boxes",
                "no independently reviewed final test source",
            ],
        },
        "proposal_generation": {
            "enabled": with_proposals,
            "model": proposal_model if with_proposals else None,
            "imgsz": 640 if with_proposals else None,
            "confidence": proposal_confidence if with_proposals else None,
            "max_per_frame": max_proposals_per_frame if with_proposals else None,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _resolve_project_path(path: Path) -> Path:
    """Interpret CLI relative paths from the repository root, not this script's caller."""
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=ROOT / "data/annotations/sprint3_5_ball_review")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/sprint3_5/training_data_manifest.json")
    parser.add_argument("--samples-per-match", type=int, default=120)
    parser.add_argument("--without-proposals", action="store_true")
    parser.add_argument("--proposal-model", default="yolo11n.pt")
    parser.add_argument("--proposal-confidence", type=float, default=0.01)
    parser.add_argument("--max-proposals-per-frame", type=int, default=3)
    args = parser.parse_args()

    payload = prepare_review_batch(
        annotation_root=args.annotation_root,
        manifest_path=args.manifest,
        samples_per_match=args.samples_per_match,
        with_proposals=not args.without_proposals,
        proposal_model=args.proposal_model,
        proposal_confidence=args.proposal_confidence,
        max_proposals_per_frame=args.max_proposals_per_frame,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
