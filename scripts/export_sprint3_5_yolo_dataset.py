"""Fail-closed export of manually reviewed Sprint 3.5 annotations to YOLO format.

This is intentionally not an annotation tool and never creates labels from
predictions.  It imports the reviewer-exported CSV, accepts only
``verified=true, source=manual`` records, verifies match-disjoint source
splits, and refuses to create a training dataset until the review thresholds
are met.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.annotation_workflow import DetectionAnnotationStore, FrameAnnotation


SUPPORTED_SPLITS = {"train", "val"}


def _resolve_project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _review_image_path(source_root: Path, ann: FrameAnnotation) -> Path:
    relative = Path(ann.image_rel_path)
    if relative.is_absolute():
        raise ValueError("image_rel_path must be relative to the annotation source root")
    resolved_root = source_root.resolve()
    resolved_image = (source_root / relative).resolve()
    if not resolved_image.is_relative_to(resolved_root):
        raise ValueError("image_rel_path escapes the annotation source root")
    return resolved_image


def validate_image_space_box(match_id: str, frame_number: int, bbox: list[int], width: int, height: int) -> None:
    """Reject, rather than clip, invalid image-space ground-truth coordinates."""
    if len(bbox) != 4:
        raise ValueError(f"{match_id} frame {frame_number}: expected four box coordinates, got {bbox!r}.")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"{match_id} frame {frame_number}: invalid box {bbox}; expected "
            f"0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}."
        )


def _validate_manual_annotation(match_id: str, source_root: Path, ann: FrameAnnotation) -> tuple[Path, int, int]:
    if ann.match_id != match_id:
        raise ValueError(f"{match_id} frame {ann.frame_number}: annotation belongs to '{ann.match_id}'.")
    if ann.frame_number < 0 or not math.isfinite(ann.timestamp):
        raise ValueError(f"{match_id} frame {ann.frame_number}: invalid frame metadata.")
    image_path = _review_image_path(source_root, ann)
    if not image_path.is_file():
        raise FileNotFoundError(f"{match_id} frame {ann.frame_number}: reviewed source image is missing: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None or image.ndim != 3:
        raise ValueError(f"{match_id} frame {ann.frame_number}: image metadata is invalid: {image_path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"{match_id} frame {ann.frame_number}: image has invalid dimensions {width}x{height}.")
    for box in ann.boxes:
        box.validate()
        validate_image_space_box(match_id, ann.frame_number, box.bbox, width, height)
    return image_path, width, height


def collect_manual_review_status(manifest: dict[str, Any], project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load reviewed CSVs and return source-level counts plus gate violations."""
    reports: list[dict[str, Any]] = []
    violations: list[str] = []
    source_split_by_match: dict[str, str] = {}

    sources = manifest.get("splits", [])
    if not isinstance(sources, list) or not sources:
        return reports, ["Manifest must contain at least one review source in 'splits'."]

    for source in sources:
        if not isinstance(source, dict):
            violations.append("Manifest contains an invalid source record.")
            continue
        match_id = str(source.get("match_id", "")).strip()
        split = str(source.get("split", "")).strip().lower()
        if not match_id:
            violations.append("Manifest source is missing match_id.")
            continue
        if split not in SUPPORTED_SPLITS:
            violations.append(f"{match_id}: unsupported split '{split}'. Expected one of {sorted(SUPPORTED_SPLITS)}.")
            continue
        if match_id in source_split_by_match:
            violations.append(f"Duplicate match ID in manifest: {match_id}.")
            continue
        source_split_by_match[match_id] = split

        source_root_value = str(source.get("annotation_root", "")).strip()
        if not source_root_value:
            violations.append(f"{match_id}: annotation_root is missing.")
            continue
        source_root = _resolve_project_path(project_root, source_root_value)
        csv_path = source_root / "detection_boxes.csv"
        if not csv_path.is_file():
            violations.append(f"{match_id}: reviewer CSV is missing: {csv_path}")
            continue

        store = DetectionAnnotationStore(source_root, match_id=match_id)
        # This accepts manual negative frames and preserves model proposals, but rejects
        # model/predicted provenance before it can be considered for training.
        try:
            store.import_boxes_from_csv(csv_path, allow_destructive=False)
        except (ValueError, OSError, csv.Error) as exc:
            violations.append(f"{match_id}: reviewer CSV is invalid: {exc}")
            continue
        annotations = [store.frames[key] for key in sorted(store.frames)]
        try:
            expected_frames = int(source["review_frames"])
        except (KeyError, TypeError, ValueError):
            violations.append(f"{match_id}: review_frames must be a positive integer.")
            continue
        if expected_frames <= 0:
            violations.append(f"{match_id}: review_frames must be positive.")
            continue
        manual = [ann for ann in annotations if ann.verified and ann.source == "manual"]
        valid_manual: list[FrameAnnotation] = []
        for ann in manual:
            try:
                _validate_manual_annotation(match_id, source_root, ann)
            except (ValueError, OSError) as exc:
                violations.append(str(exc))
                continue
            valid_manual.append(ann)
        ball_boxes = sum(1 for ann in valid_manual for box in ann.boxes if box.class_name == "ball")
        if len(annotations) != expected_frames:
            violations.append(
                f"{match_id}: expected {expected_frames} review frames, found {len(annotations)} annotation records."
            )
        if len(manual) != expected_frames:
            violations.append(
                f"{match_id}: {len(manual)}/{expected_frames} frames have manual verification."
            )
        reports.append(
            {
                "match_id": match_id,
                "split": split,
                "annotation_root": str(source_root),
                "review_frames": expected_frames,
                "manual_verified_frames": len(valid_manual),
                "manual_verified_ball_boxes": ball_boxes,
                "annotations": valid_manual,
            }
        )

    seen_splits = {item["split"] for item in reports}
    if "train" not in seen_splits or "val" not in seen_splits:
        violations.append("Both train and val source-match splits are required.")

    frame_keys_by_split: dict[str, set[tuple[str, int]]] = {split: set() for split in SUPPORTED_SPLITS}
    for report in reports:
        for ann in report["annotations"]:
            key = (str(report["match_id"]), int(ann.frame_number))
            if key in frame_keys_by_split[report["split"]]:
                violations.append(f"Duplicate review frame within {report['split']}: {key[0]} frame {key[1]}.")
            frame_keys_by_split[report["split"]].add(key)
    return reports, violations


def summarize_gate(
    manifest: dict[str, Any],
    project_root: Path,
    min_verified_frames: int,
    min_positive_ball_boxes: int,
    min_positive_ball_boxes_train: int = 1,
    min_positive_ball_boxes_val: int = 1,
    required_split_matches: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    reports, violations = collect_manual_review_status(manifest, project_root)
    verified_frames = sum(int(item["manual_verified_frames"]) for item in reports)
    ball_boxes = sum(int(item["manual_verified_ball_boxes"]) for item in reports)
    split_positive_boxes = {
        split: sum(int(item["manual_verified_ball_boxes"]) for item in reports if item["split"] == split)
        for split in SUPPORTED_SPLITS
    }
    if required_split_matches is not None:
        for split, expected_matches in required_split_matches.items():
            if split not in SUPPORTED_SPLITS:
                violations.append(f"Configuration contains unsupported required split '{split}'.")
                continue
            actual_matches = {str(item["match_id"]) for item in reports if item["split"] == split}
            expected = {str(match_id) for match_id in expected_matches}
            missing = sorted(expected - actual_matches)
            unexpected = sorted(actual_matches - expected)
            if missing:
                violations.append(f"{split}: configured match sources are missing from manifest: {missing}.")
            if unexpected:
                violations.append(f"{split}: manifest has unconfigured match sources: {unexpected}.")
    if verified_frames < min_verified_frames:
        violations.append(f"Only {verified_frames}/{min_verified_frames} required manually verified frames.")
    if ball_boxes < min_positive_ball_boxes:
        violations.append(f"Only {ball_boxes}/{min_positive_ball_boxes} required manually verified ball boxes.")
    if split_positive_boxes["train"] < min_positive_ball_boxes_train:
        violations.append(
            f"Only {split_positive_boxes['train']}/{min_positive_ball_boxes_train} required manually verified ball boxes in train."
        )
    if split_positive_boxes["val"] < min_positive_ball_boxes_val:
        violations.append(
            f"Only {split_positive_boxes['val']}/{min_positive_ball_boxes_val} required manually verified ball boxes in val."
        )

    return {
        "training_eligible": not violations,
        "thresholds": {
            "min_verified_frames": min_verified_frames,
            "min_positive_ball_boxes": min_positive_ball_boxes,
            "min_positive_ball_boxes_train": min_positive_ball_boxes_train,
            "min_positive_ball_boxes_val": min_positive_ball_boxes_val,
        },
        "totals": {
            "manual_verified_frames": verified_frames,
            "manual_verified_ball_boxes": ball_boxes,
            "manual_verified_ball_boxes_by_split": split_positive_boxes,
        },
        "sources": [{key: value for key, value in item.items() if key != "annotations"} for item in reports],
        "blockers": violations,
    }


def _write_yolo_label(path: Path, match_id: str, ann: FrameAnnotation, width: int, height: int) -> None:
    lines: list[str] = []
    for box in ann.boxes:
        if box.class_name != "ball":
            continue
        x1, y1, x2, y2 = box.bbox
        validate_image_space_box(match_id, ann.frame_number, box.bbox, width, height)
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise ValueError(f"{match_id} frame {ann.frame_number}: normalized ball box is invalid after validation.")
        lines.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_dataset(
    manifest: dict[str, Any],
    project_root: Path,
    output_root: Path,
    gate: dict[str, Any],
) -> Path:
    """Copy only manually verified review frames and write a one-class YOLO dataset."""
    if not gate["training_eligible"]:
        raise ValueError("Training export blocked: " + " | ".join(gate["blockers"]))

    reports, violations = collect_manual_review_status(manifest, project_root)
    if violations:
        raise ValueError("Training export blocked: " + " | ".join(violations))
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output_root}")

    for report in reports:
        split = report["split"]
        source_root = Path(report["annotation_root"])
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for ann in report["annotations"]:
            source_image, width, height = _validate_manual_annotation(report["match_id"], source_root, ann)
            stem = f"{report['match_id']}_frame_{ann.frame_number:06d}"
            destination_image = image_dir / f"{stem}{source_image.suffix.lower()}"
            shutil.copy2(source_image, destination_image)

            _write_yolo_label(label_dir / f"{stem}.txt", report["match_id"], ann, width, height)

    dataset_yaml = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": ["ball"],
    }
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/sprint3_5/training_data_manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sprint3_5_ball_training.yaml")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/sprint3_5_ball_yolo")
    parser.add_argument("--export", action="store_true", help="Write the YOLO dataset only after every gate passes.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    review_gate = config["review_gate"]
    status = summarize_gate(
        manifest,
        args.project_root,
        int(review_gate["min_verified_frames"]),
        int(review_gate["min_positive_ball_boxes"]),
        int(review_gate.get("min_positive_ball_boxes_train", 1)),
        int(review_gate.get("min_positive_ball_boxes_val", 1)),
        review_gate.get("required_splits"),
    )
    status["training_command_when_eligible"] = (
        "python scripts/train_sprint3_5_yolo.py --execute"
    )
    status_path = ROOT / "outputs/sprint3_5/training_gate_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))

    if args.export:
        dataset_yaml = export_yolo_dataset(manifest, args.project_root, args.output_root, status)
        print(f"Exported reviewed YOLO dataset: {dataset_yaml}")


if __name__ == "__main__":
    main()
