"""Fail-closed export of manually reviewed Sprint 3.5 annotations to YOLO format.

This is intentionally not an annotation tool and never creates labels from
predictions.  It imports the reviewer-exported CSV, accepts only
``verified=true, source=manual`` records, verifies match-disjoint source
splits, and refuses to create a training dataset until the review thresholds
are met.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.annotation_workflow import DetectionAnnotationStore, FrameAnnotation


def _resolve_project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def collect_manual_review_status(manifest: dict[str, Any], project_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load reviewed CSVs and return source-level counts plus gate violations."""
    reports: list[dict[str, Any]] = []
    violations: list[str] = []
    source_split_by_match: dict[str, str] = {}

    for source in manifest.get("splits", []):
        match_id = str(source["match_id"])
        split = str(source["split"])
        if match_id in source_split_by_match:
            violations.append(f"Duplicate match ID in manifest: {match_id}.")
            continue
        source_split_by_match[match_id] = split

        source_root = _resolve_project_path(project_root, str(source["annotation_root"]))
        csv_path = source_root / "detection_boxes.csv"
        if not csv_path.is_file():
            violations.append(f"{match_id}: reviewer CSV is missing: {csv_path}")
            continue

        store = DetectionAnnotationStore(source_root, match_id=match_id)
        # This accepts manual negative frames and preserves model proposals, but rejects
        # model/predicted provenance before it can be considered for training.
        store.import_boxes_from_csv(csv_path)
        annotations = [store.frames[key] for key in sorted(store.frames)]
        expected_frames = int(source["review_frames"])
        manual = [ann for ann in annotations if ann.verified and ann.source == "manual"]
        ball_boxes = sum(1 for ann in manual for box in ann.boxes if box.class_name == "ball")
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
                "manual_verified_frames": len(manual),
                "manual_verified_ball_boxes": ball_boxes,
                "annotations": manual,
            }
        )

    seen_splits = {item["split"] for item in reports}
    if "train" not in seen_splits or "val" not in seen_splits:
        violations.append("Both train and val source-match splits are required.")
    return reports, violations


def summarize_gate(
    manifest: dict[str, Any],
    project_root: Path,
    min_verified_frames: int,
    min_positive_ball_boxes: int,
) -> dict[str, Any]:
    reports, violations = collect_manual_review_status(manifest, project_root)
    verified_frames = sum(int(item["manual_verified_frames"]) for item in reports)
    ball_boxes = sum(int(item["manual_verified_ball_boxes"]) for item in reports)
    if verified_frames < min_verified_frames:
        violations.append(f"Only {verified_frames}/{min_verified_frames} required manually verified frames.")
    if ball_boxes < min_positive_ball_boxes:
        violations.append(f"Only {ball_boxes}/{min_positive_ball_boxes} required manually verified ball boxes.")

    return {
        "training_eligible": not violations,
        "thresholds": {
            "min_verified_frames": min_verified_frames,
            "min_positive_ball_boxes": min_positive_ball_boxes,
        },
        "totals": {
            "manual_verified_frames": verified_frames,
            "manual_verified_ball_boxes": ball_boxes,
        },
        "sources": [{key: value for key, value in item.items() if key != "annotations"} for item in reports],
        "blockers": violations,
    }


def _write_yolo_label(path: Path, ann: FrameAnnotation, width: int, height: int) -> None:
    lines: list[str] = []
    for box in ann.boxes:
        if box.class_name != "ball":
            continue
        x1, y1, x2, y2 = box.bbox
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise ValueError(f"Frame {ann.frame_number}: ball box is outside image bounds.")
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
            source_image = source_root / ann.image_rel_path
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing reviewed image: {source_image}")
            stem = f"{report['match_id']}_frame_{ann.frame_number:06d}"
            destination_image = image_dir / f"{stem}{source_image.suffix.lower()}"
            shutil.copy2(source_image, destination_image)

            import cv2  # kept local: this script can still run validation without OpenCV decoding

            image = cv2.imread(str(source_image))
            if image is None:
                raise ValueError(f"Could not read reviewed image: {source_image}")
            height, width = image.shape[:2]
            _write_yolo_label(label_dir / f"{stem}.txt", ann, width, height)

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
    )
    status["training_command_when_eligible"] = (
        "yolo detect train model=yolo11s.pt data=data/sprint3_5_ball_yolo/dataset.yaml "
        "imgsz=1280 epochs=100 patience=20 batch=2 device=cpu seed=42 deterministic=True"
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
