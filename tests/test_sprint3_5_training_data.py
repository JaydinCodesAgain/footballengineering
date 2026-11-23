"""Regression tests for Sprint 3.5's human-review training-data gate."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.export_sprint3_5_yolo_dataset import export_yolo_dataset, summarize_gate
from scripts.prepare_sprint3_5_ball_review_batch import select_sparse_frame_numbers
from src.detection.annotation_workflow import AnnotatedBox, DetectionAnnotationStore, FrameAnnotation


def _make_review_source(root: Path, match_id: str, split: str) -> dict[str, object]:
    source_root = root / "data" / "annotations" / match_id
    store = DetectionAnnotationStore(source_root, match_id)
    store.frames_dir.mkdir(parents=True, exist_ok=True)
    image_path = store.frames_dir / "frame_000010.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    store.set_manual_frame_boxes(
        10,
        [AnnotatedBox("ball", [20, 30, 40, 50])],
        timestamp=0.4,
        image_rel_path="frames/frame_000010.jpg",
        verified=True,
    )
    store.save()
    return {
        "match_id": match_id,
        "split": split,
        "annotation_root": f"data/annotations/{match_id}",
        "review_frames": 1,
    }


def test_sparse_review_sampling_is_distinct_and_spans_video() -> None:
    selected = select_sparse_frame_numbers(10_000, 120)
    assert len(selected) == 120
    assert len(set(selected)) == 120
    assert selected[0] >= 400
    assert selected[-1] <= 9_600


def test_yolo_export_requires_manual_match_disjoint_review(tmp_path: Path) -> None:
    train = _make_review_source(tmp_path, "train_match", "train")
    validation = _make_review_source(tmp_path, "validation_match", "val")
    manifest = {"splits": [train, validation]}

    gate = summarize_gate(manifest, tmp_path, min_verified_frames=2, min_positive_ball_boxes=2)
    assert gate["training_eligible"] is True

    dataset_yaml = export_yolo_dataset(manifest, tmp_path, tmp_path / "dataset", gate)
    assert dataset_yaml.is_file()
    assert (tmp_path / "dataset" / "labels" / "train" / "train_match_frame_000010.txt").read_text().startswith("0 ")


def test_yolo_export_fails_closed_for_unreviewed_data(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "annotations" / "train_match"
    store = DetectionAnnotationStore(source_root, "train_match")
    store.frames[10] = store.frames.get(10) or FrameAnnotation(
        match_id="train_match", frame_number=10, timestamp=0.4, image_rel_path="frames/frame_000010.jpg"
    )
    store.save()
    manifest = {
        "splits": [
            {"match_id": "train_match", "split": "train", "annotation_root": "data/annotations/train_match", "review_frames": 1},
            {"match_id": "validation_match", "split": "val", "annotation_root": "data/annotations/missing", "review_frames": 1},
        ]
    }
    gate = summarize_gate(manifest, tmp_path, min_verified_frames=2, min_positive_ball_boxes=1)
    assert gate["training_eligible"] is False
    with pytest.raises(ValueError, match="Training export blocked"):
        export_yolo_dataset(manifest, tmp_path, tmp_path / "dataset", gate)


def test_model_proposals_are_preserved_but_never_become_ground_truth(tmp_path: Path) -> None:
    store = DetectionAnnotationStore(tmp_path / "review", "match")
    store.frames[3] = FrameAnnotation(
        match_id="match",
        frame_number=3,
        timestamp=0.12,
        image_rel_path="frames/frame_000003.jpg",
        model_proposals=[AnnotatedBox("ball", [10, 10, 20, 20])],
    )
    store.save()
    assert store.get_verified_ground_truth() == []
    # The manual-label CSV intentionally has no proposal-derived ball row.
    csv_text = store.csv_path.read_text(encoding="utf-8")
    assert "ball" not in csv_text

    reloaded = DetectionAnnotationStore(tmp_path / "review", "match")
    assert reloaded.frames[3].model_proposals[0].bbox == [10, 10, 20, 20]
    html = reloaded.generate_html_annotator().read_text(encoding="utf-8")
    assert "SUGGESTED" in html
