"""Regression tests for Sprint 3.5's reviewed football-training data boundary."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.export_sprint3_5_yolo_dataset import (
    export_yolo_dataset,
    summarize_gate,
    validate_image_space_box,
)
from scripts.prepare_sprint3_5_ball_review_batch import (
    ReviewSource,
    prepare_review_batch,
    select_sparse_frame_numbers,
)
from scripts.train_sprint3_5_yolo import build_train_kwargs
from src.detection.annotation_workflow import AnnotatedBox, DetectionAnnotationStore, FrameAnnotation


def _make_review_source(
    root: Path,
    match_id: str,
    split: str,
    *,
    ball_box: list[int] | None = None,
) -> dict[str, object]:
    source_root = root / "data" / "annotations" / match_id
    store = DetectionAnnotationStore(source_root, match_id)
    image_path = store.frames_dir / "frame_000010.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    store.set_manual_frame_boxes(
        10,
        [] if ball_box is None else [AnnotatedBox("ball", ball_box)],
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


def _write_tiny_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index * 25, dtype=np.uint8))
    writer.release()


def test_sparse_review_sampling_is_distinct_and_spans_video() -> None:
    selected = select_sparse_frame_numbers(10_000, 120)
    assert len(selected) == 120
    assert len(set(selected)) == 120
    assert selected[0] >= 400
    assert selected[-1] <= 9_600


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ([-1, 1, 3, 4], "expected"),
        ([1, 1, 201, 4], "expected"),
        ([1, 1, 1, 4], "expected"),
        ([5, 5, 2, 7], "expected"),
    ],
)
def test_image_space_box_validation_rejects_invalid_coordinates(bbox: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_image_space_box("match", 10, bbox, width=200, height=100)


def test_image_space_box_validation_accepts_small_valid_football_box() -> None:
    validate_image_space_box("match", 10, [199, 99, 200, 100], width=200, height=100)


def test_yolo_export_requires_manual_match_disjoint_review(tmp_path: Path) -> None:
    train = _make_review_source(tmp_path, "train_match", "train", ball_box=[20, 30, 21, 31])
    validation = _make_review_source(tmp_path, "validation_match", "val", ball_box=[20, 30, 40, 50])
    manifest = {"splits": [train, validation]}

    gate = summarize_gate(
        manifest,
        tmp_path,
        min_verified_frames=2,
        min_positive_ball_boxes=2,
        min_positive_ball_boxes_train=1,
        min_positive_ball_boxes_val=1,
    )
    assert gate["training_eligible"] is True

    dataset_yaml = export_yolo_dataset(manifest, tmp_path, tmp_path / "dataset", gate)
    assert dataset_yaml.is_file()
    assert (tmp_path / "dataset" / "labels" / "train" / "train_match_frame_000010.txt").read_text().startswith("0 ")


def test_gate_requires_positive_ball_boxes_in_both_train_and_validation(tmp_path: Path) -> None:
    train = _make_review_source(tmp_path, "train_match", "train", ball_box=[20, 30, 40, 50])
    validation = _make_review_source(tmp_path, "validation_match", "val", ball_box=None)
    gate = summarize_gate(
        {"splits": [train, validation]},
        tmp_path,
        min_verified_frames=2,
        min_positive_ball_boxes=1,
        min_positive_ball_boxes_train=1,
        min_positive_ball_boxes_val=1,
    )
    assert gate["training_eligible"] is False
    assert any("in val" in blocker for blocker in gate["blockers"])


def test_gate_rejects_unsupported_split_and_missing_image(tmp_path: Path) -> None:
    bad_split = _make_review_source(tmp_path, "bad_split", "test", ball_box=[20, 30, 40, 50])
    gate = summarize_gate({"splits": [bad_split]}, tmp_path, 1, 1)
    assert gate["training_eligible"] is False
    assert any("unsupported split" in blocker for blocker in gate["blockers"])

    missing = _make_review_source(tmp_path, "missing_image", "train", ball_box=[20, 30, 40, 50])
    image_path = tmp_path / "data" / "annotations" / "missing_image" / "frames" / "frame_000010.jpg"
    image_path.unlink()
    validation = _make_review_source(tmp_path, "validation", "val", ball_box=[20, 30, 40, 50])
    gate = summarize_gate({"splits": [missing, validation]}, tmp_path, 2, 2)
    assert gate["training_eligible"] is False
    assert any("source image is missing" in blocker for blocker in gate["blockers"])


def test_gate_rejects_out_of_bounds_manual_box_and_duplicate_source(tmp_path: Path) -> None:
    invalid = _make_review_source(tmp_path, "invalid", "train", ball_box=[20, 30, 40, 50])
    source_root = tmp_path / "data" / "annotations" / "invalid"
    store = DetectionAnnotationStore(source_root, "invalid")
    store.frames[10].boxes = [AnnotatedBox("ball", [-1, 30, 40, 50])]
    store.save()
    validation = _make_review_source(tmp_path, "validation", "val", ball_box=[20, 30, 40, 50])
    gate = summarize_gate({"splits": [invalid, validation]}, tmp_path, 2, 2)
    assert gate["training_eligible"] is False
    assert any("expected 0 <= x1" in blocker for blocker in gate["blockers"])

    duplicate_gate = summarize_gate({"splits": [validation, validation]}, tmp_path, 2, 2)
    assert duplicate_gate["training_eligible"] is False
    assert any("Duplicate match ID" in blocker for blocker in duplicate_gate["blockers"])


def test_yolo_export_fails_closed_for_unreviewed_data(tmp_path: Path) -> None:
    source_root = tmp_path / "data" / "annotations" / "train_match"
    store = DetectionAnnotationStore(source_root, "train_match")
    store.frames[10] = FrameAnnotation(
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


def test_rerunning_preparation_preserves_json_and_csv_only_manual_review(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path)
    source = ReviewSource("tiny", str(video_path), "train")
    root = tmp_path / "review"
    manifest_path = tmp_path / "manifest.json"
    kwargs = {
        "annotation_root": root,
        "manifest_path": manifest_path,
        "samples_per_match": 2,
        "with_proposals": False,
        "proposal_model": "yolo11n.pt",
        "proposal_confidence": 0.01,
        "max_proposals_per_frame": 1,
        "sources": [source],
    }
    prepare_review_batch(**kwargs)

    store = DetectionAnnotationStore(root / "tiny", "tiny")
    reviewed_frame = sorted(store.frames)[0]
    # Simulate a browser-exported CSV replacing the file while JSON remains
    # unreviewed. Re-preparation must import rather than overwrite this work.
    with store.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["frame_number"]) == reviewed_frame:
            row.update({"class_name": "ball", "x1": "2", "y1": "2", "x2": "4", "y2": "4", "verified": "true", "source": "manual"})
    with store.csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    prepare_review_batch(**kwargs)
    resumed = DetectionAnnotationStore(root / "tiny", "tiny")
    assert resumed.frames[reviewed_frame].verified is True
    assert resumed.frames[reviewed_frame].boxes[0].bbox == [2, 2, 4, 4]


def test_preparation_rejects_changed_frame_selection_without_backup(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, frame_count=6)
    source = ReviewSource("tiny", str(video_path), "train")
    root = tmp_path / "review"
    common = {
        "annotation_root": root,
        "manifest_path": tmp_path / "manifest.json",
        "with_proposals": False,
        "proposal_model": "yolo11n.pt",
        "proposal_confidence": 0.01,
        "max_proposals_per_frame": 1,
        "sources": [source],
    }
    prepare_review_batch(samples_per_match=2, **common)
    store = DetectionAnnotationStore(root / "tiny", "tiny")
    reviewed_frame = sorted(store.frames)[0]
    store.set_manual_frame_boxes(reviewed_frame, [AnnotatedBox("ball", [1, 1, 3, 3])], verified=True)
    store.save()

    with pytest.raises(ValueError, match="Refusing to alter a review batch"):
        prepare_review_batch(samples_per_match=3, **common)
    preserved = DetectionAnnotationStore(root / "tiny", "tiny")
    assert preserved.frames[reviewed_frame].verified is True


def test_model_proposals_are_preserved_but_never_become_ground_truth_and_ui_supports_resume(tmp_path: Path) -> None:
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
    assert "ball" not in store.csv_path.read_text(encoding="utf-8")

    reloaded = DetectionAnnotationStore(tmp_path / "review", "match")
    assert reloaded.frames[3].model_proposals[0].bbox == [10, 10, 20, 20]
    html = reloaded.generate_html_annotator().read_text(encoding="utf-8")
    assert "SUGGESTED" in html
    assert "Zoom +" in html
    assert "pointerToImageCoordinates" in html
    assert "Save Browser Progress" in html
    assert "Import CSV" in html


def test_training_configuration_has_executable_ball_only_contract() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "sprint3_5_ball_training.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["base_model"] == "yolo11s.pt"
    assert config["class_names"] == ["ball"]
    assert config["training"]["imgsz"] == 1280
    assert config["training"]["batch"] == 2
    assert config["training"]["device"] == "cpu"
    assert config["review_gate"]["min_positive_ball_boxes_train"] == 80
    assert config["review_gate"]["min_positive_ball_boxes_val"] == 20
    kwargs = build_train_kwargs(config, Path("data/sprint3_5_ball_yolo/dataset.yaml"))
    assert kwargs["imgsz"] == 1280
    assert kwargs["batch"] == 2
    assert kwargs["device"] == "cpu"
    assert kwargs["data"].endswith("data\\sprint3_5_ball_yolo\\dataset.yaml")
