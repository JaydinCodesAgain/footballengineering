"""Tests for all jersey feature extractors, normalization, and dataset management."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.classification.dataset import (
    JerseyCropDataset,
    assign_temporal_and_track_splits,
    validate_no_split_leakage,
)
from src.classification.diagnostics import audit_dataset_labels
from src.classification.features import FEATURE_EXTRACTORS, get_feature_extractor
from src.config import MatchConfig, SplitConfig
from src.schemas import CropQualityMetrics, InternalLabel
from tests.conftest import make_synthetic_kit_crop


@pytest.mark.parametrize("method_name", list(FEATURE_EXTRACTORS.keys()))
def test_all_feature_extractors_produce_normalized_finite_vectors(method_name: str) -> None:
    crop = make_synthetic_kit_crop((20, 20, 220), (240, 240, 240), striped=True)
    extractor = get_feature_extractor(method_name)
    feat = extractor(crop)

    assert isinstance(feat, np.ndarray)
    assert feat.ndim == 1
    assert feat.dtype == np.float32
    assert len(feat) > 10
    assert np.all(np.isfinite(feat))

    if method_name == "baseline_hsv":
        assert pytest.approx(float(np.sum(feat)), rel=1e-3) == 1.0
    elif method_name != "legacy_enhanced":
        assert pytest.approx(float(np.linalg.norm(feat)), rel=1e-3) == 1.0


def test_feature_extractor_rejects_invalid_crop() -> None:
    extractor = get_feature_extractor("spatial_embedding")
    with pytest.raises(ValueError):
        extractor(np.zeros((0, 0, 3), dtype=np.uint8))


def test_dataset_persistence_splits_and_label_auditing(
    tmp_path: Path, synthetic_crops_by_class: dict[str, list[np.ndarray]]
) -> None:
    cfg = MatchConfig(
        match_id="test_match",
        match_name="Test Match",
        video_path="dummy.mp4",
        label_mapping={
            InternalLabel.TEAM_A.value: "Alpha",
            InternalLabel.TEAM_B.value: "Beta",
            InternalLabel.REFEREE.value: "REF",
            InternalLabel.UNKNOWN.value: "Unknown",
        },
    )
    ds = JerseyCropDataset(tmp_path / "crops", cfg)
    qm = CropQualityMetrics(48, 64, 50.0, 0.1, 0.0, False, True)

    # Add train and val crops with distinct timestamps and tracking IDs
    for i, img in enumerate(synthetic_crops_by_class["Team_A"][:6]):
        ds.add_crop(
            img,
            frame_number=10 + i * 10,
            timestamp=1.0 + i * 1.0,
            bbox=(10, 10, 58, 74),
            torso_bbox=(15, 20, 50, 60),
            quality=qm,
            tracking_id=100 + i,
            ground_truth_label="Alpha",
            label_verified=True,
        )
    for i, img in enumerate(synthetic_crops_by_class["Team_B"][:6]):
        ds.add_crop(
            img,
            frame_number=200 + i * 10,
            timestamp=25.0 + i * 1.0,
            bbox=(80, 10, 128, 74),
            torso_bbox=(85, 20, 120, 60),
            quality=qm,
            tracking_id=200 + i,
            ground_truth_label="Beta",
            label_verified=True,
        )
    # Add an unlabelled crop
    unlabelled_meta = ds.add_crop(
        synthetic_crops_by_class["Referee"][0],
        frame_number=350,
        timestamp=35.0,
        bbox=(150, 10, 198, 74),
        torso_bbox=(155, 20, 190, 60),
        quality=qm,
        tracking_id=301,
        ground_truth_label=None,
    )
    assert not unlabelled_meta.is_labelled

    # Assign temporal splits across a 40s video
    split_map = assign_temporal_and_track_splits(
        list(ds.records.values()),
        video_duration_seconds=40.0,
        split_config=SplitConfig(
            train_end_fraction=0.40,
            val_start_fraction=0.55,
            val_end_fraction=0.80,
            test_start_fraction=0.82,
        ),
    )
    for cid, sp in split_map.items():
        ds.records[cid].split = sp
    ds.save()

    # Reload from disk and verify leakage check passes
    ds_reloaded = JerseyCropDataset(tmp_path / "crops", cfg)
    assert len(ds_reloaded.records) == 13
    assert len(ds_reloaded.get_unlabelled_samples()) == 1

    leak_report = validate_no_split_leakage(list(ds_reloaded.records.values()))
    assert leak_report["is_clean"] is True

    # Inject a deliberate label error (assign a Team_B crop to Team_A) and verify audit_dataset_labels catches it
    wrong_crop = synthetic_crops_by_class["Team_B"][7]
    bad_meta = ds_reloaded.add_crop(
        wrong_crop,
        frame_number=95,
        timestamp=9.5,
        bbox=(200, 20, 248, 84),
        torso_bbox=(205, 30, 240, 70),
        quality=qm,
        tracking_id=999,
        ground_truth_label="Alpha",
        label_verified=True,
        split="train",
    )
    audit = audit_dataset_labels(ds_reloaded.get_labelled_samples(), feature_method="spatial_embedding")
    flagged_ids = {s["crop_id"] for s in audit["suspect_samples"]}
    assert bad_meta.crop_id in flagged_ids

