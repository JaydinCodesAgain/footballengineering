"""Tests for TeamClassifier, Unknown distance rejection, calibration, and serialization."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.classification.classifier import TeamClassifier, UnsupervisedKMeansTeamClusterer
from src.config import ClassificationConfig
from src.schemas import InternalLabel


@pytest.mark.parametrize("classifier_type", ["knn", "svm_linear", "svm_rbf", "logistic", "nearest_centroid"])
def test_team_classifier_training_and_prediction(
    classifier_type: str,
    synthetic_crops_by_class: dict[str, list[np.ndarray]],
) -> None:
    train_crops = (
        synthetic_crops_by_class["Team_A"][:6]
        + synthetic_crops_by_class["Team_B"][:4]
        + synthetic_crops_by_class["Referee"][:4]
    )
    train_labels = (
        [InternalLabel.TEAM_A.value] * 6
        + [InternalLabel.TEAM_B.value] * 4
        + [InternalLabel.REFEREE.value] * 4
    )

    cfg = ClassificationConfig(
        feature_method="spatial_embedding",
        classifier_type=classifier_type,
        n_neighbors=3,
        unknown_threshold=0.55,
        auto_calibrate_threshold=False,
        min_margin=0.01,
        balance_class_weights=True,
    )
    clf = TeamClassifier(match_id="match_1", config=cfg)
    clf.fit(train_crops, train_labels)

    pred_a = clf.predict_crop(synthetic_crops_by_class["Team_A"][7], match_id="match_1")
    pred_b = clf.predict_crop(synthetic_crops_by_class["Team_B"][7], match_id="match_1")
    pred_ref = clf.predict_crop(synthetic_crops_by_class["Referee"][6], match_id="match_1")

    assert pred_a.predicted_label == InternalLabel.TEAM_A.value
    assert pred_b.predicted_label == InternalLabel.TEAM_B.value
    assert pred_ref.predicted_label == InternalLabel.REFEREE.value
    assert not pred_a.is_unknown


def test_unknown_rejection_calibration_and_cross_match_guard(
    tmp_path: Path,
    synthetic_crops_by_class: dict[str, list[np.ndarray]],
) -> None:
    train_crops = (
        synthetic_crops_by_class["Team_A"][:6]
        + synthetic_crops_by_class["Team_B"][:6]
        + synthetic_crops_by_class["Referee"][:5]
    )
    train_labels = (
        [InternalLabel.TEAM_A.value] * 6
        + [InternalLabel.TEAM_B.value] * 6
        + [InternalLabel.REFEREE.value] * 5
    )

    val_crops = (
        synthetic_crops_by_class["Team_A"][6:]
        + synthetic_crops_by_class["Team_B"][6:]
        + synthetic_crops_by_class["Referee"][5:]
        + synthetic_crops_by_class["Unknown"]
    )
    val_labels = (
        [InternalLabel.TEAM_A.value] * 4
        + [InternalLabel.TEAM_B.value] * 4
        + [InternalLabel.REFEREE.value] * 3
        + [InternalLabel.UNKNOWN.value] * 4
    )

    cfg = ClassificationConfig(
        feature_method="spatial_embedding",
        classifier_type="knn",
        n_neighbors=3,
        unknown_threshold=0.35,
        auto_calibrate_threshold=True,
    )
    clf = TeamClassifier(match_id="norway_wales", config=cfg)
    clf.fit(train_crops, train_labels)

    cal_info = clf.calibrate_unknown_threshold(val_crops, val_labels)
    assert "selected_threshold" in cal_info
    assert len(cal_info["curve"]) > 5

    # Out-of-distribution Unknown crop should be rejected when threshold is tight
    clf.unknown_threshold = float(cal_info["known_distance_summary"]["max"]) + 0.01
    unk_pred = clf.predict_crop(synthetic_crops_by_class["Unknown"][0], match_id="norway_wales")
    assert unk_pred.predicted_label == InternalLabel.UNKNOWN.value
    assert unk_pred.is_unknown

    # Cross-match guard must block accidental evaluation on another match
    with pytest.raises(ValueError, match="Cross-match safety check failed"):
        clf.predict_crop(synthetic_crops_by_class["Team_A"][0], match_id="chelsea_sunderland")

    # Serialization round-trip
    ckpt = clf.save(tmp_path / "model.joblib")
    loaded = TeamClassifier.load(ckpt)
    assert loaded.match_id == "norway_wales"
    assert loaded.is_fitted
    p_loaded = loaded.predict_crop(synthetic_crops_by_class["Team_A"][8], match_id="norway_wales")
    assert p_loaded.predicted_label == InternalLabel.TEAM_A.value


def test_unsupervised_kmeans_clusterer(
    synthetic_crops_by_class: dict[str, list[np.ndarray]],
) -> None:
    crops = (
        synthetic_crops_by_class["Team_A"][:5]
        + synthetic_crops_by_class["Team_B"][:5]
        + synthetic_crops_by_class["Referee"][:5]
    )
    clusterer = UnsupervisedKMeansTeamClusterer(n_clusters=3, random_state=42)
    clusterer.fit(crops)
    pred = clusterer.predict(synthetic_crops_by_class["Team_A"][0])
    assert pred in ("Cluster_0", "Cluster_1", "Cluster_2")
