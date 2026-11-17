"""Offline tests for label-free, track-level team discovery."""

from __future__ import annotations

import numpy as np

from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    collect_track_appearances,
    evaluate_discovered_labels,
)
from src.config import TeamDiscoveryConfig
from src.schemas import InternalLabel
from tests.conftest import make_synthetic_kit_crop


def _observation(track_id: int, frame: int, crop: np.ndarray, *, grass: float = 0.05) -> TeamDiscoveryObservation:
    return TeamDiscoveryObservation(track_id, frame, frame / 25.0, crop, 45.0, grass, 0.05)


def test_collects_diverse_quality_gated_track_appearances() -> None:
    red = make_synthetic_kit_crop((20, 20, 220), seed=1)
    blue = make_synthetic_kit_crop((220, 50, 20), seed=2)
    observations = [
        *[_observation(10, f, red) for f in range(0, 80, 10)],
        *[_observation(20, f, blue) for f in range(0, 80, 10)],
        _observation(30, 1, red),  # too short a track
        _observation(40, 1, blue, grass=0.95),  # rejected quality
    ]
    tracks = collect_track_appearances(
        observations,
        TeamDiscoveryConfig(feature_method="spatial_embedding", min_track_observations=3, max_samples_per_track=4),
    )
    assert [item.tracking_id for item in tracks] == [10, 20]
    assert all(item.observation_count == 4 for item in tracks)
    assert all(np.isclose(np.linalg.norm(item.feature), 1.0, atol=1e-4) for item in tracks)


def test_auto_discovery_assigns_two_teams_and_rejects_outlier_group() -> None:
    observations = []
    for track_id in (1, 2, 3):
        crop = make_synthetic_kit_crop((20, 20, 220), (30, 30, 180), seed=track_id)
        observations.extend(_observation(track_id, frame, crop) for frame in (0, 10, 20, 30))
    for track_id in (11, 12, 13):
        crop = make_synthetic_kit_crop((220, 50, 20), (200, 200, 200), seed=track_id)
        observations.extend(_observation(track_id, frame, crop) for frame in (0, 10, 20, 30))
    # Yellow official-like appearance is deliberately not one of the two largest groups.
    for track_id in (21, 22):
        crop = make_synthetic_kit_crop((20, 220, 220), (20, 20, 20), seed=track_id)
        observations.extend(_observation(track_id, frame, crop) for frame in (0, 10, 20, 30))

    cfg = TeamDiscoveryConfig(candidate_clusters=3, min_cluster_tracks=2, min_track_observations=3)
    tracks = collect_track_appearances(observations, cfg)
    discoverer = AutomaticTeamDiscoverer(cfg).fit(tracks)
    labels = {track.tracking_id: discoverer.predict_track(track.tracking_id)[0] for track in tracks}

    assert len({labels[1], labels[11]}) == 2
    assert labels[1] in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
    assert labels[11] in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
    assert labels[21] == InternalLabel.UNKNOWN.value
    assert "Referee" not in labels.values()


def test_evaluation_aligns_arbitrary_cluster_ids_without_counting_unknown_as_correct() -> None:
    metrics = evaluate_discovered_labels(
        ["Team_B", "Team_B", "Team_A", "Team_A", "Unknown"],
        ["Team_A", "Team_A", "Team_B", "Team_B", "Team_A"],
    )
    assert metrics["permutation_applied"] is True
    assert metrics["classification_coverage"] == 0.8
    assert metrics["clustering_accuracy_on_covered"] == 1.0
    assert metrics["unknown_prediction_rate"] == 0.2
