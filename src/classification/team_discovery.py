"""Label-free, per-video discovery of the two playing-team appearances.

The module deliberately treats team discovery as a track-level problem.  A
single player crop is often contaminated by grass, another player, a replay or
a broadcast graphic; a robust aggregate of several quality-gated observations
is substantially less brittle.  Cluster IDs are never presented as team names:
the two selected clusters are deterministically mapped to ``Team_A`` and
``Team_B`` only after fitting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from src.classification.features import get_feature_extractor
from src.config import TeamDiscoveryConfig
from src.schemas import ClassificationPrediction, InternalLabel


@dataclass(frozen=True)
class TeamDiscoveryObservation:
    """A quality-gated torso observation emitted during an initial video pass."""

    tracking_id: Optional[int]
    frame_number: int
    timestamp: float
    crop_bgr: np.ndarray
    blur_score: float
    grass_ratio: float
    occlusion_ratio: float


@dataclass
class TrackAppearance:
    tracking_id: int
    feature: np.ndarray
    observation_count: int
    frame_numbers: List[int]


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector.astype(np.float32) / max(norm, 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(_l2(a), _l2(b)), -1.0, 1.0))


def collect_track_appearances(
    observations: Sequence[TeamDiscoveryObservation],
    config: Optional[TeamDiscoveryConfig] = None,
) -> List[TrackAppearance]:
    """Build diverse, normalized track descriptors from reliable torso crops.

    Sampling is evenly spread over a track after quality filtering, so a long
    close-up or a repeated nearly-identical sequence cannot dominate fitting.
    Detections without a persistent tracker ID are intentionally excluded.
    """
    cfg = config or TeamDiscoveryConfig()
    extractor = get_feature_extractor(cfg.feature_method)
    by_track: Dict[int, List[TeamDiscoveryObservation]] = defaultdict(list)
    for item in observations:
        if item.tracking_id is None or item.crop_bgr is None or item.crop_bgr.size == 0:
            continue
        if item.grass_ratio > 0.80 or item.occlusion_ratio > 0.65 or item.blur_score <= 0:
            continue
        by_track[int(item.tracking_id)].append(item)

    appearances: List[TrackAppearance] = []
    for track_id, items in sorted(by_track.items()):
        ordered = sorted(items, key=lambda item: (item.frame_number, item.timestamp))
        if len(ordered) < max(1, cfg.min_track_observations):
            continue
        count = min(len(ordered), max(1, cfg.max_samples_per_track))
        indexes = np.linspace(0, len(ordered) - 1, count, dtype=int)
        selected = [ordered[int(index)] for index in sorted(set(indexes.tolist()))]
        features = [_l2(extractor(item.crop_bgr)) for item in selected]
        appearances.append(
            TrackAppearance(
                tracking_id=track_id,
                feature=_l2(np.median(np.vstack(features), axis=0)),
                observation_count=len(selected),
                frame_numbers=[item.frame_number for item in selected],
            )
        )
    return appearances


class AutomaticTeamDiscoverer:
    """Discover two dominant team appearances without labels or known kit colours."""

    def __init__(self, config: Optional[TeamDiscoveryConfig] = None) -> None:
        self.config = config or TeamDiscoveryConfig()
        self.cluster_centers_: Dict[int, np.ndarray] = {}
        self.cluster_radii_: Dict[int, float] = {}
        self.team_cluster_ids_: Dict[str, int] = {}
        self.outlier_cluster_ids_: List[int] = []
        self.track_labels_: Dict[int, str] = {}
        self.track_distances_: Dict[int, float] = {}
        self.is_fitted = False

    def _fit_clusters(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        method = self.config.clustering_method.lower()
        if method == "kmeans":
            return KMeans(n_clusters=n_clusters, n_init=20, random_state=self.config.random_state).fit_predict(features)
        if method == "gmm":
            return GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=self.config.random_state).fit_predict(features)
        if method == "agglomerative":
            return AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(features)
        if method == "dbscan":
            # DBSCAN can reject clutter but cannot guarantee two playing teams;
            # use its non-noise groups only when at least two are discovered.
            labels = DBSCAN(eps=0.18, min_samples=max(2, self.config.min_cluster_tracks), metric="cosine").fit_predict(features)
            if len(set(labels) - {-1}) >= 2:
                return labels
            return KMeans(n_clusters=n_clusters, n_init=20, random_state=self.config.random_state).fit_predict(features)
        raise ValueError("clustering_method must be one of: kmeans, gmm, agglomerative, dbscan.")

    def fit(self, appearances: Sequence[TrackAppearance]) -> "AutomaticTeamDiscoverer":
        if len(appearances) < 2:
            raise ValueError("Automatic team discovery needs at least two reliable player tracks.")
        features = np.vstack([item.feature for item in appearances]).astype(np.float32)
        n_clusters = min(max(2, self.config.candidate_clusters), len(appearances))
        cluster_ids = self._fit_clusters(features, n_clusters)
        members: Dict[int, List[int]] = defaultdict(list)
        for index, cluster_id in enumerate(cluster_ids.tolist()):
            if cluster_id >= 0:
                members[int(cluster_id)].append(index)
        ranked = sorted(members, key=lambda cluster_id: (-len(members[cluster_id]), cluster_id))
        eligible = [cluster_id for cluster_id in ranked if len(members[cluster_id]) >= self.config.min_cluster_tracks]
        selected = (eligible or ranked)[:2]
        if len(selected) < 2:
            raise ValueError("Could not identify two non-noise appearance groups from the supplied tracks.")

        # Mapping uses only centroid values, making Team_A/B stable for a fixed video
        # while never claiming that either identifier is a real club/team name.
        centers = {cluster_id: _l2(np.median(features[members[cluster_id]], axis=0)) for cluster_id in selected}
        ordered_teams = sorted(selected, key=lambda cluster_id: tuple(np.round(centers[cluster_id][:8], 5)))
        self.team_cluster_ids_ = {InternalLabel.TEAM_A.value: ordered_teams[0], InternalLabel.TEAM_B.value: ordered_teams[1]}
        self.cluster_centers_ = centers
        self.outlier_cluster_ids_ = sorted(set(members) - set(selected))

        for cluster_id, center in centers.items():
            distances = [_cosine_distance(features[index], center) for index in members[cluster_id]]
            # A small floor avoids rejecting a coherent sparse cluster due to a zero radius.
            self.cluster_radii_[cluster_id] = max(0.04, float(np.percentile(distances, 90)))

        for item in appearances:
            label, distance = self.predict_feature(item.feature)
            self.track_labels_[item.tracking_id] = label
            self.track_distances_[item.tracking_id] = distance
        self.is_fitted = True
        return self

    def predict_feature(self, feature: np.ndarray) -> Tuple[str, float]:
        if not self.cluster_centers_:
            raise RuntimeError("Fit AutomaticTeamDiscoverer before prediction.")
        normalized = _l2(feature)
        label, cluster_id = min(
            self.team_cluster_ids_.items(),
            key=lambda pair: _cosine_distance(normalized, self.cluster_centers_[pair[1]]),
        )
        distance = _cosine_distance(normalized, self.cluster_centers_[cluster_id])
        radius = self.cluster_radii_.get(cluster_id, 0.04)
        if distance > radius * max(1.0, self.config.assignment_distance_multiplier):
            return InternalLabel.UNKNOWN.value, distance
        return label, distance

    def predict_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return self.predict_feature(get_feature_extractor(self.config.feature_method)(crop_bgr))

    def predict_track(self, tracking_id: Optional[int], crop_bgr: Optional[np.ndarray] = None) -> Tuple[str, float]:
        if tracking_id is not None and tracking_id in self.track_labels_:
            return self.track_labels_[tracking_id], self.track_distances_[tracking_id]
        if crop_bgr is None or crop_bgr.size == 0:
            return InternalLabel.UNKNOWN.value, 1.0
        return self.predict_crop(crop_bgr)

    def predict_classification(self, crop_bgr: Optional[np.ndarray]) -> ClassificationPrediction:
        """Expose discovery through the pipeline's existing classifier contract."""
        if crop_bgr is None or crop_bgr.size == 0:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=InternalLabel.UNKNOWN.value,
                nearest_distance=1.0,
                second_nearest_distance=1.0,
                margin=0.0,
                confidence_score=0.0,
                is_unknown=True,
                rejection_reason="filtered_crop",
            )
        label, distance = self.predict_crop(crop_bgr)
        confidence = float(np.clip(1.0 - distance / 0.5, 0.0, 1.0))
        return ClassificationPrediction(
            predicted_label=label,
            raw_nearest_label=label,
            nearest_distance=distance,
            second_nearest_distance=distance,
            margin=0.0,
            confidence_score=confidence if label != InternalLabel.UNKNOWN.value else confidence * 0.5,
            is_unknown=label == InternalLabel.UNKNOWN.value,
            rejection_reason="outside_discovered_cluster_radius" if label == InternalLabel.UNKNOWN.value else None,
        )

    def report(self, appearances: Sequence[TrackAppearance]) -> Dict[str, object]:
        counts = Counter(self.track_labels_.values())
        return {
            "mode": "unsupervised_per_video_adaptation",
            "feature_method": self.config.feature_method,
            "clustering_method": self.config.clustering_method,
            "candidate_clusters": self.config.candidate_clusters,
            "reliable_tracks": len(appearances),
            "team_cluster_ids": self.team_cluster_ids_,
            "excluded_cluster_ids": self.outlier_cluster_ids_,
            "cluster_radii_cosine_distance": {str(k): round(v, 4) for k, v in self.cluster_radii_.items()},
            "track_label_counts": dict(counts),
            "team_name_note": "Team_A and Team_B are visual clusters, not inferred club or national-team names.",
        }


def evaluate_discovered_labels(
    predicted_labels: Sequence[str],
    reference_labels: Sequence[str],
) -> Dict[str, object]:
    """Evaluate labels after the best Team_A/Team_B permutation, never during fitting."""
    if len(predicted_labels) != len(reference_labels):
        raise ValueError("predicted_labels and reference_labels must have equal length.")
    from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score

    positive = {InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value}
    pairs = list(zip(predicted_labels, reference_labels))
    direct = sum(int(p == r) for p, r in pairs)
    swapped = sum(
        int((InternalLabel.TEAM_B.value if p == InternalLabel.TEAM_A.value else InternalLabel.TEAM_A.value if p == InternalLabel.TEAM_B.value else p) == r)
        for p, r in pairs
    )
    use_swap = swapped > direct
    aligned = [
        InternalLabel.TEAM_B.value if use_swap and p == InternalLabel.TEAM_A.value else InternalLabel.TEAM_A.value if use_swap and p == InternalLabel.TEAM_B.value else p
        for p in predicted_labels
    ]
    covered = [index for index, value in enumerate(aligned) if value in positive]
    covered_true = [reference_labels[index] for index in covered]
    covered_pred = [aligned[index] for index in covered]
    labels = [InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value]
    return {
        "metrics_available": bool(reference_labels),
        "permutation_applied": use_swap,
        "classification_coverage": round(len(covered) / len(reference_labels), 4) if reference_labels else 0.0,
        "unknown_prediction_rate": round(sum(p == InternalLabel.UNKNOWN.value for p in aligned) / len(aligned), 4) if aligned else 0.0,
        "clustering_accuracy_on_covered": round(sum(p == r for p, r in zip(covered_pred, covered_true)) / len(covered), 4) if covered else None,
        "macro_f1_on_covered": round(float(f1_score(covered_true, covered_pred, labels=labels, average="macro", zero_division=0)), 4) if covered else None,
        "adjusted_rand_index": round(float(adjusted_rand_score(reference_labels, aligned)), 4) if reference_labels else None,
        "normalized_mutual_information": round(float(normalized_mutual_info_score(reference_labels, aligned)), 4) if reference_labels else None,
        "labels": labels,
        "confusion_matrix": confusion_matrix(reference_labels, aligned, labels=labels + [InternalLabel.UNKNOWN.value]).tolist() if reference_labels else [],
    }
