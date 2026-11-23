"""Label-free, per-video discovery of the two playing-team appearances.

The module deliberately treats team discovery as a track-level problem.  A
single player crop is often contaminated by grass, another player, a replay or
a broadcast graphic; a robust aggregate of several quality-gated observations
is substantially less brittle.  Cluster IDs are never presented as team names:
the two selected clusters are deterministically mapped to ``Team_A`` and
``Team_B`` only after fitting.

Key reliability safeguards (Sprint 3.1):
- Shot-aware track segmentation (``shot_id``) prevents cross-cut ID reuse from
  blending two different players into one track appearance.
- Tactical frame and crowd-box filtering prevents non-tactical player close-ups,
  managers, and spectators from polluting team clusters.
- Outlier-cluster repulsion retains coherent excluded cluster centroids (e.g.,
  referees and goalkeepers) so samples closest to an official/GK cluster are
  explicitly rejected as ``Unknown`` instead of forced onto the nearer team.
- Robust MAD/IQR cluster radii bounded by inter-team separation plus explicit
  inter-team margin and distance-ratio gates handle both distinct and similar
  kit colours conservatively.
- ``TrackAppearanceBuffer`` enables track-level feature aggregation during live
  or second-pass inference without blindly assuming track-ID correspondence
  across independent tracker runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from src.classification.features import get_feature_extractor
from src.config import TeamDiscoveryConfig
from src.schemas import ClassificationPrediction, InternalLabel

try:
    from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN
except ImportError:  # pragma: no cover
    _SklearnHDBSCAN = None


@dataclass(frozen=True)
class TeamDiscoveryObservation:
    """A quality-gated torso observation emitted during video analysis."""

    tracking_id: Optional[int]
    frame_number: int
    timestamp: float
    crop_bgr: np.ndarray
    blur_score: float
    grass_ratio: float
    occlusion_ratio: float
    shot_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    feature: Optional[np.ndarray] = None


@dataclass
class TrackAppearance:
    """Aggregated appearance descriptor for a single player track (within one shot)."""

    tracking_id: int
    feature: np.ndarray
    observation_count: int
    frame_numbers: List[int]
    shot_id: int = 0
    mean_blur: float = 0.0
    mean_grass_ratio: float = 0.0

    @property
    def segment_key(self) -> Tuple[int, int]:
        return (int(self.shot_id), int(self.tracking_id))


def _l2(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(_l2(a), _l2(b)), -1.0, 1.0))


def is_tactical_discovery_frame(
    player_bboxes: Sequence[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int],
) -> bool:
    """Reject full-screen player close-ups and celebration shots during team discovery."""
    if not player_bboxes:
        return False
    fh, fw = frame_shape[:2]
    if fh <= 0 or fw <= 0:
        return True
    frame_area = float(fh * fw)
    heights = [float(max(1, b[3] - b[1])) for b in player_bboxes]
    max_area = max(float(max(1, b[2] - b[0]) * max(1, b[3] - b[1])) for b in player_bboxes)
    med_h = float(np.median(heights))
    if med_h > 0.44 * fh or (max_area / frame_area) > 0.18:
        return False
    return True


def filter_discovery_player_indices(
    player_bboxes: Sequence[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, int],
) -> List[int]:
    """Return indices of pitch-plausible player boxes, rejecting crowd/foreground outliers."""
    if not player_bboxes:
        return []
    fh, fw = frame_shape[:2]
    if fh <= 0 or fw <= 0:
        return list(range(len(player_bboxes)))

    heights = [float(max(1, b[3] - b[1])) for b in player_bboxes]
    med_h = float(np.median(heights))
    kept: List[int] = []
    for idx, (x1, y1, x2, y2) in enumerate(player_bboxes):
        h = float(max(1, y2 - y1))
        w = float(max(1, x2 - x1))
        # Reject tiny spectators wholly inside the upper stadium stands band
        if y2 < 0.14 * fh:
            continue
        # Reject extreme foreground close-up boxes or tiny distant crowd boxes
        if len(player_bboxes) >= 4 and med_h > 20.0:
            if h > 2.4 * med_h or h < 0.38 * med_h:
                continue
        # Reject implausibly wide non-player boxes (e.g. banners or merged crowd blobs)
        if w / h > 1.85:
            continue
        kept.append(idx)
    return kept


def _compute_observation_quality_weight(
    obs: TeamDiscoveryObservation,
) -> float:
    """Compute observation reliability weight favoring crisp, larger, low-glare torso crops.

    On panoramic fixed-camera night footage (e.g. 3840x1906 floodlights), distant 16-24 px
    players suffer far more specular glare and compression artifacts than 40-70 px midfield
    players. Weighting by bbox scale, turf-free area, low occlusion, and sharpness prevents
    noisy far-touchline crops from corrupting a track's aggregated appearance descriptor.
    """
    base_clean = max(0.05, (1.0 - float(obs.grass_ratio)) * (1.0 - float(obs.occlusion_ratio)))
    blur_term = math.log1p(max(0.0, float(obs.blur_score))) / 4.5
    blur_term = max(0.25, min(1.5, blur_term))

    scale_term = 1.0
    if obs.bbox is not None and len(obs.bbox) == 4:
        bh = max(1.0, float(obs.bbox[3] - obs.bbox[1]))
        scale_term = min(1.35, max(0.20, (bh / 42.0) ** 1.35))

    glare_term = 1.0
    if obs.crop_bgr is not None and obs.crop_bgr.size > 0:
        # Down-weight crops overwhelmed by extreme blown-out floodlight glare (V>235, S<30)
        import cv2

        hsv = cv2.cvtColor(obs.crop_bgr, cv2.COLOR_BGR2HSV)
        extreme_glare = float(np.mean((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 30)))
        if 0.15 < extreme_glare < 0.65:
            glare_term = max(0.45, 1.0 - 0.8 * extreme_glare)

    return float(max(0.05, base_clean * blur_term * scale_term * glare_term))


def _weighted_trimmed_aggregate(
    features: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Robust quality-weighted descriptor aggregation with cosine-outlier down-weighting."""
    mat = np.vstack([_l2(f) for f in features]).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.05)
    if len(mat) == 1:
        return _l2(mat[0])
    if len(mat) == 2:
        return _l2(np.average(mat, axis=0, weights=w))

    # First-pass coordinate-wise median anchor, then quality-weighted mean down-weighting outliers
    anchor = _l2(np.median(mat, axis=0))
    sims = np.clip(mat @ anchor, -1.0, 1.0)
    dists = 1.0 - sims
    med_d = float(np.median(dists))
    inlier_w = np.exp(-np.maximum(0.0, dists - med_d) / max(0.08, med_d + 0.05))
    combined_w = w * inlier_w.astype(np.float32)
    if float(np.sum(combined_w)) <= 1e-8:
        return anchor
    return _l2(np.average(mat, axis=0, weights=combined_w))


def collect_track_appearances(
    observations: Sequence[TeamDiscoveryObservation],
    config: Optional[TeamDiscoveryConfig] = None,
) -> List[TrackAppearance]:
    """Build diverse, normalized track descriptors from reliable torso crops.

    Sampling is evenly spread over a track after quality filtering, so a long
    close-up or a repeated nearly-identical sequence cannot dominate fitting.
    Observations are grouped by ``(shot_id, tracking_id)`` so identity resets
    across broadcast cuts never contaminate a track descriptor.
    """
    cfg = config or TeamDiscoveryConfig()
    extractor = get_feature_extractor(cfg.feature_method)
    by_segment: Dict[Tuple[int, int], List[TeamDiscoveryObservation]] = defaultdict(list)
    for item in observations:
        if item.tracking_id is None:
            continue
        has_feat = item.feature is not None and item.feature.size > 0
        has_crop = item.crop_bgr is not None and item.crop_bgr.size > 0
        if not has_feat and not has_crop:
            continue
        if item.grass_ratio > 0.80 or item.occlusion_ratio > 0.65 or item.blur_score <= 0:
            continue
        by_segment[(int(item.shot_id), int(item.tracking_id))].append(item)

    appearances: List[TrackAppearance] = []
    for (shot_id, track_id), items in sorted(by_segment.items()):
        ordered = sorted(items, key=lambda item: (item.frame_number, item.timestamp))
        if len(ordered) < max(1, cfg.min_track_observations):
            continue
        count = min(len(ordered), max(1, cfg.max_samples_per_track))
        # Partition into `count` temporal strata and pick the highest-quality crop per stratum
        strata = np.array_split(np.arange(len(ordered)), count)
        selected: List[TeamDiscoveryObservation] = []
        selected_weights: List[float] = []
        for stratum in strata:
            if len(stratum) == 0:
                continue
            scored = [(int(i), _compute_observation_quality_weight(ordered[int(i)])) for i in stratum]
            best_idx, best_w = max(scored, key=lambda pair: pair[1])
            selected.append(ordered[best_idx])
            selected_weights.append(best_w)

        features = [
            _l2(item.feature) if (item.feature is not None and item.feature.size > 0) else _l2(extractor(item.crop_bgr))
            for item in selected
        ]
        use_weighted = bool(getattr(cfg, "quality_weighted_aggregation", True))
        agg_feat = (
            _weighted_trimmed_aggregate(features, selected_weights)
            if use_weighted
            else _l2(np.median(np.vstack(features), axis=0))
        )
        appearances.append(
            TrackAppearance(
                tracking_id=track_id,
                shot_id=shot_id,
                feature=agg_feat,
                observation_count=len(selected),
                frame_numbers=[item.frame_number for item in selected],
                mean_blur=round(float(np.mean([item.blur_score for item in selected])), 2),
                mean_grass_ratio=round(float(np.mean([item.grass_ratio for item in selected])), 4),
            )
        )
    return appearances


class AutomaticTeamDiscoverer:
    """Discover two dominant team appearances without labels or known kit colours."""

    def __init__(self, config: Optional[TeamDiscoveryConfig] = None) -> None:
        self.config = config or TeamDiscoveryConfig()
        self.cluster_centers_: Dict[int, np.ndarray] = {}
        self.outlier_centers_: Dict[int, np.ndarray] = {}
        self.cluster_radii_: Dict[int, float] = {}
        self.cluster_spreads_: Dict[int, float] = {}
        self.team_cluster_ids_: Dict[str, int] = {}
        self.outlier_cluster_ids_: List[int] = []
        self.track_labels_: Dict[int, str] = {}
        self.track_distances_: Dict[int, float] = {}
        self.segment_labels_: Dict[Tuple[int, int], str] = {}
        self.segment_predictions_: Dict[Tuple[int, int], ClassificationPrediction] = {}
        self.inter_team_distance_: float = 0.0
        self.selected_k_: int = 0
        self.silhouette_score_: Optional[float] = None
        self.is_fitted = False

    def _cluster_once(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        method = self.config.clustering_method.lower()
        n_samples, dim = features.shape
        k_eff = min(max(2, int(n_clusters)), n_samples)

        if method == "kmeans":
            return KMeans(
                n_clusters=k_eff, n_init=30, random_state=self.config.random_state
            ).fit_predict(features)

        if method == "agglomerative":
            # On L2-normalized features, Ward linkage minimizes spherical cosine variance
            # and avoids the singleton-outlier peeling failure of average linkage.
            return AgglomerativeClustering(n_clusters=k_eff, linkage="ward").fit_predict(features)

        if method == "gmm":
            n_comp_pca = max(2, min(16, n_samples - 1, dim))
            if n_samples > 2 and dim > n_comp_pca:
                projected = PCA(n_components=n_comp_pca, random_state=self.config.random_state).fit_transform(
                    features
                )
            else:
                projected = features
            return GaussianMixture(
                n_components=k_eff,
                covariance_type="full",
                reg_covar=1e-3,
                n_init=5,
                random_state=self.config.random_state,
            ).fit_predict(projected)

        if method == "hdbscan":
            if _SklearnHDBSCAN is not None and n_samples >= 6:
                min_cs = max(3, int(round(0.06 * n_samples)))
                labels = _SklearnHDBSCAN(
                    min_cluster_size=min_cs,
                    min_samples=max(2, min_cs // 2),
                    metric="cosine",
                    copy=True,
                ).fit_predict(features)
                if len(set(labels.tolist()) - {-1}) >= 2:
                    return labels
            return AgglomerativeClustering(n_clusters=k_eff, linkage="ward").fit_predict(features)

        if method == "dbscan":
            sim = np.clip(features @ features.T, -1.0, 1.0)
            dmat = np.sort(1.0 - sim, axis=1)
            k_nn = min(5, max(1, n_samples - 1))
            eps = float(np.clip(np.percentile(dmat[:, k_nn], 55), 0.08, 0.30))
            min_s = max(2, min(self.config.min_cluster_tracks, max(2, int(0.05 * n_samples))))
            labels = DBSCAN(eps=eps, min_samples=min_s, metric="cosine").fit_predict(features)
            if len(set(labels.tolist()) - {-1}) >= 2:
                return labels
            return AgglomerativeClustering(n_clusters=k_eff, linkage="ward").fit_predict(features)

        raise ValueError("clustering_method must be one of: kmeans, gmm, agglomerative, dbscan, hdbscan.")

    def _fit_clusters(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        method = self.config.clustering_method.lower()
        n_samples = len(features)

        if not self.config.auto_select_k or method in ("dbscan", "hdbscan") or n_samples < 6:
            labels = self._cluster_once(features, n_clusters)
            self.selected_k_ = len(set(labels.tolist()) - {-1})
            return labels

        max_k = min(max(2, int(self.config.max_candidate_clusters), int(n_clusters)), n_samples - 1)
        best_labels: Optional[np.ndarray] = None
        best_score = -1e9
        best_k = min(max(2, n_clusters), n_samples)

        for k_try in range(2, max_k + 1):
            labels = self._cluster_once(features, k_try)
            counts = Counter(labels.tolist())
            non_tiny = [c for c, cnt in counts.items() if c >= 0 and cnt >= max(2, int(0.03 * n_samples))]
            if len(non_tiny) < 2:
                continue
            try:
                sil = float(silhouette_score(features, labels, metric="cosine"))
            except ValueError:
                sil = 0.0

            ranked = sorted(counts, key=lambda c: (-counts[c], c))
            centers = {c: _l2(np.median(features[labels == c], axis=0)) for c in ranked}

            # Penalize over-splitting one playing team into two nearly identical sub-clusters
            extra_penalty = 0.0
            if k_try > 2 and len(ranked) >= 3:
                d_to_top2 = min(
                    _cosine_distance(centers[ranked[2]], centers[ranked[0]]),
                    _cosine_distance(centers[ranked[2]], centers[ranked[1]]),
                )
                if d_to_top2 < 0.16:
                    extra_penalty = 0.25

            score = sil - extra_penalty
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k_try

        if best_labels is None:
            best_labels = self._cluster_once(features, n_clusters)
            best_k = len(set(best_labels.tolist()) - {-1})

        self.selected_k_ = best_k
        return best_labels

    def _select_team_clusters(
        self,
        features: np.ndarray,
        appearances: Sequence[TrackAppearance],
        members: Dict[int, List[int]],
    ) -> Tuple[List[int], Dict[int, np.ndarray], Dict[int, float]]:
        n_total = len(features)
        centers_all = {cid: _l2(np.median(features[idxs], axis=0)) for cid, idxs in members.items()}
        spreads_all = {
            cid: float(np.median([_cosine_distance(features[i], centers_all[cid]) for i in idxs]))
            for cid, idxs in members.items()
        }

        # Compute simultaneous multi-player frame presence per cluster (strong signal
        # separating multi-player teams from 1-per-frame referees and goalkeepers).
        cluster_frame_counts: Dict[int, Counter[int]] = {cid: Counter() for cid in members}
        total_dense_frames = 0
        all_frame_occupancy: Counter[int] = Counter()
        for cid, idxs in members.items():
            for idx in idxs:
                for fno in set(appearances[idx].frame_numbers):
                    cluster_frame_counts[cid][fno] += 1
                    all_frame_occupancy[fno] += 1
        dense_frames = {fno for fno, cnt in all_frame_occupancy.items() if cnt >= 4}
        total_dense_frames = len(dense_frames)

        multi_player_rate: Dict[int, float] = {}
        for cid in members:
            if total_dense_frames >= 5:
                mp_frames = sum(1 for fno in dense_frames if cluster_frame_counts[cid][fno] >= 2)
                multi_player_rate[cid] = mp_frames / float(total_dense_frames)
            else:
                multi_player_rate[cid] = 0.0

        eligible = [cid for cid, idxs in members.items() if len(idxs) >= self.config.min_cluster_tracks]
        if len(eligible) < 2:
            eligible = sorted(members, key=lambda c: (-len(members[c]), c))
        if len(eligible) < 2:
            raise ValueError("Could not identify two non-noise appearance groups from the supplied tracks.")

        best_pair: Optional[Tuple[int, int]] = None
        best_score = -1e9
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                ca, cb = eligible[i], eligible[j]
                na, nb = len(members[ca]), len(members[cb])
                sep = _cosine_distance(centers_all[ca], centers_all[cb])
                cov_frac = (na + nb) / float(max(1, n_total))
                balance = min(na, nb) / float(max(na, nb))
                compactness = 1.0 - 0.5 * (spreads_all[ca] + spreads_all[cb])
                sep_term = min(1.0, sep / 0.25)

                if total_dense_frames >= 5:
                    # Reward pairs where BOTH clusters appear with >=2 simultaneous players
                    mp_term = 0.5 * (multi_player_rate[ca] + multi_player_rate[cb])
                    pair_score = (
                        0.38 * cov_frac
                        + 0.18 * balance
                        + 0.18 * sep_term
                        + 0.12 * compactness
                        + 0.14 * mp_term
                    )
                else:
                    pair_score = (
                        0.45 * cov_frac
                        + 0.20 * balance
                        + 0.20 * sep_term
                        + 0.15 * compactness
                    )

                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (ca, cb)

        assert best_pair is not None
        return list(best_pair), centers_all, spreads_all

    def fit(self, appearances: Sequence[TrackAppearance]) -> "AutomaticTeamDiscoverer":
        if len(appearances) < 2:
            raise ValueError("Automatic team discovery needs at least two reliable player tracks.")

        features = np.vstack([_l2(item.feature) for item in appearances]).astype(np.float32)
        n_clusters = min(max(2, self.config.candidate_clusters), len(appearances))
        cluster_ids = self._fit_clusters(features, n_clusters)

        members: Dict[int, List[int]] = defaultdict(list)
        for index, cluster_id in enumerate(cluster_ids.tolist()):
            if cluster_id >= 0:
                members[int(cluster_id)].append(index)

        selected, centers_all, spreads_all = self._select_team_clusters(features, appearances, members)

        # Deterministic ordering of Team_A vs Team_B using centroid values only
        ordered_teams = sorted(selected, key=lambda cid: tuple(np.round(centers_all[cid][:8], 5)))
        self.team_cluster_ids_ = {
            InternalLabel.TEAM_A.value: ordered_teams[0],
            InternalLabel.TEAM_B.value: ordered_teams[1],
        }
        self.cluster_centers_ = {cid: centers_all[cid] for cid in ordered_teams}
        self.cluster_spreads_ = {cid: spreads_all[cid] for cid in ordered_teams}
        self.inter_team_distance_ = _cosine_distance(
            centers_all[ordered_teams[0]], centers_all[ordered_teams[1]]
        )

        # Retain coherent outlier cluster centroids (referees, goalkeepers, distinct non-team groups)
        self.outlier_cluster_ids_ = sorted(set(members) - set(ordered_teams))
        self.outlier_centers_ = {
            cid: centers_all[cid]
            for cid in self.outlier_cluster_ids_
            if len(members[cid]) >= 2 and spreads_all[cid] <= 0.35
        }

        # Robust MAD/IQR cluster radii bounded by inter-team separation
        self.cluster_radii_ = {}
        cap_radius = min(0.36, max(0.08, 0.78 * self.inter_team_distance_))
        for cid in ordered_teams:
            dists = np.array(
                [_cosine_distance(features[idx], centers_all[cid]) for idx in members[cid]],
                dtype=np.float64,
            )
            med = float(np.median(dists))
            mad = float(np.median(np.abs(dists - med))) * 1.4826
            q75, q25 = float(np.percentile(dists, 75)), float(np.percentile(dists, 25))
            iqr_std = (q75 - q25) / 1.349
            robust_sigma = max(mad, iqr_std, 0.025)
            raw_r = med + float(self.config.radius_mad_multiplier) * robust_sigma
            self.cluster_radii_[cid] = float(np.clip(raw_r, 0.07, cap_radius))

        # Compute cluster silhouette diagnostic when >= 2 clusters and >= 3 samples
        valid_mask = cluster_ids >= 0
        if int(np.sum(valid_mask)) >= 3 and len(set(cluster_ids[valid_mask].tolist())) >= 2:
            try:
                self.silhouette_score_ = round(
                    float(silhouette_score(features[valid_mask], cluster_ids[valid_mask], metric="cosine")),
                    4,
                )
            except ValueError:
                self.silhouette_score_ = None
        else:
            self.silhouette_score_ = None

        self.track_labels_.clear()
        self.track_distances_.clear()
        self.segment_labels_.clear()
        self.segment_predictions_.clear()
        for item in appearances:
            pred = self.predict_feature_detailed(item.feature)
            self.segment_labels_[item.segment_key] = pred.predicted_label
            self.segment_predictions_[item.segment_key] = pred
            self.track_labels_[item.tracking_id] = pred.predicted_label
            self.track_distances_[item.tracking_id] = pred.nearest_distance

        self.is_fitted = True
        return self

    def predict_feature_detailed(self, feature: np.ndarray) -> ClassificationPrediction:
        """Predict team assignment from a normalized feature vector with full uncertainty telemetry."""
        if not self.cluster_centers_:
            raise RuntimeError("Fit AutomaticTeamDiscoverer before prediction.")

        normalized = _l2(feature)
        cid_a = self.team_cluster_ids_[InternalLabel.TEAM_A.value]
        cid_b = self.team_cluster_ids_[InternalLabel.TEAM_B.value]
        d_a = _cosine_distance(normalized, self.cluster_centers_[cid_a])
        d_b = _cosine_distance(normalized, self.cluster_centers_[cid_b])

        if d_a <= d_b:
            raw_label, chosen_cid, nearest_d, second_d = (
                InternalLabel.TEAM_A.value,
                cid_a,
                d_a,
                d_b,
            )
        else:
            raw_label, chosen_cid, nearest_d, second_d = (
                InternalLabel.TEAM_B.value,
                cid_b,
                d_b,
                d_a,
            )

        margin = max(0.0, second_d - nearest_d)
        ratio = nearest_d / max(second_d, 1e-6)
        effective_radius = self.cluster_radii_.get(chosen_cid, 0.12) * max(
            0.5, float(self.config.assignment_distance_multiplier)
        )

        d_outlier = min(
            (_cosine_distance(normalized, oc) for oc in self.outlier_centers_.values()),
            default=999.0,
        )

        radial_conf = float(np.clip(1.0 - nearest_d / max(effective_radius * 1.45, 0.25), 0.0, 1.0))
        margin_conf = float(
            np.clip(margin / max(0.12, self.inter_team_distance_ * 0.45), 0.0, 1.0)
        )
        base_conf = round(0.55 * radial_conf + 0.45 * margin_conf, 4)

        rejection_reason: Optional[str] = None
        if d_outlier < nearest_d * float(self.config.outlier_repulsion_ratio):
            rejection_reason = f"closer_to_outlier_cluster({d_outlier:.3f}<{nearest_d:.3f})"
        elif nearest_d > effective_radius:
            rejection_reason = f"outside_discovered_cluster_radius({nearest_d:.3f}>{effective_radius:.3f})"
        elif margin < float(self.config.min_team_margin):
            rejection_reason = f"low_inter_team_margin({margin:.3f}<{self.config.min_team_margin:.3f})"
        elif ratio > float(self.config.max_distance_ratio):
            rejection_reason = f"high_distance_ratio({ratio:.3f}>{self.config.max_distance_ratio:.3f})"

        if rejection_reason is not None:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=raw_label,
                nearest_distance=round(nearest_d, 4),
                second_nearest_distance=round(second_d, 4),
                margin=round(margin, 4),
                confidence_score=round(base_conf * 0.35, 4),
                is_unknown=True,
                rejection_reason=rejection_reason,
            )

        return ClassificationPrediction(
            predicted_label=raw_label,
            raw_nearest_label=raw_label,
            nearest_distance=round(nearest_d, 4),
            second_nearest_distance=round(second_d, 4),
            margin=round(margin, 4),
            confidence_score=base_conf,
            is_unknown=False,
            rejection_reason=None,
        )

    def predict_feature(self, feature: np.ndarray) -> Tuple[str, float]:
        pred = self.predict_feature_detailed(feature)
        return pred.predicted_label, pred.nearest_distance

    def predict_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        pred = self.predict_classification(crop_bgr)
        return pred.predicted_label, pred.nearest_distance

    def predict_track(
        self,
        tracking_id: Optional[int],
        crop_bgr: Optional[np.ndarray] = None,
        shot_id: int = 0,
    ) -> Tuple[str, float]:
        if tracking_id is not None:
            seg_key = (int(shot_id), int(tracking_id))
            if seg_key in self.segment_predictions_:
                pred = self.segment_predictions_[seg_key]
                return pred.predicted_label, pred.nearest_distance
            if int(tracking_id) in self.track_labels_:
                return self.track_labels_[int(tracking_id)], self.track_distances_[int(tracking_id)]
        if crop_bgr is None or crop_bgr.size == 0:
            return InternalLabel.UNKNOWN.value, 1.0
        return self.predict_crop(crop_bgr)

    def predict_classification(
        self,
        crop_bgr: Optional[np.ndarray],
        feature: Optional[np.ndarray] = None,
    ) -> ClassificationPrediction:
        """Expose discovery through the pipeline's existing classifier contract."""
        if feature is not None and feature.size > 0:
            return self.predict_feature_detailed(feature)
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
        extractor = get_feature_extractor(self.config.feature_method)
        return self.predict_feature_detailed(extractor(crop_bgr))

    def report(self, appearances: Sequence[TrackAppearance]) -> Dict[str, Any]:
        counts = Counter(self.segment_labels_.values() if self.segment_labels_ else self.track_labels_.values())
        return {
            "mode": "unsupervised_per_video_adaptation",
            "feature_method": self.config.feature_method,
            "clustering_method": self.config.clustering_method,
            "candidate_clusters": self.config.candidate_clusters,
            "selected_k": self.selected_k_,
            "reliable_tracks": len(appearances),
            "team_cluster_ids": self.team_cluster_ids_,
            "excluded_cluster_ids": self.outlier_cluster_ids_,
            "retained_outlier_cluster_count": len(self.outlier_centers_),
            "inter_team_cosine_distance": round(self.inter_team_distance_, 4),
            "silhouette_score": self.silhouette_score_,
            "cluster_radii_cosine_distance": {str(k): round(v, 4) for k, v in self.cluster_radii_.items()},
            "cluster_spreads_median_cosine": {str(k): round(v, 4) for k, v in self.cluster_spreads_.items()},
            "track_label_counts": dict(counts),
            "team_name_note": "Team_A and Team_B are visual clusters, not inferred club or national-team names.",
        }


@dataclass
class _BufferedTrackState:
    last_frame: int
    last_bbox: Tuple[int, int, int, int]
    features: List[np.ndarray] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)


class TrackAppearanceBuffer:
    """Online track-level descriptor aggregator for second-pass or streaming inference.

    Prevents single noisy crops from flipping a player's team assignment, resets
    track buffers across camera cuts or implausible spatial jumps, and never
    blindly assumes Pass-1 tracking IDs match Pass-2 tracking IDs.
    """

    def __init__(
        self,
        discoverer: AutomaticTeamDiscoverer,
        max_buffer_samples: int = 12,
        max_missing_frames: int = 30,
        max_spatial_jump_ratio: float = 3.5,
    ) -> None:
        self.discoverer = discoverer
        self.extractor = get_feature_extractor(discoverer.config.feature_method)
        self.max_buffer_samples = max(2, int(max_buffer_samples))
        self.max_missing_frames = max(1, int(max_missing_frames))
        self.max_spatial_jump_ratio = float(max_spatial_jump_ratio)
        self.shot_id: int = 0
        self._buffers: Dict[Tuple[int, int], _BufferedTrackState] = {}

    def on_camera_cut(self) -> None:
        """Advance shot counter so no appearance history leaks across a camera cut."""
        self.shot_id += 1

    def _should_reset(
        self,
        state: _BufferedTrackState,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        gap = frame_number - state.last_frame
        if gap > self.max_missing_frames:
            return True
        px1, py1, px2, py2 = state.last_bbox
        cx1, cy1, cx2, cy2 = bbox
        dist = math.hypot((cx1 + cx2) / 2.0 - (px1 + px2) / 2.0, (cy1 + cy2) / 2.0 - (py1 + py2) / 2.0)
        ref_h = max(25.0, ((py2 - py1) + (cy2 - cy1)) / 2.0)
        return dist > self.max_spatial_jump_ratio * ref_h * max(1, min(gap, 5))

    def classify_observation(
        self,
        tracking_id: Optional[int],
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        crop_bgr: Optional[np.ndarray],
        feature: Optional[np.ndarray] = None,
        quality_weight: float = 1.0,
    ) -> ClassificationPrediction:
        """Aggregate clean torso features per track and classify the track-level median descriptor."""
        feat: Optional[np.ndarray] = None
        if feature is not None and feature.size > 0:
            feat = _l2(feature)
        elif crop_bgr is not None and crop_bgr.size > 0:
            feat = _l2(self.extractor(crop_bgr))

        if tracking_id is None:
            if feat is None:
                return self.discoverer.predict_classification(None)
            return self.discoverer.predict_feature_detailed(feat)

        key = (self.shot_id, int(tracking_id))
        state = self._buffers.get(key)
        if state is None:
            state = _BufferedTrackState(last_frame=frame_number, last_bbox=bbox)
            self._buffers[key] = state
        elif self._should_reset(state, frame_number, bbox):
            state.features.clear()
            state.weights.clear()

        state.last_frame = frame_number
        state.last_bbox = bbox

        if feat is not None:
            state.features.append(feat)
            state.weights.append(max(0.05, float(quality_weight)))
            if len(state.features) > self.max_buffer_samples:
                # Retain evenly spread + most recent observations (drop oldest low-weight entry)
                state.features.pop(0)
                state.weights.pop(0)

        if not state.features:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=InternalLabel.UNKNOWN.value,
                nearest_distance=1.0,
                second_nearest_distance=1.0,
                margin=0.0,
                confidence_score=0.0,
                is_unknown=True,
                rejection_reason="filtered_crop_no_track_history",
            )

        agg_feature = _weighted_trimmed_aggregate(state.features, state.weights)
        return self.discoverer.predict_feature_detailed(agg_feature)


def evaluate_discovered_labels(
    predicted_labels: Sequence[str],
    reference_labels: Sequence[str],
) -> Dict[str, Any]:
    """Evaluate unsupervised team discovery after best Team_A/Team_B permutation, never during fitting.

    Explicitly separates accuracy on covered predictions from overall accuracy
    including rejected samples, and reports referee/unknown misclassification rates
    whenever reference annotations include those classes.
    """
    if len(predicted_labels) != len(reference_labels):
        raise ValueError("predicted_labels and reference_labels must have equal length.")

    if not reference_labels:
        return {
            "metrics_available": False,
            "permutation_applied": False,
            "classification_coverage": 0.0,
            "unknown_prediction_rate": 0.0,
            "clustering_accuracy_on_covered": None,
            "macro_f1_on_covered": None,
            "adjusted_rand_index": None,
            "normalized_mutual_information": None,
            "labels": [InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value],
            "confusion_matrix": [],
        }

    positive = {InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value}
    team_pairs = [(p, r) for p, r in zip(predicted_labels, reference_labels) if r in positive]
    direct = sum(int(p == r) for p, r in team_pairs)
    swapped = sum(
        int(
            (
                InternalLabel.TEAM_B.value
                if p == InternalLabel.TEAM_A.value
                else InternalLabel.TEAM_A.value
                if p == InternalLabel.TEAM_B.value
                else p
            )
            == r
        )
        for p, r in team_pairs
    )
    use_swap = swapped > direct
    aligned = [
        InternalLabel.TEAM_B.value
        if use_swap and p == InternalLabel.TEAM_A.value
        else InternalLabel.TEAM_A.value
        if use_swap and p == InternalLabel.TEAM_B.value
        else p
        for p in predicted_labels
    ]

    covered = [idx for idx, val in enumerate(aligned) if val in positive]
    covered_true = [reference_labels[idx] for idx in covered]
    covered_pred = [aligned[idx] for idx in covered]

    team_indices = [idx for idx, val in enumerate(reference_labels) if val in positive]
    ref_indices = [idx for idx, val in enumerate(reference_labels) if val == InternalLabel.REFEREE.value]
    unk_indices = [idx for idx, val in enumerate(reference_labels) if val == InternalLabel.UNKNOWN.value]

    # Collapsed 3-way target: non-team GT (Referee / Unknown / GK) should be predicted as Unknown
    gt_3way = [r if r in positive else InternalLabel.UNKNOWN.value for r in reference_labels]
    labels_2 = [InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value]
    labels_3 = labels_2 + [InternalLabel.UNKNOWN.value]

    # Per-team P/R/F1 on covered predictions and across all samples
    p_cov, r_cov, f1_cov_arr, sup_cov = precision_recall_fscore_support(
        covered_true, covered_pred, labels=labels_2, zero_division=0
    ) if covered else (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=int))

    p_all, r_all, f1_all_arr, sup_all = precision_recall_fscore_support(
        gt_3way, aligned, labels=labels_3, zero_division=0
    )

    per_team_on_covered = {
        lbl: {
            "precision": round(float(p_cov[i]), 4),
            "recall": round(float(r_cov[i]), 4),
            "f1": round(float(f1_cov_arr[i]), 4),
            "support": int(sup_cov[i]),
        }
        for i, lbl in enumerate(labels_2)
    }
    per_class_all_samples = {
        lbl: {
            "precision": round(float(p_all[i]), 4),
            "recall": round(float(r_all[i]), 4),
            "f1": round(float(f1_all_arr[i]), 4),
            "support": int(sup_all[i]),
        }
        for i, lbl in enumerate(labels_3)
    }

    n_total = len(reference_labels)
    team_covered = sum(1 for i in team_indices if aligned[i] in positive)
    team_correct_all = sum(1 for i in team_indices if aligned[i] == reference_labels[i])
    overall_3way_correct = sum(1 for p, r in zip(aligned, gt_3way) if p == r)

    ref_mis_rate = (
        round(sum(1 for i in ref_indices if aligned[i] in positive) / float(len(ref_indices)), 4)
        if ref_indices
        else None
    )
    unk_mis_rate = (
        round(sum(1 for i in unk_indices if aligned[i] in positive) / float(len(unk_indices)), 4)
        if unk_indices
        else None
    )

    return {
        "metrics_available": True,
        "permutation_applied": use_swap,
        "total_samples": n_total,
        "covered_samples": len(covered),
        "classification_coverage": round(len(covered) / float(n_total), 4),
        "team_coverage": round(team_covered / float(len(team_indices)), 4) if team_indices else None,
        "unknown_prediction_rate": round(
            sum(p == InternalLabel.UNKNOWN.value for p in aligned) / float(n_total), 4
        ),
        "clustering_accuracy_on_covered": round(
            sum(p == r for p, r in zip(covered_pred, covered_true)) / float(len(covered)), 4
        )
        if covered
        else None,
        "macro_f1_on_covered": round(
            float(f1_score(covered_true, covered_pred, labels=labels_2, average="macro", zero_division=0)),
            4,
        )
        if covered
        else None,
        "team_accuracy_including_rejected": round(team_correct_all / float(len(team_indices)), 4)
        if team_indices
        else None,
        "overall_accuracy_including_rejected": round(overall_3way_correct / float(n_total), 4),
        "macro_f1_including_unknown": round(
            float(f1_score(gt_3way, aligned, labels=labels_3, average="macro", zero_division=0)),
            4,
        ),
        "per_team_on_covered": per_team_on_covered,
        "per_class_all_samples": per_class_all_samples,
        "referee_misclassification_rate": ref_mis_rate,
        "unknown_misclassification_rate": unk_mis_rate,
        "adjusted_rand_index": round(float(adjusted_rand_score(reference_labels, aligned)), 4),
        "adjusted_rand_index_3way": round(float(adjusted_rand_score(gt_3way, aligned)), 4),
        "normalized_mutual_information": round(
            float(normalized_mutual_info_score(reference_labels, aligned)), 4
        ),
        "normalized_mutual_information_3way": round(
            float(normalized_mutual_info_score(gt_3way, aligned)), 4
        ),
        "labels": labels_2,
        "labels_with_unknown": labels_3,
        "confusion_matrix": confusion_matrix(reference_labels, aligned, labels=labels_3).tolist(),
        "confusion_matrix_3way": confusion_matrix(gt_3way, aligned, labels=labels_3).tolist(),
        "denominators": {
            "clustering_accuracy_on_covered": f"{len(covered)} predictions assigned Team_A or Team_B",
            "team_accuracy_including_rejected": f"{len(team_indices)} ground-truth playing-team samples (rejected Unknown counts as miss)",
            "overall_accuracy_including_rejected": f"{n_total} total samples (GT Referee/Unknown mapped to expected Unknown)",
            "referee_misclassification_rate": f"{len(ref_indices)} ground-truth Referee samples",
            "unknown_misclassification_rate": f"{len(unk_indices)} ground-truth Unknown/GK samples",
        },
    }
