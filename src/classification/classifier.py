"""Supervised and unsupervised team classifiers with Unknown rejection and calibration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from src.classification.features import get_feature_extractor
from src.config import ClassificationConfig
from src.schemas import ClassificationPrediction, InternalLabel


class TeamClassifier:
    """
    Match-aware team classifier supporting KNN, Linear SVM, RBF SVM, Logistic Regression,
    and Nearest Centroid, with explicit Unknown distance-rejection and validation calibration.
    """

    def __init__(
        self,
        match_id: str,
        config: Optional[ClassificationConfig] = None,
    ) -> None:
        self.match_id = match_id
        self.config = config or ClassificationConfig()
        self.feature_fn = get_feature_extractor(self.config.feature_method)
        self.model: Any = None
        self.nn_index: Optional[KNeighborsClassifier] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: List[str] = []
        self.class_counts_: Dict[str, int] = {}
        self.class_priors_inv_: Dict[str, float] = {}
        self.class_centroids_: Dict[str, np.ndarray] = {}
        self.unknown_threshold: float = float(self.config.unknown_threshold)
        self.min_margin: float = float(self.config.min_margin)
        self.is_fitted: bool = False

    def extract_features(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract feature vector for a single BGR torso crop."""
        return self.feature_fn(crop_bgr)

    def fit(
        self,
        crops_or_features: Sequence[np.ndarray],
        labels: Sequence[str],
        features_precomputed: bool = False,
    ) -> "TeamClassifier":
        """
        Train the classifier on positive class examples (`Team_A`, `Team_B`, `Referee`).
        Any sample explicitly labelled `Unknown` is excluded from positive neighbor templates
        (and used instead during threshold calibration).
        """
        if len(crops_or_features) != len(labels):
            raise ValueError("Length of crops_or_features and labels must match.")

        X_list: List[np.ndarray] = []
        y_list: List[str] = []
        for item, lbl in zip(crops_or_features, labels):
            if lbl == InternalLabel.UNKNOWN.value:
                continue
            if lbl not in InternalLabel.valid_training_labels():
                raise ValueError(
                    f"Invalid internal training label '{lbl}'. Expected one of {InternalLabel.valid_training_labels()}."
                )
            feat = item.flatten().astype(np.float32) if features_precomputed else self.extract_features(item)
            X_list.append(feat)
            y_list.append(lbl)

        if len(X_list) < 3 or len(set(y_list)) < 2:
            raise ValueError(
                f"Insufficient training data: got {len(X_list)} samples across {len(set(y_list))} classes. "
                "Need at least 3 samples across at least 2 distinct positive classes."
            )

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=str)

        self.X_train = X
        self.y_train = y
        counts = Counter(y.tolist())
        self.class_counts_ = dict(counts)
        self.classes_ = sorted(counts.keys())

        # Compute inverse class frequency weights (dampened square-root inverse frequency)
        max_count = float(max(counts.values()))
        self.class_priors_inv_ = {
            cls: float(np.sqrt(max_count / max(1.0, float(cnt)))) for cls, cnt in counts.items()
        }

        # Compute per-class L2-normalized centroids for distance & margin diagnostics
        self.class_centroids_ = {}
        for cls in self.classes_:
            cls_vecs = X[y == cls]
            centroid = np.mean(cls_vecs, axis=0)
            norm = float(np.linalg.norm(centroid))
            self.class_centroids_[cls] = centroid / (norm + 1e-8)

        # Always fit a nearest-neighbor index for exact distance diagnostics
        k_eff = min(int(self.config.n_neighbors), len(X))
        self.nn_index = KNeighborsClassifier(
            n_neighbors=k_eff,
            weights="distance",
            metric=self.config.metric,
        )
        self.nn_index.fit(X, y)

        ctype = self.config.classifier_type.lower()
        if ctype == "knn":
            self.model = self.nn_index
        elif ctype == "svm_linear":
            self.model = SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced" if self.config.balance_class_weights else None,
                random_state=self.config.random_state,
            )
            self.model.fit(X, y)
        elif ctype == "svm_rbf":
            self.model = SVC(
                kernel="rbf",
                C=2.0,
                class_weight="balanced" if self.config.balance_class_weights else None,
                random_state=self.config.random_state,
            )
            self.model.fit(X, y)
        elif ctype == "logistic":
            self.model = LogisticRegression(
                C=2.0,
                class_weight="balanced" if self.config.balance_class_weights else None,
                max_iter=500,
                random_state=self.config.random_state,
            )
            self.model.fit(X, y)
        elif ctype == "nearest_centroid":
            self.model = NearestCentroid(metric="euclidean")
            self.model.fit(X, y)
        else:
            raise ValueError(
                f"Unsupported classifier_type '{self.config.classifier_type}'. "
                "Choose from: knn, svm_linear, svm_rbf, logistic, nearest_centroid."
            )

        self.is_fitted = True
        return self

    def _compute_distances_and_vote(
        self,
        feat_2d: np.ndarray,
    ) -> Tuple[str, float, float, float]:
        """
        Compute the predicted positive class, nearest distance, second-nearest class distance, and margin.
        """
        assert self.nn_index is not None and self.X_train is not None and self.y_train is not None

        k_all = len(self.X_train)
        distances, indices = self.nn_index.kneighbors(feat_2d, n_neighbors=k_all)
        dists = distances[0]
        idxs = indices[0]
        neighbor_labels = self.y_train[idxs]

        # Per-class minimum distance (1-NN distance to each class)
        class_min_dist: Dict[str, float] = {}
        for d, lbl in zip(dists, neighbor_labels):
            if lbl not in class_min_dist:
                class_min_dist[lbl] = float(d)
                if len(class_min_dist) == len(self.classes_):
                    break

        sorted_class_dists = sorted(class_min_dist.items(), key=lambda kv: kv[1])
        nearest_dist = float(sorted_class_dists[0][1])
        second_nearest_dist = (
            float(sorted_class_dists[1][1]) if len(sorted_class_dists) > 1 else nearest_dist + 0.5
        )
        margin = max(0.0, second_nearest_dist - nearest_dist)

        ctype = self.config.classifier_type.lower()
        if ctype == "knn":
            k_eff = min(int(self.config.n_neighbors), k_all)
            top_dists = dists[:k_eff]
            top_labels = neighbor_labels[:k_eff]
            scores: Dict[str, float] = {cls: 0.0 for cls in self.classes_}
            for d, lbl in zip(top_dists, top_labels):
                w = 1.0 / (float(d) + 1e-5)
                if self.config.balance_class_weights:
                    w *= self.class_priors_inv_.get(lbl, 1.0)
                scores[lbl] += w
            best_label = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            best_label = str(self.model.predict(feat_2d)[0])
            # Align nearest_dist with the chosen class's minimum neighbor distance
            nearest_dist = float(class_min_dist.get(best_label, nearest_dist))
            other_dists = [d for c, d in class_min_dist.items() if c != best_label]
            second_nearest_dist = float(min(other_dists)) if other_dists else nearest_dist + 0.5
            margin = max(0.0, second_nearest_dist - nearest_dist)

        return best_label, nearest_dist, second_nearest_dist, margin

    def predict_crop(
        self,
        crop_bgr: Optional[np.ndarray],
        match_id: Optional[str] = None,
        allow_cross_match: bool = False,
    ) -> ClassificationPrediction:
        """
        Classify a single BGR torso crop with distance-based Unknown rejection.
        """
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before calling predict_crop.")

        if match_id is not None and match_id != self.match_id and not allow_cross_match:
            raise ValueError(
                f"Cross-match safety check failed: classifier was trained on '{self.match_id}' "
                f"but evaluated on '{match_id}'. Pass allow_cross_match=True only for explicit "
                "cross-match domain-shift experiments."
            )

        if crop_bgr is None or crop_bgr.size == 0:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=InternalLabel.UNKNOWN.value,
                nearest_distance=1.0,
                second_nearest_distance=1.0,
                margin=0.0,
                confidence_score=0.0,
                is_unknown=True,
                rejection_reason="empty_crop",
            )

        feat = self.extract_features(crop_bgr).reshape(1, -1)
        return self.predict_feature(feat)

    def predict_feature(self, feat: np.ndarray) -> ClassificationPrediction:
        """Classify a precomputed feature vector (1D or 2D `(1, D)`)."""
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before calling predict_feature.")

        feat_2d = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        best_label, nearest_dist, second_dist, margin = self._compute_distances_and_vote(feat_2d)

        # Heuristic similarity score in [0, 1] (explicitly NOT a calibrated probability)
        sim_score = float(np.clip(1.0 - nearest_dist / max(self.unknown_threshold * 1.5, 0.5), 0.0, 1.0))

        if nearest_dist > self.unknown_threshold:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=best_label,
                nearest_distance=nearest_dist,
                second_nearest_distance=second_dist,
                margin=margin,
                confidence_score=sim_score * 0.3,
                is_unknown=True,
                rejection_reason=f"distance({nearest_dist:.3f})>threshold({self.unknown_threshold:.3f})",
            )

        if margin < self.min_margin and len(self.classes_) > 1:
            return ClassificationPrediction(
                predicted_label=InternalLabel.UNKNOWN.value,
                raw_nearest_label=best_label,
                nearest_distance=nearest_dist,
                second_nearest_distance=second_dist,
                margin=margin,
                confidence_score=sim_score * 0.4,
                is_unknown=True,
                rejection_reason=f"margin({margin:.3f})<min_margin({self.min_margin:.3f})",
            )

        return ClassificationPrediction(
            predicted_label=best_label,
            raw_nearest_label=best_label,
            nearest_distance=nearest_dist,
            second_nearest_distance=second_dist,
            margin=margin,
            confidence_score=sim_score,
            is_unknown=False,
            rejection_reason=None,
        )

    def calibrate_unknown_threshold(
        self,
        val_crops_or_features: Sequence[np.ndarray],
        val_labels: Sequence[str],
        features_precomputed: bool = False,
        candidate_thresholds: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Select the optimal Unknown distance threshold using validation data,
        and generate a coverage-versus-accuracy trade-off curve.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit classifier before calibrating unknown threshold.")

        raw_preds: List[str] = []
        nearest_dists: List[float] = []
        margins: List[float] = []

        for item in val_crops_or_features:
            feat = (
                np.asarray(item, dtype=np.float32).reshape(1, -1)
                if features_precomputed
                else self.extract_features(item).reshape(1, -1)
            )
            lbl, nd, _, mg = self._compute_distances_and_vote(feat)
            raw_preds.append(lbl)
            nearest_dists.append(nd)
            margins.append(mg)

        dists_arr = np.asarray(nearest_dists, dtype=np.float32)
        y_true = list(val_labels)

        if candidate_thresholds is None:
            min_d = max(0.05, float(np.min(dists_arr)) * 0.8)
            max_d = min(1.20, float(np.max(dists_arr)) * 1.25 + 0.05)
            grid = np.unique(
                np.concatenate([
                    np.linspace(min_d, max_d, 25),
                    np.quantile(dists_arr, np.linspace(0.1, 1.0, 15)),
                    np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75]),
                ])
            )
        else:
            grid = np.asarray(candidate_thresholds, dtype=np.float32)

        curve: List[Dict[str, float]] = []
        best_threshold = self.unknown_threshold
        best_objective = -1.0

        known_mask = np.array([y != InternalLabel.UNKNOWN.value for y in y_true], dtype=bool)
        unknown_mask = ~known_mask

        for thresh in sorted(grid):
            t_val = float(thresh)
            preds_at_t = [
                raw_p if (d <= t_val and m >= self.min_margin) else InternalLabel.UNKNOWN.value
                for raw_p, d, m in zip(raw_preds, nearest_dists, margins)
            ]

            overall_acc = float(accuracy_score(y_true, preds_at_t))
            macro_f1 = float(
                f1_score(
                    y_true,
                    preds_at_t,
                    labels=list(InternalLabel.all_labels()),
                    average="macro",
                    zero_division=0,
                )
            )
            positive_f1 = float(
                f1_score(
                    y_true,
                    preds_at_t,
                    labels=list(InternalLabel.valid_training_labels()),
                    average="macro",
                    zero_division=0,
                )
            )

            accepted = [i for i, p in enumerate(preds_at_t) if p != InternalLabel.UNKNOWN.value]
            coverage = len(accepted) / float(len(y_true)) if y_true else 0.0
            unknown_rate = 1.0 - coverage

            if accepted:
                selective_acc = float(
                    accuracy_score([y_true[i] for i in accepted], [preds_at_t[i] for i in accepted])
                )
            else:
                selective_acc = 0.0

            # Objective balances overall accuracy, positive-class macro F1, and unknown recall
            unk_recall = 1.0
            if np.any(unknown_mask):
                unk_indices = np.where(unknown_mask)[0]
                unk_recall = float(
                    sum(1 for i in unk_indices if preds_at_t[int(i)] == InternalLabel.UNKNOWN.value)
                ) / float(len(unk_indices))

            # Prefer tighter thresholds when accuracy/F1 are tied so OOD Unknowns are rejected
            objective = 0.45 * overall_acc + 0.40 * positive_f1 + 0.15 * unk_recall - 0.01 * t_val

            curve.append(
                {
                    "threshold": round(t_val, 4),
                    "overall_accuracy": round(overall_acc, 4),
                    "selective_accuracy": round(selective_acc, 4),
                    "coverage": round(coverage, 4),
                    "unknown_rate": round(unknown_rate, 4),
                    "macro_f1": round(macro_f1, 4),
                    "positive_macro_f1": round(positive_f1, 4),
                    "unknown_recall": round(unk_recall, 4),
                    "objective": round(objective, 4),
                }
            )

            if objective > best_objective:
                best_objective = objective
                best_threshold = t_val

        if self.config.auto_calibrate_threshold:
            self.unknown_threshold = best_threshold

        known_dists = dists_arr[known_mask].tolist() if np.any(known_mask) else []
        unk_dists = dists_arr[unknown_mask].tolist() if np.any(unknown_mask) else []

        return {
            "selected_threshold": round(float(best_threshold), 4),
            "default_threshold": round(float(self.config.unknown_threshold), 4),
            "known_distance_summary": {
                "count": len(known_dists),
                "min": round(float(np.min(known_dists)), 4) if known_dists else None,
                "median": round(float(np.median(known_dists)), 4) if known_dists else None,
                "p90": round(float(np.percentile(known_dists, 90)), 4) if known_dists else None,
                "max": round(float(np.max(known_dists)), 4) if known_dists else None,
            },
            "unknown_distance_summary": {
                "count": len(unk_dists),
                "min": round(float(np.min(unk_dists)), 4) if unk_dists else None,
                "median": round(float(np.median(unk_dists)), 4) if unk_dists else None,
                "max": round(float(np.max(unk_dists)), 4) if unk_dists else None,
            },
            "curve": curve,
        }

    def save(self, path: str | Path) -> Path:
        """Serialize the trained classifier and its configuration metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted TeamClassifier.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "match_id": self.match_id,
            "config": self.config,
            "model": self.model,
            "nn_index": self.nn_index,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "classes_": self.classes_,
            "class_counts_": self.class_counts_,
            "class_priors_inv_": self.class_priors_inv_,
            "class_centroids_": self.class_centroids_,
            "unknown_threshold": self.unknown_threshold,
            "min_margin": self.min_margin,
        }
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TeamClassifier":
        """Load a serialized TeamClassifier from disk."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {source}")
        payload = joblib.load(source)
        instance = cls(match_id=payload["match_id"], config=payload["config"])
        instance.model = payload["model"]
        instance.nn_index = payload["nn_index"]
        instance.X_train = payload["X_train"]
        instance.y_train = payload["y_train"]
        instance.classes_ = payload["classes_"]
        instance.class_counts_ = payload["class_counts_"]
        instance.class_priors_inv_ = payload["class_priors_inv_"]
        instance.class_centroids_ = payload["class_centroids_"]
        instance.unknown_threshold = float(payload["unknown_threshold"])
        instance.min_margin = float(payload["min_margin"])
        instance.is_fitted = True
        return instance


class UnsupervisedKMeansTeamClusterer:
    """
    Unsupervised K-Means jersey colour clusterer (preserved and improved from Cell 2).
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_to_label: Dict[int, str] = {}
        self.is_fitted = False

    @staticmethod
    def extract_median_bgr(crop: np.ndarray) -> np.ndarray:
        return np.median(crop, axis=(0, 1)).astype(np.float32)

    def fit(
        self,
        crops: Sequence[np.ndarray],
        cluster_mapping: Optional[Dict[int, str]] = None,
    ) -> "UnsupervisedKMeansTeamClusterer":
        if len(crops) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} crops to fit KMeans.")
        colours = np.asarray([self.extract_median_bgr(c) for c in crops], dtype=np.float32)
        self.kmeans.fit(colours)
        self.cluster_to_label = cluster_mapping or {
            i: f"Cluster_{i}" for i in range(self.n_clusters)
        }
        self.is_fitted = True
        return self

    def predict(self, crop: np.ndarray) -> str:
        if not self.is_fitted:
            raise RuntimeError("Fit UnsupervisedKMeansTeamClusterer before calling predict.")
        colour = self.extract_median_bgr(crop).reshape(1, 3)
        cluster_id = int(self.kmeans.predict(colour)[0])
        return self.cluster_to_label.get(cluster_id, InternalLabel.UNKNOWN.value)
