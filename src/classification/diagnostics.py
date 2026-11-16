"""Automated label auditing, prototype distance analysis, and dataset diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from src.classification.features import get_feature_extractor
from src.schemas import InternalLabel, JerseyCropMetadata


def audit_dataset_labels(
    samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    feature_method: str = "spatial_embedding",
) -> Dict[str, Any]:
    """
    Audit labelled jersey crops using Leave-One-Out (LOO) nearest-neighbor and class-centroid
    consistency checks. Flags any sample whose appearance strongly contradicts its assigned label
    (e.g. a referee or opposing player accidentally included in another team's index list).
    """
    if len(samples) < 4:
        return {"total_checked": len(samples), "suspect_samples": [], "loo_consistency": 1.0}

    feature_fn = get_feature_extractor(feature_method)
    feats = np.asarray([feature_fn(img) for _, img, _ in samples], dtype=np.float32)
    labels = [lbl for _, _, lbl in samples]
    metas = [meta for meta, _, _ in samples]

    # Normalize features for cosine distance
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats_norm = feats / norms

    # Compute per-class centroids excluding Unknown
    pos_classes = sorted({l for l in labels if l != InternalLabel.UNKNOWN.value})
    centroids: Dict[str, np.ndarray] = {}
    for cls in pos_classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        if idx:
            c = np.mean(feats_norm[idx], axis=0)
            centroids[cls] = c / (np.linalg.norm(c) + 1e-8)

    # Pairwise cosine distance matrix: 1 - dot(u, v)
    sim_matrix = np.clip(feats_norm @ feats_norm.T, -1.0, 1.0)
    dist_matrix = 1.0 - sim_matrix

    suspect_samples: List[Dict[str, Any]] = []
    consistent_count = 0
    checked_count = 0

    for i, (meta, lbl) in enumerate(zip(metas, labels)):
        if lbl == InternalLabel.UNKNOWN.value:
            continue
        checked_count += 1

        # Leave-one-out nearest neighbor among all other positive samples
        other_indices = [j for j in range(len(samples)) if j != i and labels[j] != InternalLabel.UNKNOWN.value]
        other_dists = [(dist_matrix[i, j], labels[j], metas[j].crop_id) for j in other_indices]
        other_dists.sort(key=lambda x: x[0])

        nearest_dist, nearest_lbl, nearest_crop_id = other_dists[0]

        # Distance to own class centroid vs other class centroids (leave-one-out for own class)
        own_other_idx = [j for j in other_indices if labels[j] == lbl]
        if own_other_idx:
            own_c = np.mean(feats_norm[own_other_idx], axis=0)
            own_c /= np.linalg.norm(own_c) + 1e-8
            own_centroid_dist = float(1.0 - np.dot(feats_norm[i], own_c))
        else:
            own_centroid_dist = 0.0

        best_other_cls = lbl
        best_other_centroid_dist = 1.0
        for cls, c_vec in centroids.items():
            if cls == lbl:
                continue
            d_c = float(1.0 - np.dot(feats_norm[i], c_vec))
            if d_c < best_other_centroid_dist:
                best_other_centroid_dist = d_c
                best_other_cls = cls

        if nearest_lbl == lbl:
            consistent_count += 1

        # Flag if both LOO 1-NN and centroid agree on a DIFFERENT class, or own distance is extreme
        is_suspect = (
            nearest_lbl != lbl and best_other_centroid_dist + 0.03 < own_centroid_dist
        ) or (own_centroid_dist > 0.45 and nearest_lbl != lbl)

        if is_suspect:
            suspect_samples.append(
                {
                    "crop_id": meta.crop_id,
                    "split": meta.split,
                    "legacy_gallery_index": meta.legacy_gallery_index,
                    "assigned_label": lbl,
                    "suggested_label": best_other_cls if best_other_centroid_dist < own_centroid_dist else nearest_lbl,
                    "loo_nearest_label": nearest_lbl,
                    "loo_nearest_crop_id": nearest_crop_id,
                    "loo_nearest_distance": round(float(nearest_dist), 4),
                    "own_centroid_distance": round(own_centroid_dist, 4),
                    "closest_other_centroid_distance": round(best_other_centroid_dist, 4),
                }
            )

    loo_consistency = consistent_count / float(max(1, checked_count))
    return {
        "feature_method": feature_method,
        "total_checked": checked_count,
        "loo_consistency": round(loo_consistency, 4),
        "suspect_count": len(suspect_samples),
        "suspect_samples": suspect_samples,
    }

