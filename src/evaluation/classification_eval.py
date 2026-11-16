"""Comprehensive evaluation and model comparison for jersey team classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.classification.classifier import TeamClassifier
from src.config import ClassificationConfig, MatchConfig
from src.schemas import ClassificationPrediction, InternalLabel, JerseyCropMetadata


def evaluate_team_classifier(
    classifier: TeamClassifier,
    eval_samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    match_config: Optional[MatchConfig] = None,
    allow_cross_match: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained `TeamClassifier` on held-out `(metadata, crop_bgr, true_internal_label)` samples.
    """
    if not eval_samples:
        raise ValueError("eval_samples cannot be empty.")

    target_match_id = match_config.match_id if match_config else eval_samples[0][0].match_id
    if target_match_id != classifier.match_id and not allow_cross_match:
        raise ValueError(
            f"Incompatible match evaluation: classifier trained on '{classifier.match_id}' "
            f"cannot be evaluated on '{target_match_id}' without allow_cross_match=True."
        )

    y_true: List[str] = []
    y_pred: List[str] = []
    y_raw_nearest: List[str] = []
    distances: List[float] = []
    predictions_detail: List[Dict[str, Any]] = []
    misclassified: List[Dict[str, Any]] = []

    for meta, crop_bgr, true_lbl in eval_samples:
        pred: ClassificationPrediction = classifier.predict_crop(
            crop_bgr, match_id=target_match_id, allow_cross_match=allow_cross_match
        )
        y_true.append(true_lbl)
        y_pred.append(pred.predicted_label)
        y_raw_nearest.append(pred.raw_nearest_label)
        distances.append(pred.nearest_distance)

        disp_true = match_config.to_display_label(true_lbl) if match_config else true_lbl
        disp_pred = (
            match_config.to_display_label(pred.predicted_label)
            if match_config
            else pred.predicted_label
        )
        disp_raw = (
            match_config.to_display_label(pred.raw_nearest_label)
            if match_config
            else pred.raw_nearest_label
        )

        rec_dict = {
            "crop_id": meta.crop_id,
            "legacy_gallery_index": meta.legacy_gallery_index,
            "frame_number": meta.frame_number,
            "timestamp": meta.timestamp,
            "true_internal": true_lbl,
            "pred_internal": pred.predicted_label,
            "raw_nearest_internal": pred.raw_nearest_label,
            "true_display": disp_true,
            "pred_display": disp_pred,
            "raw_nearest_display": disp_raw,
            "nearest_distance": round(pred.nearest_distance, 4),
            "second_nearest_distance": round(pred.second_nearest_distance, 4),
            "margin": round(pred.margin, 4),
            "confidence_score": round(pred.confidence_score, 4),
            "is_unknown": pred.is_unknown,
            "rejection_reason": pred.rejection_reason,
            "correct": true_lbl == pred.predicted_label,
            "image_path": meta.image_path,
        }
        predictions_detail.append(rec_dict)
        if true_lbl != pred.predicted_label:
            misclassified.append(rec_dict)

    label_order = list(InternalLabel.all_labels())
    display_label_order = [
        match_config.to_display_label(lbl) if match_config else lbl for lbl in label_order
    ]

    overall_accuracy = float(accuracy_score(y_true, y_pred))
    raw_unthresholded_accuracy = float(
        accuracy_score(
            [t for t in y_true if t != InternalLabel.UNKNOWN.value],
            [r for t, r in zip(y_true, y_raw_nearest) if t != InternalLabel.UNKNOWN.value],
        )
    ) if any(t != InternalLabel.UNKNOWN.value for t in y_true) else 0.0

    accepted_indices = [i for i, p in enumerate(y_pred) if p != InternalLabel.UNKNOWN.value]
    coverage = len(accepted_indices) / float(len(y_pred))
    unknown_rate = 1.0 - coverage

    if accepted_indices:
        selective_accuracy = float(
            accuracy_score(
                [y_true[i] for i in accepted_indices],
                [y_pred[i] for i in accepted_indices],
            )
        )
    else:
        selective_accuracy = 0.0

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, internal_lbl in enumerate(label_order):
        disp_lbl = display_label_order[idx]
        per_class[internal_lbl] = {
            "display_name": disp_lbl,
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    macro_f1_all = float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0))
    positive_labels = list(InternalLabel.valid_training_labels())
    macro_f1_positive = float(
        f1_score(y_true, y_pred, labels=positive_labels, average="macro", zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    return {
        "match_id": target_match_id,
        "classifier_match_id": classifier.match_id,
        "feature_method": classifier.config.feature_method,
        "classifier_type": classifier.config.classifier_type,
        "n_neighbors": classifier.config.n_neighbors,
        "unknown_threshold": round(float(classifier.unknown_threshold), 4),
        "total_samples": len(eval_samples),
        "overall_accuracy": round(overall_accuracy, 4),
        "selective_accuracy": round(selective_accuracy, 4),
        "raw_known_accuracy_no_rejection": round(raw_unthresholded_accuracy, 4),
        "macro_f1": round(macro_f1_all, 4),
        "positive_macro_f1": round(macro_f1_positive, 4),
        "coverage": round(coverage, 4),
        "unknown_rejection_rate": round(unknown_rate, 4),
        "per_class": per_class,
        "labels_internal": label_order,
        "labels_display": display_label_order,
        "confusion_matrix": cm.tolist(),
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified,
        "predictions": predictions_detail,
    }


def run_classification_benchmark(
    train_samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    val_samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    match_config: MatchConfig,
    feature_methods: Sequence[str] = (
        "baseline_hsv",
        "legacy_enhanced",
        "hsv_lab",
        "foreground_masked",
        "dominant_colour",
        "spatial_embedding",
    ),
    classifier_specs: Sequence[Tuple[str, int, bool]] = (
        ("knn", 3, False),
        ("knn", 3, True),
        ("knn", 1, True),
        ("svm_linear", 3, True),
        ("svm_rbf", 3, True),
        ("logistic", 3, True),
        ("nearest_centroid", 3, True),
    ),
) -> List[Dict[str, Any]]:
    """
    Compare multiple feature extractors, classifiers, and threshold calibration settings
    fairly on the exact same training and validation splits.
    """
    train_crops = [img for _, img, _ in train_samples]
    train_labels = [lbl for _, _, lbl in train_samples]
    val_crops = [img for _, img, _ in val_samples]
    val_labels = [lbl for _, _, lbl in val_samples]

    results: List[Dict[str, Any]] = []

    for feat_name in feature_methods:
        # 1. Fixed legacy threshold (0.45) with unweighted KNN (to reproduce baseline vs enhanced)
        legacy_cfg = ClassificationConfig(
            feature_method=feat_name,
            classifier_type="knn",
            n_neighbors=3,
            metric="cosine",
            unknown_threshold=0.45,
            auto_calibrate_threshold=False,
            min_margin=0.0,
            balance_class_weights=False,
        )
        clf_fixed = TeamClassifier(match_id=match_config.match_id, config=legacy_cfg)
        clf_fixed.fit(train_crops, train_labels)
        eval_fixed = evaluate_team_classifier(clf_fixed, val_samples, match_config=match_config)
        eval_fixed["experiment_tag"] = f"{feat_name}__knn3_uncalibrated_t0.45"
        eval_fixed["calibrated"] = False
        results.append(eval_fixed)

        # 2. Calibrated runs across classifier specs
        for c_type, k_val, bal_weights in classifier_specs:
            cfg = ClassificationConfig(
                feature_method=feat_name,
                classifier_type=c_type,
                n_neighbors=k_val,
                metric="cosine",
                unknown_threshold=0.35,
                auto_calibrate_threshold=True,
                min_margin=0.015,
                balance_class_weights=bal_weights,
            )
            clf = TeamClassifier(match_id=match_config.match_id, config=cfg)
            clf.fit(train_crops, train_labels)
            cal_info = clf.calibrate_unknown_threshold(val_crops, val_labels)
            eval_res = evaluate_team_classifier(clf, val_samples, match_config=match_config)
            w_tag = "balanced" if bal_weights else "unweighted"
            eval_res["experiment_tag"] = f"{feat_name}__{c_type}_k{k_val}_{w_tag}_calibrated"
            eval_res["calibrated"] = True
            eval_res["calibration_summary"] = {
                "selected_threshold": cal_info["selected_threshold"],
                "known_distance_summary": cal_info["known_distance_summary"],
                "unknown_distance_summary": cal_info["unknown_distance_summary"],
            }
            results.append(eval_res)

    return results

