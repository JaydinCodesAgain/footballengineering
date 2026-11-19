"""Reproducible Sprint 3.1 Stage B benchmark: Unsupervised Team Discovery on verified matches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from src.classification.dataset import JerseyCropDataset
from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    collect_track_appearances,
    evaluate_discovered_labels,
)
from src.config import TeamDiscoveryConfig, load_match_config
from src.evaluation.harness import evaluate_team_discovery_on_verified_crops
from src.evaluation.report import write_csv_rows
from src.visualization.plots import save_side_by_side_confusion_matrices


def main() -> None:
    out_dir = Path("outputs/sprint3_1/team_discovery")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("norway_wales", "configs/norway_wales.yaml"),
        ("chelsea_sunderland", "configs/chelsea_sunderland.yaml"),
    ]
    features = [
        "baseline_hsv",
        "hsv_lab",
        "foreground_masked",
        "dominant_colour",
        "spatial_embedding",
        "hybrid_hsv_lab_fg",
    ]
    clusterers = ["kmeans", "agglomerative", "gmm", "hdbscan", "dbscan"]
    seeds = (0, 7, 21, 42, 99)

    csv_rows: List[Dict[str, Any]] = []
    full_report: Dict[str, Any] = {
        "protocol": (
            "Unsupervised per-match clustering evaluated strictly post-hoc via Team_A/Team_B "
            "permutation alignment against independently human-verified jersey labels. "
            "No ground-truth labels are ever exposed during fitting."
        ),
        "selected_configuration": {
            "feature_method": "hsv_lab",
            "clustering_method": "kmeans",
            "auto_select_k": True,
            "max_candidate_clusters": 4,
            "radius_mad_multiplier": 2.6,
            "min_team_margin": 0.03,
            "max_distance_ratio": 0.85,
            "outlier_repulsion_ratio": 1.05,
        },
        "mobilenet_v3_small_ablation_note": (
            "Benchmarked separately in Stage B ablation: torchvision MobileNetV3-Small (576-d, ~81 ms/crop CPU) "
            "achieved ARI=0.099 (f1_3way=0.529) on Norway vs Wales and ARI=-0.002 (f1_3way=0.383) on "
            "Chelsea vs Sunderland because ImageNet representations prioritize pose/shape invariance over "
            "kit colour. Rejected on both accuracy and ~100x latency cost."
        ),
        "matches": {},
    }

    cm_panels = []

    for match_id, cfg_path in datasets:
        cfg = load_match_config(cfg_path)
        crops_dir = Path("data/crops") / match_id
        match_results: Dict[str, Any] = {}

        # Measure feature latency once per feature
        ds = JerseyCropDataset(crops_dir, cfg)
        samples = ds.get_labelled_samples(exclude_unknown=False, verified_only=True)
        crops_only = [c for _, c, _ in samples]

        for feat in features:
            from src.classification.features import get_feature_extractor

            fn = get_feature_extractor(feat)
            t0 = time.perf_counter()
            for c in crops_only:
                fn(c)
            ms_per_crop = round((time.perf_counter() - t0) / max(1, len(crops_only)) * 1000.0, 2)

            for cl in clusterers:
                tcfg = TeamDiscoveryConfig(
                    enabled=True,
                    feature_method=feat,
                    clustering_method=cl,
                    candidate_clusters=3,
                    auto_select_k=True,
                    max_candidate_clusters=4,
                    random_state=42,
                )
                res = evaluate_team_discovery_on_verified_crops(
                    match_config=cfg,
                    crops_root=crops_dir,
                    discovery_config=tcfg,
                    seeds=seeds,
                )
                ev = res["evaluation"]
                ss = res["seed_stability"]
                ts = res["temporal_subset_stability"]
                h1 = ts.get("first_half", {})
                h2 = ts.get("second_half", {})

                row = {
                    "match_id": match_id,
                    "n_verified_samples": ev["total_samples"],
                    "feature_method": feat,
                    "clustering_method": cl,
                    "ms_per_crop": ms_per_crop,
                    "selected_k": res["discovery_report"]["selected_k"],
                    "inter_team_cosine_distance": res["discovery_report"]["inter_team_cosine_distance"],
                    "silhouette_score": res["discovery_report"]["silhouette_score"],
                    "classification_coverage": ev["classification_coverage"],
                    "team_coverage": ev["team_coverage"],
                    "unknown_prediction_rate": ev["unknown_prediction_rate"],
                    "clustering_accuracy_on_covered": ev["clustering_accuracy_on_covered"],
                    "macro_f1_on_covered": ev["macro_f1_on_covered"],
                    "team_accuracy_including_rejected": ev["team_accuracy_including_rejected"],
                    "overall_accuracy_including_rejected": ev["overall_accuracy_including_rejected"],
                    "macro_f1_including_unknown": ev["macro_f1_including_unknown"],
                    "team_a_precision_covered": ev["per_team_on_covered"]["Team_A"]["precision"],
                    "team_a_recall_covered": ev["per_team_on_covered"]["Team_A"]["recall"],
                    "team_b_precision_covered": ev["per_team_on_covered"]["Team_B"]["precision"],
                    "team_b_recall_covered": ev["per_team_on_covered"]["Team_B"]["recall"],
                    "referee_misclassification_rate": ev["referee_misclassification_rate"],
                    "unknown_misclassification_rate": ev["unknown_misclassification_rate"],
                    "adjusted_rand_index": ev["adjusted_rand_index"],
                    "adjusted_rand_index_3way": ev["adjusted_rand_index_3way"],
                    "normalized_mutual_information": ev["normalized_mutual_information"],
                    "seed_macro_f1_covered_mean": ss["macro_f1_on_covered_mean"],
                    "seed_macro_f1_covered_std": ss["macro_f1_on_covered_std"],
                    "seed_macro_f1_3way_mean": ss["macro_f1_3way_mean"],
                    "seed_macro_f1_3way_std": ss["macro_f1_3way_std"],
                    "half1_macro_f1_3way": h1.get("macro_f1_including_unknown"),
                    "half2_macro_f1_3way": h2.get("macro_f1_including_unknown"),
                }
                csv_rows.append(row)
                match_results[f"{feat}__{cl}"] = res

                if feat == "hsv_lab" and cl == "kmeans":
                    cm_panels.append(
                        (
                            f"{cfg.match_name} (n={ev['total_samples']})\n"
                            f"hsv_lab + KMeans | Cov F1={ev['macro_f1_on_covered']:.1%}, 3-way F1={ev['macro_f1_including_unknown']:.1%}",
                            ev["confusion_matrix_3way"],
                            ["Team_A", "Team_B", "Unknown/Ref"],
                        )
                    )

        full_report["matches"][match_id] = {
            "selected_hsv_lab_kmeans": match_results["hsv_lab__kmeans"],
            "spatial_embedding_kmeans": match_results["spatial_embedding__kmeans"],
            "hybrid_hsv_lab_fg_kmeans": match_results["hybrid_hsv_lab_fg__kmeans"],
            "hsv_lab_gmm": match_results["hsv_lab__gmm"],
            "hsv_lab_agglomerative": match_results["hsv_lab__agglomerative"],
            "hsv_lab_hdbscan": match_results["hsv_lab__hdbscan"],
        }

    write_csv_rows(out_dir / "team_metrics.csv", csv_rows)
    (out_dir / "clustering_report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    save_side_by_side_confusion_matrices(cm_panels, out_dir / "team_confusion_matrix.png")

    print("Saved Stage B team discovery deliverables to:", out_dir)
    for mid in ("norway_wales", "chelsea_sunderland"):
        sel = full_report["matches"][mid]["selected_hsv_lab_kmeans"]
        ev = sel["evaluation"]
        ss = sel["seed_stability"]
        print(
            f"  {mid}: f1_cov={ev['macro_f1_on_covered']:.4f} f1_3way={ev['macro_f1_including_unknown']:.4f} "
            f"acc_cov={ev['clustering_accuracy_on_covered']:.4f} acc_all={ev['overall_accuracy_including_rejected']:.4f} "
            f"team_cov={ev['team_coverage']:.4f} ref_mis={ev['referee_misclassification_rate']:.4f} "
            f"unk_mis={ev['unknown_misclassification_rate']:.4f} ARI3={ev['adjusted_rand_index_3way']:.4f} "
            f"seed_std={ss['macro_f1_3way_std']:.4f}"
        )


if __name__ == "__main__":
    main()

