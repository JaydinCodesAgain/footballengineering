"""Consolidated analytics report generator (JSON, CSV, and Markdown)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.schemas import FrameObservationRecord


def write_csv_rows(path: str | Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """Write a sequence of dictionaries to a CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target

    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return target


def export_possession_foundation_records(
    path: str | Path,
    records: Sequence[FrameObservationRecord],
) -> Path:
    """
    Export per-frame player/team/ball observation records to CSV for downstream possession analytics.
    """
    dict_rows = [r.to_dict() for r in records]
    return write_csv_rows(path, dict_rows)


def generate_consolidated_report(
    output_dir: str | Path,
    match_id: str,
    match_name: str,
    detection_summary: Dict[str, Any],
    classification_summary: Dict[str, Any],
    tracking_summary: Dict[str, Any],
    tracker_comparison: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_benchmark: Optional[Sequence[Dict[str, Any]]] = None,
    label_audit: Optional[Dict[str, Any]] = None,
    artifacts_generated: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """
    Produce consolidated JSON, CSV, and Markdown summary reports inside `output_dir`.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    consolidated = {
        "generated_at": timestamp_iso,
        "match_id": match_id,
        "match_name": match_name,
        "detection": detection_summary,
        "classification": classification_summary,
        "tracking": tracking_summary,
        "tracker_comparison": tracker_comparison or {},
        "label_audit": label_audit or {},
        "artifacts": artifacts_generated or {},
    }

    json_file = out_path / "analytics_report.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    saved_files: Dict[str, Path] = {"report_json": json_file}

    # Export classification benchmark CSV if present
    if feature_benchmark:
        bench_rows = []
        for item in feature_benchmark:
            bench_rows.append(
                {
                    "experiment_tag": item.get("experiment_tag"),
                    "feature_method": item.get("feature_method"),
                    "classifier_type": item.get("classifier_type"),
                    "n_neighbors": item.get("n_neighbors"),
                    "calibrated": item.get("calibrated"),
                    "unknown_threshold": item.get("unknown_threshold"),
                    "overall_accuracy": item.get("overall_accuracy"),
                    "selective_accuracy": item.get("selective_accuracy"),
                    "raw_known_accuracy_no_rejection": item.get("raw_known_accuracy_no_rejection"),
                    "macro_f1": item.get("macro_f1"),
                    "positive_macro_f1": item.get("positive_macro_f1"),
                    "coverage": item.get("coverage"),
                    "unknown_rejection_rate": item.get("unknown_rejection_rate"),
                    "misclassified_count": item.get("misclassified_count"),
                }
            )
        saved_files["classification_benchmark_csv"] = write_csv_rows(
            out_path / "classification_benchmark.csv", bench_rows
        )

    # Export tracker comparison CSV if present
    if tracker_comparison:
        trk_rows = list(tracker_comparison.values())
        saved_files["tracker_comparison_csv"] = write_csv_rows(
            out_path / "tracker_comparison.csv", trk_rows
        )

    # Export Markdown summary
    md_lines = [
        f"# Football AI Analytics Report — {match_name} (`{match_id}`)",
        f"**Generated:** {timestamp_iso}",
        "",
        "## 1. Detection Summary",
        f"- **Total Frames Processed:** {detection_summary.get('total_frames_processed', 0)}",
        f"- **Average Players / Frame:** {detection_summary.get('avg_players_per_frame', 0.0):.2f}",
        f"- **Frames with Ball Detected:** {detection_summary.get('frames_with_ball', 0)} "
        f"({detection_summary.get('ball_coverage_pct', 0.0):.1f}%)",
    ]

    iou_eval = detection_summary.get("iou_evaluation", {})
    if iou_eval and iou_eval.get("metrics_available"):
        p_m = iou_eval["player"]
        b_m = iou_eval["ball"]
        md_lines.extend(
            [
                f"- **Verified Bounding-Box IoU Evaluation (Frames={iou_eval.get('evaluated_frames')}):**",
                f"  - **Player:** Precision={p_m['precision']:.2%}, Recall={p_m['recall']:.2%}, "
                f"F1={p_m['f1']:.2%}, AP@50={p_m['ap50']:.2%}",
                f"  - **Ball:** Precision={b_m['precision']:.2%}, Recall={b_m['recall']:.2%}, "
                f"F1={b_m['f1']:.2%}, AP@50={b_m['ap50']:.2%}",
            ]
        )
    else:
        md_lines.append(
            "- **Bounding-Box IoU Evaluation:** Unavailable (awaiting manually verified ground-truth bounding boxes)."
        )

    md_lines.extend(
        [
            "",
            "## 2. Team Classification Summary",
            f"- **Feature Method:** `{classification_summary.get('feature_method', 'N/A')}`",
            f"- **Classifier:** `{classification_summary.get('classifier_type', 'N/A')}` "
            f"(threshold={classification_summary.get('unknown_threshold', 'N/A')})",
            f"- **Training Dataset Size:** {classification_summary.get('training_size', 0)} "
            f"(`{classification_summary.get('training_class_counts', {})}`)",
            f"- **Held-Out Evaluation Accuracy:** {classification_summary.get('overall_accuracy', 0.0):.2%}",
            f"- **Selective Accuracy (Non-Unknown):** {classification_summary.get('selective_accuracy', 0.0):.2%}",
            f"- **Macro F1 (All Classes):** {classification_summary.get('macro_f1', 0.0):.2%}",
            f"- **Macro F1 (Positive Classes):** {classification_summary.get('positive_macro_f1', 0.0):.2%}",
            f"- **Classification Coverage:** {classification_summary.get('coverage', 0.0):.2%} "
            f"(Unknown Rate: {classification_summary.get('unknown_rejection_rate', 0.0):.2%})",
            "",
            "### Per-Class Metrics",
            "| Internal Label | Display Name | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|---|",
        ]
    )

    for int_lbl, c_stats in classification_summary.get("per_class", {}).items():
        md_lines.append(
            f"| `{int_lbl}` | {c_stats.get('display_name', int_lbl)} | "
            f"{c_stats.get('precision', 0.0):.2%} | {c_stats.get('recall', 0.0):.2%} | "
            f"{c_stats.get('f1', 0.0):.2%} | {c_stats.get('support', 0)} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Tracking & Temporal Stability Summary",
            f"- **Tracker Config:** `{tracking_summary.get('tracker_config', 'bytetrack.yaml')}`",
            f"- **Distinct Tracking IDs:** {tracking_summary.get('distinct_tracking_ids', 0)}",
            f"- **Tracks < 5 Observations:** {tracking_summary.get('short_tracks_under_5_obs', 0)}",
            f"- **Mean Track Lifetime (obs):** {tracking_summary.get('mean_track_observations', 0.0)}",
            f"- **Total Player Observations:** {tracking_summary.get('total_player_observations', 0)}",
            f"- **Raw Team-Label Changes:** {tracking_summary.get('raw_team_label_changes', 0)}",
            f"- **Smoothed Team-Label Changes:** {tracking_summary.get('smoothed_team_label_changes', 0)}",
            f"- **Stable-Label Coverage:** {tracking_summary.get('stable_label_coverage', 0.0):.2%}",
            f"- **Identity Resets Triggered:** {tracking_summary.get('identity_resets_triggered', 0)}",
        ]
    )

    md_file = out_path / "analytics_report.md"
    md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    saved_files["report_md"] = md_file
    return saved_files

