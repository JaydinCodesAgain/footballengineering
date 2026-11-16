"""Matplotlib diagnostic plots: confusion matrices, galleries, calibration curves, and timelines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.schemas import JerseyCropMetadata


def save_confusion_matrix_plot(
    confusion_mat: Sequence[Sequence[int]] | np.ndarray,
    display_labels: Sequence[str],
    title: str,
    output_path: str | Path,
) -> Path:
    """Render and save a publication-quality confusion matrix image."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(confusion_mat, dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(display_labels))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target


def save_side_by_side_confusion_matrices(
    matrices: Sequence[Tuple[str, Sequence[Sequence[int]] | np.ndarray, Sequence[str]]],
    output_path: str | Path,
) -> Path:
    """Save multiple confusion matrices side-by-side for baseline vs improved comparison."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    n = max(1, len(matrices))
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0))
    if n == 1:
        axes = [axes]

    for ax, (title, cm, labels) in zip(axes, matrices):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(cm, dtype=int),
            display_labels=list(labels),
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target


def save_misclassification_gallery(
    eval_samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    misclassified_records: Sequence[Dict[str, Any]],
    output_path: str | Path,
    max_items: int = 20,
) -> Path:
    """Render an image grid of misclassified crops with true, predicted, and distance details."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    crop_lookup = {meta.crop_id: img for meta, img, _ in eval_samples}
    items = list(misclassified_records)[:max_items]

    if not items:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(
            0.5,
            0.5,
            "Zero Misclassifications on Evaluation Split!",
            ha="center",
            va="center",
            fontsize=13,
            color="green",
            fontweight="bold",
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(str(target), dpi=150)
        plt.close(fig)
        return target

    cols = min(5, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    flat_axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in flat_axes:
        ax.axis("off")

    for ax, rec in zip(flat_axes, items):
        cid = rec["crop_id"]
        img = crop_lookup.get(cid)
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        idx_str = f"#{rec['legacy_gallery_index']} " if rec.get("legacy_gallery_index") is not None else ""
        ax.set_title(
            f"{idx_str}({cid[-13:]})\nTrue: {rec['true_display']} | Pred: {rec['pred_display']}\n"
            f"d={rec['nearest_distance']:.2f} (raw:{rec['raw_nearest_display']})",
            fontsize=8.5,
            color="darkred",
        )

    fig.suptitle("Misclassified Jersey Crops Gallery", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target


def save_crop_reference_gallery(
    samples: Sequence[Tuple[JerseyCropMetadata, np.ndarray, str]],
    title: str,
    output_path: str | Path,
    columns: int = 8,
) -> Path:
    """Save a contact-sheet gallery of crops with their stable ID, gallery index, and label."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not samples:
        return target

    rows = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.4 * rows))
    flat_axes = np.array(axes).flatten()

    for ax in flat_axes:
        ax.axis("off")

    for ax, (meta, img, lbl) in zip(flat_axes, samples):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        idx_tag = f"#{meta.legacy_gallery_index} " if meta.legacy_gallery_index is not None else ""
        disp = meta.raw_label or lbl
        ax.set_title(f"{idx_tag}{disp}\nf{meta.frame_number}", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(target), dpi=150)
    plt.close(fig)
    return target


def save_threshold_calibration_plot(
    calibration_data: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot coverage vs selective accuracy, overall accuracy, and macro F1 across distance thresholds."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    curve = calibration_data.get("curve", [])
    if not curve:
        return target

    thresholds = [r["threshold"] for r in curve]
    overall_acc = [r["overall_accuracy"] for r in curve]
    sel_acc = [r["selective_accuracy"] for r in curve]
    coverage = [r["coverage"] for r in curve]
    macro_f1 = [r["macro_f1"] for r in curve]
    selected_t = calibration_data.get("selected_threshold")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(thresholds, overall_acc, label="Overall Accuracy", linewidth=2)
    ax.plot(thresholds, sel_acc, label="Selective Accuracy (Non-Unknown)", linestyle="--", linewidth=1.8)
    ax.plot(thresholds, macro_f1, label="Macro F1", linewidth=2)
    ax.plot(thresholds, coverage, label="Classification Coverage", linestyle=":", linewidth=2)

    if selected_t is not None:
        ax.axvline(
            float(selected_t),
            color="red",
            linestyle="-.",
            label=f"Selected Threshold ({float(selected_t):.3f})",
        )

    ax.set_xlabel("Nearest-Neighbor Cosine Distance Threshold")
    ax.set_ylabel("Score / Rate")
    ax.set_title("Unknown Rejection Threshold Calibration: Coverage vs Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target


def save_detections_timeline_plot(
    people_per_frame: Sequence[int],
    balls_per_frame: Sequence[int],
    output_fps: float,
    output_path: str | Path,
) -> Path:
    """Plot player and ball detection counts over time."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    seconds = np.arange(len(people_per_frame)) / max(1.0, float(output_fps))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(seconds, people_per_frame, label="Players detected", linewidth=1.6)
    ax.plot(seconds, balls_per_frame, label="Footballs detected", linewidth=1.6)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Detections per frame")
    ax.set_title("Detections Over Time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(target), dpi=150)
    plt.close(fig)
    return target


def save_possession_timeline_chart(
    timeline: Sequence[Any],
    match_config: Any,
    possession_summary: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Render a two-panel possession analytics chart:
    - Top panel: Cumulative attributed possession share (%) for Team A and Team B over time,
      plus a pie/bar summary of total analysed time distribution across all 6 states.
    - Bottom panel: Frame-by-frame possession state strip and ball tracking status strip.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not timeline:
        return target

    times = [float(r.timestamp) for r in timeline]
    share_a = [float(r.running_team_a_share_pct) for r in timeline]
    share_b = [float(r.running_team_b_share_pct) for r in timeline]

    team_a_name = match_config.to_display_label("Team_A")
    team_b_name = match_config.to_display_label("Team_B")

    state_order = [
        "TEAM_A_CONTROL",
        "TEAM_B_CONTROL",
        "CONTESTED",
        "LOOSE_BALL",
        "UNKNOWN",
        "OUT_OF_PLAY",
    ]
    state_labels = [
        f"{team_a_name} Control",
        f"{team_b_name} Control",
        "Contested",
        "Loose Ball",
        "Unknown",
        "Out of Play",
    ]
    state_to_idx = {st: idx for idx, st in enumerate(state_order)}
    state_indices = [state_to_idx.get(r.possession_state, 4) for r in timeline]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]}
    )

    # Panel 1: Attributed possession share curve
    ax1.plot(times, share_a, label=f"{team_a_name} Attributed Share (%)", linewidth=2.2, color="#d62728")
    ax1.plot(times, share_b, label=f"{team_b_name} Attributed Share (%)", linewidth=2.2, color="#1f77b4")
    ax1.axhline(50.0, color="grey", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Attributed Share (%)")
    scope_tag = possession_summary.get("segment_scope", "segment")
    dur_s = possession_summary.get("analysed_duration_seconds", 0.0)
    ax1.set_title(
        f"{match_config.match_name} — Estimated Possession Timeline ({dur_s:.1f}s {scope_tag})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    # Panel 2: Discrete state timeline
    ax2.step(times, state_indices, where="post", linewidth=1.8, color="#2ca02c")
    ax2.set_yticks(range(len(state_labels)))
    ax2.set_yticklabels(state_labels, fontsize=9)
    ax2.set_xlabel("Video Timestamp (seconds)")
    ax2.set_title("Frame-by-Frame Possession State Machine Output", fontsize=10.5, fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.invert_yaxis()

    # Panel 3: Total analysed time distribution bar chart
    dist_pct = possession_summary.get("total_analysed_time_distribution_pct", {})
    vals = [float(dist_pct.get(st, 0.0)) for st in state_order]
    bar_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f", "#9467bd"]
    bars = ax3.barh(state_labels, vals, color=bar_colors)
    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Percentage of Total Analysed Segment Duration (%)")
    ax3.set_title("Total Analysed Time Distribution (All 6 States)", fontsize=10.5, fontweight="bold")
    ax3.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax3.text(min(98.0, v + 1.2), bar.get_y() + bar.get_height() / 2.0, f"{v:.1f}%", va="center", fontsize=8.5)
    ax3.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target


def save_ball_detection_diagnostic_gallery(
    diagnostic_items: Sequence[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """
    Render a diagnostic gallery illustrating football detection successes and failure modes:
    - High-confidence detections
    - Low-confidence / high-res / tiled recoveries
    - Rejected false positives (socks, torso logos, scoreboard)
    - Missed / motion-blurred / occluded balls
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not diagnostic_items:
        return target

    items = list(diagnostic_items)[:12]
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
    flat_axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in flat_axes:
        ax.axis("off")

    for ax, item in zip(flat_axes, items):
        img = item.get("crop_or_frame_bgr")
        if img is not None and img.size > 0:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cat = item.get("category", "Diagnostic")
        fno = item.get("frame_number", 0)
        note = item.get("note", "")
        color = item.get("title_color", "navy")
        ax.set_title(f"[{cat}] Frame {fno}\n{note}", fontsize=8.5, color=color, fontweight="bold")

    fig.suptitle(
        "Football Detection Failure-Mode & Recovery Diagnostic Gallery",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(target), dpi=160)
    plt.close(fig)
    return target

