"""Possession state annotation workflow and quantitative validation metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import MatchConfig
from src.schemas import InternalLabel, PossessionFrameRecord, PossessionState


@dataclass
class PossessionIntervalAnnotation:
    """Human-annotated time/frame interval for ball-control validation."""

    match_id: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    state: str  # One of PossessionState.all_states()
    uncertain: bool = False
    verified: bool = False
    source: str = "template"  # Must be "manual" with verified=True for independent evaluation
    notes: str = ""

    def validate(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) < start_frame ({self.start_frame})")
        if self.state and self.state not in PossessionState.all_states():
            raise ValueError(
                f"Invalid possession state '{self.state}'. Expected one of {PossessionState.all_states()}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PossessionAnnotationStore:
    """
    Manages persistent human-labelled possession intervals in CSV and JSON format.
    Enforces strict provenance so unverified templates or model predictions are never
    evaluated as independent ground truth.
    """

    def __init__(self, root_dir: str | Path, match_id: str) -> None:
        self.root_dir = Path(root_dir)
        self.match_id = match_id
        self.csv_path = self.root_dir / "possession_annotations.csv"
        self.json_path = self.root_dir / "possession_annotations.json"
        self.intervals: List[PossessionIntervalAnnotation] = []
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.json_path.is_file():
            self.load()

    def create_template_intervals(
        self,
        start_frame: int,
        end_frame: int,
        fps: float = 25.0,
        interval_frames: int = 30,
    ) -> List[PossessionIntervalAnnotation]:
        """
        Generate unverified template intervals in `possession_annotations.csv` so a human
        reviewer can label ball-control states across a video segment without writing code.
        """
        if self.intervals:
            return self.intervals

        fps_val = max(1.0, float(fps))
        cur = int(start_frame)
        while cur <= end_frame:
            nxt = min(int(end_frame), cur + interval_frames - 1)
            self.intervals.append(
                PossessionIntervalAnnotation(
                    match_id=self.match_id,
                    start_frame=cur,
                    end_frame=nxt,
                    start_timestamp=round(cur / fps_val, 3),
                    end_timestamp=round(nxt / fps_val, 3),
                    state="",
                    uncertain=False,
                    verified=False,
                    source="template",
                    notes="Fill state with TEAM_A_CONTROL, TEAM_B_CONTROL, CONTESTED, LOOSE_BALL, UNKNOWN, or OUT_OF_PLAY; set verified=true and source=manual",
                )
            )
            cur = nxt + 1

        self.save()
        return self.intervals

    def set_manual_intervals(
        self,
        intervals: Sequence[PossessionIntervalAnnotation],
    ) -> None:
        """Register validated manual possession intervals."""
        validated: List[PossessionIntervalAnnotation] = []
        for iv in intervals:
            if iv.source.lower() in ("model", "predicted", "auto", "engine"):
                raise ValueError(
                    f"Cannot register ground-truth possession interval with source='{iv.source}'."
                )
            iv.validate()
            validated.append(iv)
        self.intervals = validated
        self.save()

    def save(self) -> None:
        verified = [iv for iv in self.intervals if iv.verified and iv.source == "manual" and iv.state]
        payload = {
            "match_id": self.match_id,
            "total_intervals": len(self.intervals),
            "verified_manual_intervals": len(verified),
            "evaluation_ready": len(verified) > 0,
            "intervals": [iv.to_dict() for iv in self.intervals],
        }
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        fieldnames = [
            "match_id",
            "start_frame",
            "end_frame",
            "start_timestamp",
            "end_timestamp",
            "state",
            "uncertain",
            "verified",
            "source",
            "notes",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for iv in self.intervals:
                writer.writerow(iv.to_dict())

    def load(self) -> None:
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.intervals = [
            PossessionIntervalAnnotation(**item) for item in data.get("intervals", [])
        ]

    def import_from_csv(self, csv_path: Optional[str | Path] = None) -> int:
        target = Path(csv_path) if csv_path else self.csv_path
        if not target.is_file():
            raise FileNotFoundError(f"Possession annotation CSV not found: {target}")

        loaded: List[PossessionIntervalAnnotation] = []
        with target.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source", "manual").strip() or "manual"
                if src.lower() in ("model", "predicted", "auto", "engine"):
                    raise ValueError(
                        f"Possession annotations cannot be imported with source='{src}'."
                    )
                ver = row.get("verified", "false").strip().lower() in ("true", "1", "yes", "y")
                unc = row.get("uncertain", "false").strip().lower() in ("true", "1", "yes", "y")
                state_str = row.get("state", "").strip()
                iv = PossessionIntervalAnnotation(
                    match_id=row.get("match_id", self.match_id).strip() or self.match_id,
                    start_frame=int(float(row["start_frame"])),
                    end_frame=int(float(row["end_frame"])),
                    start_timestamp=float(row.get("start_timestamp", 0.0) or 0.0),
                    end_timestamp=float(row.get("end_timestamp", 0.0) or 0.0),
                    state=state_str,
                    uncertain=unc,
                    verified=bool(ver and src == "manual" and state_str != ""),
                    source=src,
                    notes=row.get("notes", "").strip(),
                )
                if state_str:
                    iv.validate()
                loaded.append(iv)

        self.intervals = loaded
        self.save()
        return sum(1 for iv in loaded if iv.verified)

    def get_verified_intervals(self) -> List[PossessionIntervalAnnotation]:
        return [
            iv
            for iv in self.intervals
            if iv.verified and iv.source == "manual" and iv.state in PossessionState.all_states()
        ]


def evaluate_possession_predictions(
    predicted_timeline: Sequence[PossessionFrameRecord],
    verified_intervals: Sequence[PossessionIntervalAnnotation],
    match_config: Optional[MatchConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate predicted per-frame possession states against manually verified ground-truth intervals.
    If no verified manual intervals exist, returns `metrics_available=False` and `status='pending_manual_annotations'`.
    """
    valid_ivs = [
        iv
        for iv in verified_intervals
        if iv.verified and iv.source == "manual" and iv.state in PossessionState.all_states()
    ]
    if not valid_ivs or not predicted_timeline:
        return {
            "metrics_available": False,
            "status": "pending_manual_possession_annotations",
            "message": (
                "No manually verified ground-truth possession intervals are present yet. "
                "Fill `data/annotations/<match_id>/possession_annotations.csv` with `verified=true` "
                "and `source=manual` to compute ground-truth possession accuracy."
            ),
        }

    # Map frame_number -> (true_state, uncertain)
    gt_by_frame: Dict[int, Tuple[str, bool]] = {}
    for iv in valid_ivs:
        for fno in range(iv.start_frame, iv.end_frame + 1):
            gt_by_frame[fno] = (iv.state, iv.uncertain)

    y_true: List[str] = []
    y_pred: List[str] = []
    dts: List[float] = []
    uncertain_frames = 0

    for rec in predicted_timeline:
        if rec.frame_number not in gt_by_frame:
            continue
        gt_state, is_unc = gt_by_frame[rec.frame_number]
        if is_unc:
            uncertain_frames += 1
            continue
        y_true.append(gt_state)
        y_pred.append(rec.possession_state)
        dts.append(max(0.0, float(rec.dt_seconds)))

    if not y_true:
        return {
            "metrics_available": False,
            "status": "no_overlapping_certain_frames",
            "uncertain_frames_skipped": uncertain_frames,
        }

    all_states = list(PossessionState.all_states())
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_states, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_states)

    per_state: Dict[str, Dict[str, Any]] = {}
    for i, st in enumerate(all_states):
        per_state[st] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support_frames": int(support[i]),
        }

    # Compute time-based team possession errors
    gt_a_sec = sum(dt for t, dt in zip(y_true, dts) if t == PossessionState.TEAM_A_CONTROL.value)
    gt_b_sec = sum(dt for t, dt in zip(y_true, dts) if t == PossessionState.TEAM_B_CONTROL.value)
    pred_a_sec = sum(dt for p, dt in zip(y_pred, dts) if p == PossessionState.TEAM_A_CONTROL.value)
    pred_b_sec = sum(dt for p, dt in zip(y_pred, dts) if p == PossessionState.TEAM_B_CONTROL.value)

    gt_attr = gt_a_sec + gt_b_sec
    pred_attr = pred_a_sec + pred_b_sec
    gt_a_share = (gt_a_sec / gt_attr * 100.0) if gt_attr > 0 else 50.0
    pred_a_share = (pred_a_sec / pred_attr * 100.0) if pred_attr > 0 else 50.0
    share_error_pct_pts = abs(pred_a_share - gt_a_share)

    # State transition counts
    gt_transitions = sum(1 for a, b in zip(y_true, y_true[1:]) if a != b)
    pred_transitions = sum(1 for a, b in zip(y_pred, y_pred[1:]) if a != b)
    transition_error = abs(pred_transitions - gt_transitions)

    total_eval_sec = sum(dts)
    confident_pred_sec = sum(
        dt
        for p, dt in zip(y_pred, dts)
        if p in (PossessionState.TEAM_A_CONTROL.value, PossessionState.TEAM_B_CONTROL.value)
    )
    confident_coverage_pct = (
        (confident_pred_sec / total_eval_sec * 100.0) if total_eval_sec > 0 else 0.0
    )

    return {
        "metrics_available": True,
        "status": "verified_manual_possession_evaluation",
        "evaluated_frames": len(y_true),
        "uncertain_frames_excluded": uncertain_frames,
        "state_accuracy": round(acc, 4),
        "per_state": per_state,
        "team_a_time_error_seconds": round(abs(pred_a_sec - gt_a_sec), 2),
        "team_b_time_error_seconds": round(abs(pred_b_sec - gt_b_sec), 2),
        "team_possession_share_error_pct_points": round(share_error_pct_pts, 2),
        "gt_team_a_share_pct": round(gt_a_share, 2),
        "pred_team_a_share_pct": round(pred_a_share, 2),
        "gt_transitions": gt_transitions,
        "pred_transitions": pred_transitions,
        "transition_count_error": transition_error,
        "confidently_assigned_possession_pct": round(confident_coverage_pct, 2),
        "state_labels": all_states,
        "confusion_matrix": cm.tolist(),
    }

