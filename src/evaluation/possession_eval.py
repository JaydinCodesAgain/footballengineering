"""Possession state annotation workflow and quantitative validation metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.config import MatchConfig
from src.schemas import InternalLabel, PossessionFrameRecord, PossessionState


def validate_no_overlapping_intervals(
    intervals: Sequence["PossessionIntervalAnnotation"],
) -> None:
    """
    Validate that no two annotated intervals overlap in frame range `[start_frame, end_frame]`.
    Raises `ValueError` if any two intervals overlap.
    """
    active = [iv for iv in intervals if iv.state]
    sorted_ivs = sorted(active, key=lambda x: (int(x.start_frame), int(x.end_frame)))
    for i in range(len(sorted_ivs) - 1):
        cur = sorted_ivs[i]
        nxt = sorted_ivs[i + 1]
        if int(nxt.start_frame) <= int(cur.end_frame):
            raise ValueError(
                f"Overlapping possession interval annotations detected: "
                f"[{cur.start_frame}, {cur.end_frame}] ({cur.state}) overlaps with "
                f"[{nxt.start_frame}, {nxt.end_frame}] ({nxt.state})."
            )


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
        """Register validated manual possession intervals, rejecting overlapping intervals."""
        validated: List[PossessionIntervalAnnotation] = []
        for iv in intervals:
            if iv.source.lower() in ("model", "predicted", "auto", "engine"):
                raise ValueError(
                    f"Cannot register ground-truth possession interval with source='{iv.source}'."
                )
            iv.validate()
            validated.append(iv)
        validate_no_overlapping_intervals(validated)
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

        validate_no_overlapping_intervals(loaded)
        self.intervals = loaded
        self.save()
        return sum(1 for iv in loaded if iv.verified)

    def get_verified_intervals(self) -> List[PossessionIntervalAnnotation]:
        return [
            iv
            for iv in self.intervals
            if iv.verified and iv.source == "manual" and iv.state in PossessionState.all_states()
        ]

    def generate_html_annotator(
        self,
        match_config: Optional[MatchConfig] = None,
        fps: float = 25.0,
    ) -> Path:
        """
        Generate an interactive HTML5 interval annotator (`possession_annotator.html`)
        with match-specific team labels, overlap prevention, uncertain flag, verification checkbox,
        and direct CSV/JSON export.
        """
        html_path = self.root_dir / "possession_annotator.html"
        team_a_name = (
            match_config.to_display_label(InternalLabel.TEAM_A.value)
            if match_config is not None
            else "TEAM_A"
        )
        team_b_name = (
            match_config.to_display_label(InternalLabel.TEAM_B.value)
            if match_config is not None
            else "TEAM_B"
        )
        intervals_json = json.dumps([iv.to_dict() for iv in self.intervals])
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Possession Interval Annotator — {self.match_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
  .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 16px; }}
  button, select, input {{ background: #334155; color: #f8fafc; border: 1px solid #475569; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
  button:hover {{ background: #475569; cursor: pointer; }}
  button.primary {{ background: #2563eb; border-color: #3b82f6; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ background: #0f172a; color: #94a3b8; }}
  .error {{ color: #f87171; font-weight: 600; margin-left: 12px; }}
  .ok {{ color: #4ade80; font-weight: 600; margin-left: 12px; }}
</style>
</head>
<body>
<div class="header">
  <h2>Possession Interval Ground-Truth Annotator — {self.match_id} ({team_a_name} vs {team_b_name})</h2>
  <div>
    <button onclick="addInterval()">+ Add Interval</button>
    <button onclick="markAllVerified()">Mark Selected Verified</button>
    <button class="primary" onclick="exportCSV()">Export possession_annotations.csv</button>
    <button onclick="exportJSON()">Export possession_annotations.json</button>
  </div>
</div>
<div class="controls">
  <span>States: <b>TEAM_A_CONTROL</b> ({team_a_name}) | <b>TEAM_B_CONTROL</b> ({team_b_name}) | <b>CONTESTED</b> | <b>LOOSE_BALL</b> | <b>UNKNOWN</b> | <b>OUT_OF_PLAY</b></span>
  <span id="validationMsg" class="ok">No overlapping intervals</span>
</div>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Start Frame</th>
      <th>End Frame</th>
      <th>Start (s)</th>
      <th>End (s)</th>
      <th>Possession State</th>
      <th>Uncertain</th>
      <th>Verified (Manual)</th>
      <th>Notes</th>
      <th>Delete</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<script>
const matchId = {json.dumps(self.match_id)};
const fps = {float(fps)};
const states = ["", "TEAM_A_CONTROL", "TEAM_B_CONTROL", "CONTESTED", "LOOSE_BALL", "UNKNOWN", "OUT_OF_PLAY"];
let intervals = {intervals_json};

function checkOverlaps() {{
  const active = intervals.filter(iv => iv.state !== "").slice().sort((a, b) => a.start_frame - b.start_frame);
  const msg = document.getElementById("validationMsg");
  for (let i = 0; i < active.length - 1; i++) {{
    if (Number(active[i+1].start_frame) <= Number(active[i].end_frame)) {{
      msg.className = "error";
      msg.textContent = `Overlap error: [${{active[i].start_frame}}, ${{active[i].end_frame}}] overlaps [${{active[i+1].start_frame}}, ${{active[i+1].end_frame}}]`;
      return false;
    }}
  }}
  msg.className = "ok";
  msg.textContent = `Valid (${{active.filter(x => x.verified).length}} verified manual intervals)`;
  return true;
}}

function render() {{
  const tb = document.getElementById("tbody");
  tb.innerHTML = "";
  intervals.forEach((iv, i) => {{
    const tr = document.createElement("tr");
    const opts = states.map(st => `<option value="${{st}}" ${{iv.state===st?'selected':''}}>${{st || '-- Select State --'}}</option>`).join("");
    tr.innerHTML = `
      <td>${{i+1}}</td>
      <td><input type="number" value="${{iv.start_frame}}" style="width:80px" onchange="updateFrame(${{i}}, 'start_frame', this.value)"/></td>
      <td><input type="number" value="${{iv.end_frame}}" style="width:80px" onchange="updateFrame(${{i}}, 'end_frame', this.value)"/></td>
      <td>${{Number(iv.start_timestamp).toFixed(2)}}s</td>
      <td>${{Number(iv.end_timestamp).toFixed(2)}}s</td>
      <td><select onchange="updateState(${{i}}, this.value)">${{opts}}</select></td>
      <td><input type="checkbox" ${{iv.uncertain?'checked':''}} onchange="intervals[${{i}}].uncertain=this.checked; render()"/></td>
      <td><input type="checkbox" ${{iv.verified?'checked':''}} onchange="intervals[${{i}}].verified=this.checked; intervals[${{i}}].source='manual'; render()"/></td>
      <td><input type="text" value="${{iv.notes || ''}}" style="width:200px" onchange="intervals[${{i}}].notes=this.value"/></td>
      <td><button onclick="intervals.splice(${{i}},1); render()">X</button></td>
    `;
    tb.appendChild(tr);
  }});
  checkOverlaps();
}}

function updateFrame(idx, field, val) {{
  const f = parseInt(val, 10) || 0;
  intervals[idx][field] = f;
  if (field === "start_frame") intervals[idx].start_timestamp = Number((f / fps).toFixed(3));
  if (field === "end_frame") intervals[idx].end_timestamp = Number((f / fps).toFixed(3));
  render();
}}

function updateState(idx, val) {{
  intervals[idx].state = val;
  if (val !== "") {{
    intervals[idx].verified = true;
    intervals[idx].source = "manual";
  }}
  render();
}}

function addInterval() {{
  const lastEnd = intervals.length ? Number(intervals[intervals.length - 1].end_frame) + 1 : 0;
  intervals.push({{
    match_id: matchId,
    start_frame: lastEnd,
    end_frame: lastEnd + 29,
    start_timestamp: Number((lastEnd / fps).toFixed(3)),
    end_timestamp: Number(((lastEnd + 29) / fps).toFixed(3)),
    state: "",
    uncertain: false,
    verified: false,
    source: "manual",
    notes: ""
  }});
  render();
}}

function markAllVerified() {{
  intervals.forEach(iv => {{
    if (iv.state !== "") {{
      iv.verified = true;
      iv.source = "manual";
    }}
  }});
  render();
}}

function exportCSV() {{
  if (!checkOverlaps()) {{ alert("Please resolve interval overlaps before exporting."); return; }}
  const rows = ["match_id,start_frame,end_frame,start_timestamp,end_timestamp,state,uncertain,verified,source,notes"];
  intervals.forEach(iv => {{
    rows.push(`${{iv.match_id}},${{iv.start_frame}},${{iv.end_frame}},${{iv.start_timestamp}},${{iv.end_timestamp}},${{iv.state}},${{iv.uncertain}},${{iv.verified}},${{iv.source}},"${{(iv.notes||'').replace(/"/g,'""')}}"`);
  }});
  const blob = new Blob([rows.join("\\n")], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "possession_annotations.csv";
  a.click();
}}

function exportJSON() {{
  if (!checkOverlaps()) {{ alert("Please resolve interval overlaps before exporting."); return; }}
  const payload = {{
    match_id: matchId,
    total_intervals: intervals.length,
    verified_manual_intervals: intervals.filter(x => x.verified && x.source === "manual" && x.state !== "").length,
    intervals: intervals
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "possession_annotations.json";
  a.click();
}}

render();
</script>
</body>
</html>
"""
        html_path.write_text(html_content, encoding="utf-8")
        return html_path


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
            "validation_blocker": "missing_verified_manual_possession_annotations",
            "message": (
                "No manually verified ground-truth possession intervals are present yet. "
                "Open `data/annotations/<match_id>/possession_annotator.html` or edit "
                "`possession_annotations.csv` with `verified=true` and `source=manual` to "
                "compute ground-truth possession accuracy."
            ),
        }

    validate_no_overlapping_intervals(valid_ivs)

    # Map frame_number -> (true_state, uncertain)
    gt_by_frame: Dict[int, Tuple[str, bool]] = {}
    for iv in valid_ivs:
        for fno in range(iv.start_frame, iv.end_frame + 1):
            gt_by_frame[fno] = (iv.state, iv.uncertain)

    y_true: List[str] = []
    y_pred: List[str] = []
    dts: List[float] = []
    timestamps: List[float] = []
    frame_numbers: List[int] = []
    uncertain_frames = 0
    mispredicted_frames: List[Dict[str, Any]] = []

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
        timestamps.append(float(rec.timestamp))
        frame_numbers.append(int(rec.frame_number))
        if gt_state != rec.possession_state and len(mispredicted_frames) < 25:
            mispredicted_frames.append(
                {
                    "frame_number": int(rec.frame_number),
                    "timestamp": round(float(rec.timestamp), 2),
                    "ground_truth_state": gt_state,
                    "predicted_state": rec.possession_state,
                    "ball_observation_mode": rec.ball_observation_mode,
                    "closest_normalized_distance": rec.closest_normalized_distance,
                }
            )

    if not y_true:
        return {
            "metrics_available": False,
            "status": "no_overlapping_certain_frames",
            "validation_blocker": "no_overlapping_certain_frames",
            "uncertain_frames_skipped": uncertain_frames,
        }

    all_states = list(PossessionState.all_states())
    present_states = sorted(set(y_true) | set(y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=present_states, average="macro", zero_division=0))
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

    # State transition counts & transition timing error (seconds)
    gt_transition_times = [
        timestamps[i + 1] for i in range(len(y_true) - 1) if y_true[i] != y_true[i + 1]
    ]
    pred_transition_times = [
        timestamps[i + 1] for i in range(len(y_pred) - 1) if y_pred[i] != y_pred[i + 1]
    ]
    gt_transitions = len(gt_transition_times)
    pred_transitions = len(pred_transition_times)
    transition_error = abs(pred_transitions - gt_transitions)

    if gt_transition_times and pred_transition_times:
        timing_diffs = [
            min(abs(gt_t - pr_t) for pr_t in pred_transition_times)
            for gt_t in gt_transition_times
        ]
        transition_timing_error_sec = float(np.mean(timing_diffs))
    elif not gt_transition_times and not pred_transition_times:
        transition_timing_error_sec = 0.0
    else:
        transition_timing_error_sec = float(sum(dts))

    total_eval_sec = sum(dts)
    confident_pred_sec = sum(
        dt
        for p, dt in zip(y_pred, dts)
        if p in (PossessionState.TEAM_A_CONTROL.value, PossessionState.TEAM_B_CONTROL.value)
    )
    unknown_pred_sec = sum(
        dt for p, dt in zip(y_pred, dts) if p == PossessionState.UNKNOWN.value
    )
    confident_coverage = (confident_pred_sec / total_eval_sec) if total_eval_sec > 0 else 0.0
    unknown_state_rate = (unknown_pred_sec / total_eval_sec) if total_eval_sec > 0 else 0.0

    return {
        "metrics_available": True,
        "status": "verified_manual_possession_evaluation",
        "validation_blocker": None,
        "evaluated_frames": len(y_true),
        "uncertain_frames_excluded": uncertain_frames,
        "state_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_state": per_state,
        "team_a_time_error_seconds": round(abs(pred_a_sec - gt_a_sec), 2),
        "team_b_time_error_seconds": round(abs(pred_b_sec - gt_b_sec), 2),
        "team_possession_share_error_pct_points": round(share_error_pct_pts, 2),
        "gt_team_a_share_pct": round(gt_a_share, 2),
        "pred_team_a_share_pct": round(pred_a_share, 2),
        "gt_transitions": gt_transitions,
        "pred_transitions": pred_transitions,
        "transition_count_error": transition_error,
        "transition_timing_error_seconds": round(transition_timing_error_sec, 3),
        "unknown_state_rate": round(unknown_state_rate, 4),
        "confident_attribution_coverage": round(confident_coverage, 4),
        "confidently_assigned_possession_pct": round(confident_coverage * 100.0, 2),
        "state_labels": all_states,
        "confusion_matrix": cm.tolist(),
        "mispredicted_frames": mispredicted_frames,
    }

