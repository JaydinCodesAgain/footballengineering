"""Ground-truth bounding-box annotation workflow and interactive HTML5 review tool for player and ball detection."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np


def sample_representative_frame_numbers(
    total_frames: int,
    num_samples: int = 45,
    start_fraction: float = 0.12,
    end_fraction: float = 0.96,
    mandatory_frames: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Sample 40–60 representative frames across a match video, preserving any mandatory
    historical evaluation frames while spanning wide-angle play, transitions, and late-match action.
    """
    if total_frames <= 0:
        return list(mandatory_frames or [])

    low = max(0, int(total_frames * start_fraction))
    high = max(low + 1, min(total_frames - 1, int(total_frames * end_fraction)))
    grid = np.unique(np.linspace(low, high, num_samples).astype(int)).tolist()

    combined = sorted(set(grid) | {int(f) for f in (mandatory_frames or []) if 0 <= int(f) < total_frames})
    return combined


@dataclass
class AnnotatedBox:
    """Single ground-truth bounding box annotation."""

    class_name: str  # "player" or "ball"
    bbox: List[int]  # [x1, y1, x2, y2]

    def validate(self) -> None:
        if self.class_name not in ("player", "ball"):
            raise ValueError(f"Invalid class_name '{self.class_name}'. Expected 'player' or 'ball'.")
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError(f"Invalid bounding box coordinates: {self.bbox}")


@dataclass
class FrameAnnotation:
    """Ground-truth annotation state for a single video frame."""

    match_id: str
    frame_number: int
    timestamp: float
    image_rel_path: str
    verified: bool = False
    source: str = "unlabelled"  # MUST be "manual" with verified=True for independent evaluation
    scene_tag: str = "general_play"
    boxes: List[AnnotatedBox] = field(default_factory=list)
    # Suggestions are deliberately separate from annotations.  They may be drawn in
    # the review UI, but are never exported as ground truth until a human redraws
    # and verifies the corresponding box.
    model_proposals: List[AnnotatedBox] = field(default_factory=list)
    legacy_counts: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameAnnotation":
        boxes = [AnnotatedBox(class_name=b["class_name"], bbox=list(b["bbox"])) for b in data.get("boxes", [])]
        model_proposals = [
            AnnotatedBox(class_name=b["class_name"], bbox=list(b["bbox"]))
            for b in data.get("model_proposals", [])
        ]
        return cls(
            match_id=str(data["match_id"]),
            frame_number=int(data["frame_number"]),
            timestamp=float(data.get("timestamp", 0.0)),
            image_rel_path=str(data.get("image_rel_path", "")),
            verified=bool(data.get("verified", False)),
            source=str(data.get("source", "unlabelled")),
            scene_tag=str(data.get("scene_tag", "general_play")),
            boxes=boxes,
            model_proposals=model_proposals,
            legacy_counts=data.get("legacy_counts"),
        )


class DetectionAnnotationStore:
    """
    Manages frame-level bounding-box annotations on disk via JSON, CSV, and an interactive HTML5
    canvas review/annotation interface (`annotator.html`).
    Enforces strict provenance: model-predicted proposals are NEVER treated as verified ground truth.
    """

    def __init__(self, root_dir: str | Path, match_id: str) -> None:
        self.root_dir = Path(root_dir)
        self.match_id = match_id
        self.frames_dir = self.root_dir / "frames"
        self.json_path = self.root_dir / "detection_annotations.json"
        self.csv_path = self.root_dir / "detection_boxes.csv"
        self.html_path = self.root_dir / "detection_annotator.html"
        self.frames: Dict[int, FrameAnnotation] = {}

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.json_path.is_file():
            self.load()

    def export_frames_for_annotation(
        self,
        video_path: str | Path,
        frame_numbers: Sequence[int],
        legacy_count_records: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> List[FrameAnnotation]:
        """
        Extract specific video frames to `frames/frame_<id>.jpg`, initialize empty annotation
        templates ready for manual bounding-box labelling, and generate `detection_annotator.html`.

        Any CSV exported by the browser is imported before this method writes.  This
        makes preparation resumable and prevents a stale JSON file from silently
        replacing completed CSV-only review work.
        """
        if self.csv_path.is_file():
            self.import_boxes_from_csv(self.csv_path, allow_destructive=False)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video for frame extraction: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        exported: List[FrameAnnotation] = []
        scene_categories = [
            "normal_possession",
            "aerial_or_long_pass",
            "crowded_box",
            "camera_pan_or_zoom",
            "partial_occlusion",
            "difficult_background",
        ]

        try:
            for idx, fno in enumerate(sorted(set(int(f) for f in frame_numbers))):
                rel_path = f"frames/frame_{fno:06d}.jpg"
                abs_img = self.root_dir / rel_path
                if not abs_img.is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    if not cv2.imwrite(str(abs_img), frame):
                        raise ValueError(f"Could not write annotation frame: {abs_img}")

                tag = scene_categories[idx % len(scene_categories)]
                if fno not in self.frames:
                    self.frames[fno] = FrameAnnotation(
                        match_id=self.match_id,
                        frame_number=fno,
                        timestamp=round(fno / fps, 3),
                        image_rel_path=rel_path,
                        verified=False,
                        source="unlabelled",
                        scene_tag=tag,
                        boxes=[],
                        legacy_counts=(legacy_count_records or {}).get(fno),
                    )
                elif legacy_count_records and fno in legacy_count_records:
                    self.frames[fno].legacy_counts = legacy_count_records[fno]

                exported.append(self.frames[fno])
        finally:
            cap.release()

        self.save()
        self.generate_html_annotator()
        return exported

    def set_manual_frame_boxes(
        self,
        frame_number: int,
        boxes: Sequence[AnnotatedBox],
        timestamp: float = 0.0,
        image_rel_path: str = "",
        verified: bool = True,
        scene_tag: str = "general_play",
    ) -> FrameAnnotation:
        """Record manually verified ground-truth bounding boxes for a frame."""
        validated_boxes: List[AnnotatedBox] = []
        for b in boxes:
            b.validate()
            validated_boxes.append(b)

        existing = self.frames.get(frame_number)
        ann = FrameAnnotation(
            match_id=self.match_id,
            frame_number=int(frame_number),
            timestamp=existing.timestamp if existing else round(float(timestamp), 3),
            image_rel_path=existing.image_rel_path if existing else image_rel_path,
            verified=bool(verified),
            source="manual",
            scene_tag=existing.scene_tag if existing else scene_tag,
            boxes=validated_boxes,
            model_proposals=existing.model_proposals if existing else [],
            legacy_counts=existing.legacy_counts if existing else None,
        )
        self.frames[int(frame_number)] = ann
        return ann

    def save(self) -> None:
        """Save annotations to JSON and a flat human-editable CSV table."""
        ordered = [self.frames[k] for k in sorted(self.frames.keys())]
        verified_frames = [
            f for f in ordered if f.verified and f.source == "manual"
        ]
        payload = {
            "match_id": self.match_id,
            "total_frames": len(ordered),
            "verified_manual_box_frames": len(verified_frames),
            "evaluation_ready": len(verified_frames) > 0,
            "validation_blocker": (
                None
                if len(verified_frames) > 0
                else f"0/{len(ordered)} frames have verified manual bounding boxes (`verified=true, source=manual`)."
            ),
            "frames": [f.to_dict() for f in ordered],
        }
        json_temp_path = self.json_path.with_suffix(".json.tmp")
        with json_temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        json_temp_path.replace(self.json_path)

        fieldnames = [
            "frame_number",
            "timestamp",
            "scene_tag",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "verified",
            "source",
        ]
        csv_temp_path = self.csv_path.with_suffix(".csv.tmp")
        with csv_temp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for f_ann in ordered:
                if not f_ann.boxes:
                    writer.writerow(
                        {
                            "frame_number": f_ann.frame_number,
                            "timestamp": f_ann.timestamp,
                            "scene_tag": f_ann.scene_tag,
                            "class_name": "",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "verified": f_ann.verified,
                            "source": f_ann.source,
                        }
                    )
                else:
                    for b in f_ann.boxes:
                        writer.writerow(
                            {
                                "frame_number": f_ann.frame_number,
                                "timestamp": f_ann.timestamp,
                                "scene_tag": f_ann.scene_tag,
                                "class_name": b.class_name,
                                "x1": b.bbox[0],
                                "y1": b.bbox[1],
                                "x2": b.bbox[2],
                                "y2": b.bbox[3],
                                "verified": f_ann.verified,
                                "source": f_ann.source,
                            }
                        )

        csv_temp_path.replace(self.csv_path)

    def load(self) -> None:
        """Load annotations from `detection_annotations.json`."""
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.frames.clear()
        for item in data.get("frames", []):
            ann = FrameAnnotation.from_dict(item)
            self.frames[ann.frame_number] = ann

    def import_boxes_from_csv(
        self,
        csv_path: Optional[str | Path] = None,
        *,
        allow_destructive: bool = False,
    ) -> int:
        """
        Load manual bounding box annotations from `detection_boxes.csv` (or a specified CSV file).
        Rejects model provenance and inconsistent per-frame metadata. Existing
        verified manual work may not be removed or changed unless an explicit
        caller opts into ``allow_destructive`` after making a backup.
        """
        target = Path(csv_path) if csv_path else self.csv_path
        if not target.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {target}")

        grouped_boxes: Dict[int, List[AnnotatedBox]] = {}
        frame_meta: Dict[int, Tuple[float, bool, str, str]] = {}

        with target.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fno_str = row.get("frame_number", "").strip()
                if not fno_str:
                    continue
                fno = int(fno_str)
                ts = float(row.get("timestamp", 0.0) or 0.0)
                ver = row.get("verified", "false").strip().lower() in ("true", "1", "yes", "y")
                src = (row.get("source", "manual").strip() or "manual").lower()
                tag = row.get("scene_tag", "general_play").strip() or "general_play"
                if src.lower() in ("model", "model_prediction", "auto", "predicted", "yolo"):
                    raise ValueError(
                        f"Frame {fno}: Ground-truth boxes cannot be imported with source='{src}'. "
                        "Predictions must never be presented as independent ground truth."
                    )
                if src not in ("manual", "unlabelled"):
                    raise ValueError(
                        f"Frame {fno}: unsupported annotation source='{src}'. Expected 'manual' or 'unlabelled'."
                    )
                if ver and src != "manual":
                    raise ValueError(
                        f"Frame {fno}: verified annotations must have source='manual', not '{src}'."
                    )

                metadata = (ts, ver, src, tag)
                if fno in frame_meta and frame_meta[fno] != metadata:
                    raise ValueError(
                        f"Frame {fno}: CSV rows disagree on timestamp, verified flag, source, or scene tag."
                    )
                frame_meta[fno] = metadata
                grouped_boxes.setdefault(fno, [])

                cls_name = row.get("class_name", "").strip()
                if cls_name:
                    x1 = int(float(row["x1"]))
                    y1 = int(float(row["y1"]))
                    x2 = int(float(row["x2"]))
                    y2 = int(float(row["y2"]))
                    box = AnnotatedBox(class_name=cls_name, bbox=[x1, y1, x2, y2])
                    box.validate()
                    if any(existing_box.class_name == box.class_name and existing_box.bbox == box.bbox for existing_box in grouped_boxes[fno]):
                        raise ValueError(f"Frame {fno}: duplicate {cls_name} bounding box in CSV.")
                    grouped_boxes[fno].append(box)

        if not allow_destructive:
            for existing_fno, existing in self.frames.items():
                if not (existing.verified and existing.source == "manual"):
                    continue
                incoming = frame_meta.get(existing_fno)
                incoming_boxes = grouped_boxes.get(existing_fno)
                if incoming is None or not (incoming[1] and incoming[2] == "manual"):
                    raise ValueError(
                        f"Frame {existing_fno}: refusing to discard existing verified manual review. "
                        "Restore the complete exported CSV or use an explicit destructive recovery workflow with a backup."
                    )
                existing_signature = sorted((box.class_name, tuple(box.bbox)) for box in existing.boxes)
                incoming_signature = sorted((box.class_name, tuple(box.bbox)) for box in (incoming_boxes or []))
                if incoming_signature != existing_signature:
                    raise ValueError(
                        f"Frame {existing_fno}: refusing to replace existing verified manual boxes without an explicit "
                        "destructive recovery workflow and backup."
                    )

        updated = 0
        for fno, boxes in grouped_boxes.items():
            ts, ver, src, tag = frame_meta[fno]
            existing = self.frames.get(fno)
            self.frames[fno] = FrameAnnotation(
                match_id=self.match_id,
                frame_number=fno,
                timestamp=existing.timestamp if existing else ts,
                image_rel_path=existing.image_rel_path if existing else f"frames/frame_{fno:06d}.jpg",
                verified=bool(ver and src == "manual"),
                source=src,
                scene_tag=tag,
                boxes=boxes,
                model_proposals=existing.model_proposals if existing else [],
                legacy_counts=existing.legacy_counts if existing else None,
            )
            updated += 1

        self.save()
        return updated

    def get_verified_ground_truth(self) -> List[FrameAnnotation]:
        """
        Return ONLY frames that have been explicitly verified (`verified=True`)
        with independent manual provenance (`source='manual'`).
        """
        return [
            self.frames[k]
            for k in sorted(self.frames.keys())
            if self.frames[k].verified and self.frames[k].source == "manual"
        ]

    def generate_html_annotator(self) -> Path:
        """
        Generate a standalone, zero-dependency HTML5 canvas bounding-box annotation interface
        (`detection_annotator.html`) so a human reviewer can draw player & ball boxes in any
        browser and export `detection_boxes.csv`.
        """
        frames_list = [self.frames[k].to_dict() for k in sorted(self.frames.keys())]
        frames_json = json.dumps(frames_list)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Football AI — Detection Ground-Truth Annotator ({self.match_id})</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #141619; color: #f0f0f0; margin: 0; padding: 16px; }}
  .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; background: #212529; padding: 10px 14px; border-radius: 6px; }}
  button, select {{ padding: 6px 12px; border-radius: 4px; border: 1px solid #495057; background: #343a40; color: #fff; cursor: pointer; font-weight: 600; }}
  button.primary {{ background: #198754; border-color: #198754; }}
  button.active-cls {{ outline: 2px solid #ffc107; }}
  #canvas-wrap {{ position: relative; display: inline-block; border: 2px solid #495057; overflow: auto; max-width: 100%; }}
  canvas {{ display: block; cursor: crosshair; }}
  .status {{ font-size: 14px; color: #adb5bd; }}
</style>
</head>
<body>
<div class="toolbar">
  <strong>Match: {self.match_id}</strong>
  <button onclick="prevFrame()">&#9664; Prev Frame</button>
  <span id="frame-info">Frame 0/0</span>
  <button onclick="nextFrame()">Next Frame &#9654;</button>
  <button id="btn-ball" class="active-cls" onclick="setClass('ball')">Class: BALL (Key 1)</button>
  <button id="btn-player" onclick="setClass('player')">Class: PLAYER (Key 2)</button>
  <button onclick="setZoom(zoom - 0.25)">Zoom −</button>
  <button onclick="setZoom(zoom + 0.25)">Zoom +</button>
  <button onclick="setZoom(1)">100%</button>
  <button onclick="undoBox()">Undo Last Box (Z)</button>
  <button onclick="clearFrameBoxes()">Clear Frame</button>
  <label><input type="checkbox" id="chk-verified" onchange="toggleVerified(this.checked)"> Mark Frame Verified (Manual GT)</label>
  <button onclick="saveLocalProgress()">Save Browser Progress</button>
  <button onclick="restoreLocalProgress()">Restore Browser Progress</button>
  <label>Import CSV <input id="csv-import" type="file" accept=".csv,text/csv" onchange="importCSV(event)"></label>
  <button class="primary" onclick="exportCSV()">Download detection_boxes.csv</button>
  <span id="review-status" class="status">Dashed teal boxes are model suggestions only: inspect them, then redraw every accepted box. Browser progress is saved locally after each edit; download CSV before moving browsers or devices.</span>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<script>
const frames = {frames_json};
const storageKey = 'football-engineering-annotations:{self.match_id}:' + frames.map(f => f.frame_number).join('-');
let idx = 0, currentClass = 'ball', dragging = false, startX = 0, startY = 0, zoom = 1;
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const img = new Image();

function setStatus(message) {{ document.getElementById('review-status').textContent = message; }}
function setCanvasDisplaySize() {{
  canvas.style.width = `${{Math.round(canvas.width * zoom)}}px`;
  canvas.style.height = `${{Math.round(canvas.height * zoom)}}px`;
}}
function setZoom(value) {{
  zoom = Math.max(0.5, Math.min(4, Math.round(value * 100) / 100));
  setCanvasDisplaySize();
  setStatus(`Zoom ${{Math.round(zoom * 100)}}%. Coordinates remain in native image pixels.`);
}}
function pointerToImageCoordinates(event) {{
  const rect = canvas.getBoundingClientRect();
  return [
    Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    Math.round((event.clientY - rect.top) * canvas.height / rect.height),
  ];
}}
function progressPayload() {{
  return frames.map(f => ({{
    frame_number: f.frame_number,
    boxes: f.boxes,
    verified: !!f.verified,
    source: f.source,
  }}));
}}
function saveLocalProgress(quiet = false) {{
  localStorage.setItem(storageKey, JSON.stringify(progressPayload()));
  if (!quiet) setStatus('Browser progress saved. Download the CSV as the portable review record.');
}}
function restoreLocalProgress() {{
  const raw = localStorage.getItem(storageKey);
  if (!raw) {{ setStatus('No browser progress exists for this exact review batch.'); return; }}
  let saved;
  try {{ saved = JSON.parse(raw); }} catch (_) {{ setStatus('Saved browser progress is unreadable; it was not applied.'); return; }}
  if (!Array.isArray(saved) || saved.length !== frames.length) {{ setStatus('Saved browser progress belongs to another batch; it was not applied.'); return; }}
  const byFrame = new Map(saved.map(item => [item.frame_number, item]));
  if (frames.some(f => !byFrame.has(f.frame_number))) {{ setStatus('Saved browser progress has different frames; it was not applied.'); return; }}
  if (!confirm('Restore browser-saved manual boxes and verification state? Current unsaved browser edits will be replaced.')) return;
  for (const f of frames) {{
    const savedFrame = byFrame.get(f.frame_number);
    f.boxes = Array.isArray(savedFrame.boxes) ? savedFrame.boxes : [];
    f.verified = !!savedFrame.verified;
    f.source = savedFrame.source === 'manual' ? 'manual' : 'unlabelled';
  }}
  loadFrame(idx);
  setStatus('Browser progress restored. Download CSV to make this review portable.');
}}

function loadFrame(i) {{
  if (!frames.length) return;
  idx = Math.max(0, Math.min(frames.length - 1, i));
  const f = frames[idx];
  const proposalCount = (f.model_proposals || []).length;
  document.getElementById('frame-info').textContent = `#${{idx+1}}/${{frames.length}} | Frame ${{f.frame_number}} (${{f.timestamp}}s) [${{f.scene_tag}}] | Manual: ${{f.boxes.length}} | Suggestions: ${{proposalCount}}`;
  document.getElementById('chk-verified').checked = !!f.verified;
  img.onload = () => {{ canvas.width = img.width; canvas.height = img.height; setCanvasDisplaySize(); render(); }};
  img.src = f.image_rel_path;
}}

function render(previewRect) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  const f = frames[idx];
  ctx.setLineDash([7, 5]);
  for (const b of (f.model_proposals || [])) {{
    const [x1, y1, x2, y2] = b.bbox;
    ctx.strokeStyle = '#20c997';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = '#20c997';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(`SUGGESTED ${{b.class_name.toUpperCase()}}`, x1, Math.max(14, y1 - 4));
  }}
  ctx.setLineDash([]);
  for (const b of f.boxes) {{
    const [x1, y1, x2, y2] = b.bbox;
    ctx.strokeStyle = b.class_name === 'ball' ? '#ffc107' : '#20c997';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(b.class_name.toUpperCase(), x1, Math.max(14, y1 - 4));
  }}
  if (previewRect) {{
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 2;
    ctx.strokeRect(previewRect[0], previewRect[1], previewRect[2] - previewRect[0], previewRect[3] - previewRect[1]);
  }}
}}

canvas.addEventListener('mousedown', e => {{
  [startX, startY] = pointerToImageCoordinates(e);
  dragging = true;
}});
canvas.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const [cx, cy] = pointerToImageCoordinates(e);
  render([Math.min(startX, cx), Math.min(startY, cy), Math.max(startX, cx), Math.max(startY, cy)]);
}});
canvas.addEventListener('mouseup', e => {{
  if (!dragging) return;
  dragging = false;
  const [cx, cy] = pointerToImageCoordinates(e);
  const x1 = Math.min(startX, cx), y1 = Math.min(startY, cy), x2 = Math.max(startX, cx), y2 = Math.max(startY, cy);
  if (x2 - x1 >= 4 && y2 - y1 >= 4) {{
    frames[idx].boxes.push({{ class_name: currentClass, bbox: [x1, y1, x2, y2] }});
    frames[idx].verified = true;
    frames[idx].source = 'manual';
    document.getElementById('chk-verified').checked = true;
    saveLocalProgress(true);
  }}
  loadFrame(idx);
}});

function setClass(c) {{
  currentClass = c;
  document.getElementById('btn-ball').className = c === 'ball' ? 'active-cls' : '';
  document.getElementById('btn-player').className = c === 'player' ? 'active-cls' : '';
}}
function undoBox() {{ if (frames[idx].boxes.length) {{ frames[idx].boxes.pop(); saveLocalProgress(true); }} loadFrame(idx); }}
function clearFrameBoxes() {{
  if (frames[idx].boxes.length && !confirm('Clear all manual boxes on this frame? Use this only after inspecting the frame again.')) return;
  frames[idx].boxes = []; saveLocalProgress(true); loadFrame(idx);
}}
function toggleVerified(v) {{ frames[idx].verified = v; if (v) frames[idx].source = 'manual'; saveLocalProgress(true); }}
function prevFrame() {{ loadFrame(idx - 1); }}
function nextFrame() {{ loadFrame(idx + 1); }}
window.addEventListener('keydown', e => {{
  if (e.key === '1') setClass('ball');
  if (e.key === '2') setClass('player');
  if (e.key === 'z' || e.key === 'Z') undoBox();
  if (e.key === 'ArrowLeft') prevFrame();
  if (e.key === 'ArrowRight') nextFrame();
}});

function exportCSV() {{
  saveLocalProgress(true);
  const rows = ['frame_number,timestamp,scene_tag,class_name,x1,y1,x2,y2,verified,source'];
  for (const f of frames) {{
    const src = f.verified ? 'manual' : f.source;
    if (!f.boxes.length) {{
      rows.push(`${{f.frame_number}},${{f.timestamp}},${{f.scene_tag}},,,,,,${{f.verified}},${{src}}`);
    }} else {{
      for (const b of f.boxes) {{
        rows.push(`${{f.frame_number}},${{f.timestamp}},${{f.scene_tag}},${{b.class_name}},${{b.bbox[0]}},${{b.bbox[1]}},${{b.bbox[2]}},${{b.bbox[3]}},${{f.verified}},${{src}}`);
      }}
    }}
  }}
  const blob = new Blob([rows.join('\\n') + '\\n'], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'detection_boxes.csv';
  a.click();
  setStatus('CSV downloaded. Replace detection_boxes.csv in this match folder, then run the training gate.');
}}
function importCSV(event) {{
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    const lines = String(reader.result).trim().split(/\r?\n/);
    const expectedHeader = 'frame_number,timestamp,scene_tag,class_name,x1,y1,x2,y2,verified,source';
    if (!lines.length || lines[0] !== expectedHeader) {{ setStatus('CSV header is invalid; no progress was changed.'); return; }}
    const pending = new Map();
    for (const line of lines.slice(1)) {{
      if (!line) continue;
      const values = line.split(',');
      if (values.length !== 10) {{ setStatus('CSV row is invalid; no progress was changed.'); return; }}
      const [frameNumber, , , className, x1, y1, x2, y2, verified, source] = values;
      const fno = Number(frameNumber);
      const target = frames.find(f => f.frame_number === fno);
      if (!target || !Number.isInteger(fno) || (source !== 'manual' && source !== 'unlabelled')) {{ setStatus('CSV contains an unknown frame or unsupported provenance; no progress was changed.'); return; }}
      if (!pending.has(fno)) pending.set(fno, {{ boxes: [], verified: verified === 'true', source }});
      const state = pending.get(fno);
      if (state.verified !== (verified === 'true') || state.source !== source) {{ setStatus('CSV has inconsistent rows for one frame; no progress was changed.'); return; }}
      if (className) {{
        const box = [Number(x1), Number(y1), Number(x2), Number(y2)];
        if (className !== 'ball' && className !== 'player' || !box.every(Number.isFinite) || box[2] <= box[0] || box[3] <= box[1]) {{ setStatus('CSV contains an invalid box; no progress was changed.'); return; }}
        state.boxes.push({{ class_name: className, bbox: box }});
      }}
    }}
    if (pending.size !== frames.length) {{ setStatus('CSV is incomplete for this review batch; no progress was changed.'); return; }}
    if (!confirm('Import this complete CSV and replace current browser review state?')) return;
    for (const f of frames) {{ const state = pending.get(f.frame_number); f.boxes = state.boxes; f.verified = state.verified; f.source = state.source; }}
    saveLocalProgress(true); loadFrame(idx); setStatus('CSV imported into browser progress. Verify it visually, then download a fresh CSV.');
  }};
  reader.readAsText(file);
}}
loadFrame(0);
</script>
</body>
</html>
"""
        self.html_path.write_text(html_content, encoding="utf-8")
        return self.html_path
