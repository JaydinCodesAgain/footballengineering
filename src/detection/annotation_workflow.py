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
    legacy_counts: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameAnnotation":
        boxes = [AnnotatedBox(class_name=b["class_name"], bbox=list(b["bbox"])) for b in data.get("boxes", [])]
        return cls(
            match_id=str(data["match_id"]),
            frame_number=int(data["frame_number"]),
            timestamp=float(data.get("timestamp", 0.0)),
            image_rel_path=str(data.get("image_rel_path", "")),
            verified=bool(data.get("verified", False)),
            source=str(data.get("source", "unlabelled")),
            scene_tag=str(data.get("scene_tag", "general_play")),
            boxes=boxes,
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
        """
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
                    cv2.imwrite(str(abs_img), frame)

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
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

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
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
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

    def load(self) -> None:
        """Load annotations from `detection_annotations.json`."""
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.frames.clear()
        for item in data.get("frames", []):
            ann = FrameAnnotation.from_dict(item)
            self.frames[ann.frame_number] = ann

    def import_boxes_from_csv(self, csv_path: Optional[str | Path] = None) -> int:
        """
        Load manual bounding box annotations from `detection_boxes.csv` (or a specified CSV file).
        Rejects any row whose `source` is `'model_prediction'` or `'auto'` from being marked `verified`.
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
                src = row.get("source", "manual").strip() or "manual"
                tag = row.get("scene_tag", "general_play").strip() or "general_play"
                if src.lower() in ("model", "model_prediction", "auto", "predicted", "yolo"):
                    raise ValueError(
                        f"Frame {fno}: Ground-truth boxes cannot be imported with source='{src}'. "
                        "Predictions must never be presented as independent ground truth."
                    )

                frame_meta[fno] = (ts, ver, src, tag)
                grouped_boxes.setdefault(fno, [])

                cls_name = row.get("class_name", "").strip()
                if cls_name:
                    x1 = int(float(row["x1"]))
                    y1 = int(float(row["y1"]))
                    x2 = int(float(row["x2"]))
                    y2 = int(float(row["y2"]))
                    box = AnnotatedBox(class_name=cls_name, bbox=[x1, y1, x2, y2])
                    box.validate()
                    grouped_boxes[fno].append(box)

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
  <button onclick="undoBox()">Undo Last Box (Z)</button>
  <button onclick="clearFrameBoxes()">Clear Frame</button>
  <label><input type="checkbox" id="chk-verified" onchange="toggleVerified(this.checked)"> Mark Frame Verified (Manual GT)</label>
  <button class="primary" onclick="exportCSV()">Download detection_boxes.csv</button>
  <span class="status">Drag on image to draw box. Save CSV to data/annotations/{self.match_id}/detection_boxes.csv</span>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<script>
const frames = {frames_json};
let idx = 0, currentClass = 'ball', dragging = false, startX = 0, startY = 0;
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const img = new Image();

function loadFrame(i) {{
  if (!frames.length) return;
  idx = Math.max(0, Math.min(frames.length - 1, i));
  const f = frames[idx];
  document.getElementById('frame-info').textContent = `#${{idx+1}}/${{frames.length}} | Frame ${{f.frame_number}} (${{f.timestamp}}s) [${{f.scene_tag}}] | Boxes: ${{f.boxes.length}}`;
  document.getElementById('chk-verified').checked = !!f.verified;
  img.onload = () => {{ canvas.width = img.width; canvas.height = img.height; render(); }};
  img.src = f.image_rel_path;
}}

function render(previewRect) {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  const f = frames[idx];
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
  const r = canvas.getBoundingClientRect();
  startX = Math.round(e.clientX - r.left);
  startY = Math.round(e.clientY - r.top);
  dragging = true;
}});
canvas.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  const cx = Math.round(e.clientX - r.left), cy = Math.round(e.clientY - r.top);
  render([Math.min(startX, cx), Math.min(startY, cy), Math.max(startX, cx), Math.max(startY, cy)]);
}});
canvas.addEventListener('mouseup', e => {{
  if (!dragging) return;
  dragging = false;
  const r = canvas.getBoundingClientRect();
  const cx = Math.round(e.clientX - r.left), cy = Math.round(e.clientY - r.top);
  const x1 = Math.min(startX, cx), y1 = Math.min(startY, cy), x2 = Math.max(startX, cx), y2 = Math.max(startY, cy);
  if (x2 - x1 >= 4 && y2 - y1 >= 4) {{
    frames[idx].boxes.push({{ class_name: currentClass, bbox: [x1, y1, x2, y2] }});
    frames[idx].verified = true;
    frames[idx].source = 'manual';
    document.getElementById('chk-verified').checked = true;
  }}
  loadFrame(idx);
}});

function setClass(c) {{
  currentClass = c;
  document.getElementById('btn-ball').className = c === 'ball' ? 'active-cls' : '';
  document.getElementById('btn-player').className = c === 'player' ? 'active-cls' : '';
}}
function undoBox() {{ frames[idx].boxes.pop(); loadFrame(idx); }}
function clearFrameBoxes() {{ frames[idx].boxes = []; loadFrame(idx); }}
function toggleVerified(v) {{ frames[idx].verified = v; if (v) frames[idx].source = 'manual'; }}
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
}}
loadFrame(0);
</script>
</body>
</html>
"""
        self.html_path.write_text(html_content, encoding="utf-8")
        return self.html_path
