# Football AI Engineering — Broadcast Match Analytics Pipeline

An end-to-end computer vision and sports analytics pipeline for broadcast football footage using **Ultralytics YOLO11**, **OpenCV**, **Scikit-learn**, **ByteTrack**, and **BoT-SORT**.

## Key Capabilities

1. **Player & Football Detection (`src/detection/`)**
   - Decoupled confidence thresholds (`person_conf` vs `ball_conf`) to improve small, fast-moving football recall without inflating player false positives.
   - Ground-truth bounding-box annotation store (`DetectionAnnotationStore`) with strict manual-provenance enforcement (`verified=True, source="manual"`).
   - One-to-one IoU matching (`greedy_confidence` and `hungarian`), Precision, Recall, F1, AP@0.50, mAP@0.50:0.95, and confidence-threshold sweep diagnostics.

2. **Match-Agnostic Jersey Classification (`src/classification/`)**
   - Generic internal labels (`Team_A`, `Team_B`, `Referee`, `Unknown`) mapped via YAML configurations (`configs/norway_wales.yaml`, `configs/chelsea_sunderland.yaml`) with cross-match safety guards.
   - Torso extraction with automatic quality filtering: Laplacian variance blur rejection, HSV pitch-grass ratio filtering, and inter-player torso occlusion checks.
   - Persistent dataset manifests (`manifest.json`, `manifest.csv`) with stable crop identifiers (`crop_id`), timestamps, bounding boxes, and leakage-free temporal/track splits.
   - Six benchmarked feature extractors: `baseline_hsv`, `legacy_enhanced`, `hsv_lab`, `foreground_masked`, `dominant_colour`, and `spatial_embedding`.
   - Automated Leave-One-Out (LOO) + centroid label auditing (`audit_dataset_labels`) and validation-calibrated Unknown distance rejection.

3. **Persistent Tracking & Temporal Stabilization (`src/tracking/`)**
   - Supports both **ByteTrack** (`bytetrack.yaml`) and **BoT-SORT** (`botsort.yaml`) with apples-to-apples diagnostic comparison.
   - `TemporalTeamStabilizer` with confidence-weighted voting, minimum evidence requirements, transient Unknown handling, soft track-level label locking, sustained-contradiction unlocking, and spatial-jump / missing-frame identity resets.

4. **Possession Analytics Foundation & Reporting (`src/evaluation/`, `src/visualization/`)**
   - Exports structured per-frame telemetry (`frame_observations.csv`) recording frame number, timestamp, player ID, internal/display team ID, player & ball bounding boxes, detection confidences, and classification uncertainty.
   - Generates timestamped experiment directories under `outputs/` containing `analytics_report.json`, `analytics_report.md`, benchmark CSVs, confusion matrices, calibration curves, misclassification galleries, and annotated MP4 videos.

---

## Project Structure

```text
footballengineering/
├── configs/
│   ├── chelsea_sunderland.yaml
│   └── norway_wales.yaml
├── data/
│   ├── raw/
│   ├── crops/
│   ├── annotations/
│   └── splits/
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── detection/
│   │   ├── detector.py
│   │   └── annotation_workflow.py
│   ├── classification/
│   │   ├── cropper.py
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── classifier.py
│   │   └── diagnostics.py
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── stabilizer.py
│   ├── evaluation/
│   │   ├── detection_eval.py
│   │   ├── classification_eval.py
│   │   ├── tracking_eval.py
│   │   └── report.py
│   ├── visualization/
│   │   ├── video_annotator.py
│   │   └── plots.py
│   └── pipeline/
│       ├── legacy_migration.py
│       ├── runner.py
│       └── cli.py
├── notebooks/
│   └── football_engineering.ipynb
├── tests/
├── outputs/
├── docs/
│   └── TECHNICAL_AUDIT.md
├── requirements.txt
└── README.md
```

---

## Installation & Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the Automated Unit Test Suite

All unit tests use deterministic synthetic fixtures and run offline without requiring external video files or network access:

```bash
python -m pytest -v
```

### 3. Run the End-to-End Pipeline

**Norway vs Wales:**

```bash
python -m src.pipeline.cli --config configs/norway_wales.yaml --max-frames 180 --compare-trackers
```

**Chelsea vs Sunderland:**

```bash
python -m src.pipeline.cli --config configs/chelsea_sunderland.yaml --max-frames 180 --compare-trackers
```

---

## How to Process a Completely New Match (Zero Code Changes)

1. **Create a Match Config YAML:**
   Copy `configs/norway_wales.yaml` to `configs/my_new_match.yaml` and update:
   - `match_id: my_new_match`
   - `match_name: "Home Team vs Away Team"`
   - `video_path: "data/raw/my_new_match.mp4"`
   - `label_mapping`: map `Team_A` and `Team_B` to your teams' display names and set `team_colours_bgr`.

2. **Extract Candidate Jersey Crops:**
   Run dataset preparation to automatically extract quality-filtered torso crops and assign leakage-free `train`, `val`, and `test` temporal splits:
   ```python
   from src.config import load_match_config
   from src.pipeline.runner import prepare_dataset_for_match

   cfg = load_match_config("configs/my_new_match.yaml")
   ds = prepare_dataset_for_match(cfg)
   print("Manifest exported to:", ds.manifest_csv_path)
   ```

3. **Label Crops via CSV (No Python Dictionaries):**
   Open `data/crops/my_new_match/manifest.csv`, inspect the crop images in `data/crops/my_new_match/images/`, fill the `ground_truth_label` column (`Team_A`, `Team_B`, `Referee`, or `Unknown`), and import:
   ```python
   ds.import_labels_from_csv(ds.manifest_csv_path)
   ```

4. **Train, Evaluate, and Generate Annotated Video:**
   ```bash
   python -m src.pipeline.cli --config configs/my_new_match.yaml --compare-trackers
   ```

