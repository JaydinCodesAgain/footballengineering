# Football AI Engineering — Detection, Classification, Tracking & Possession Pipeline

An end-to-end computer vision and sports analytics pipeline for broadcast football footage using **Ultralytics YOLO11**, **OpenCV**, **Scikit-learn**, **ByteTrack**, **BoT-SORT**, and a **2D Kalman Ball Tracker + Possession State Machine**.

---

## Key Capabilities

1. **Player & Football Detection (`src/detection/`)**
   - Pre-ByteTrack raw ball capture hook that prevents ByteTrack's two-frame IoU activation gate (`is_activated`) from discarding fast-moving $14 \times 14\text{ px}$ footballs.
   - Multi-strategy football detection benchmark (`benchmark_ball_detection_strategies`) comparing full-frame YOLO11n (`imgsz=1088`), high-resolution (`imgsz=1280`), decoupled low-confidence with geometric/torso false-positive filtering, 2-tile horizontal pitch slicing (`detect_balls_tiled`), and Kalman ROI recovery (`detect_ball_in_roi`).
   - Ground-truth bounding-box annotation store (`DetectionAnnotationStore`) with strict manual provenance enforcement (`verified=True, source="manual"`).

2. **Match-Agnostic Jersey Classification (`src/classification/`)**
   - Generic internal labels (`Team_A`, `Team_B`, `Referee`, `Unknown`) mapped via YAML configurations (`configs/norway_wales.yaml`, `configs/chelsea_sunderland.yaml`) with cross-match safety guards.
   - **Fresh-Clone Dataset Integrity & Deterministic Regeneration (`validate_integrity`, `regenerate_missing_crops`):** Automatically detects missing gitignored crop PNGs and deterministically regenerates them by seeking to `frame_number` and slicing `torso_bbox` from the source video, verifying dimensions and appearance metrics (`grass_ratio`, `blur_score`) before preserving verified labels.
   - Six benchmarked feature extractors (`baseline_hsv`, `legacy_enhanced`, `hsv_lab`, `foreground_masked`, `dominant_colour`, `spatial_embedding`) and validation-only Unknown threshold calibration.

3. **Persistent Player & Ball Tracking (`src/tracking/`, `src/possession/ball_tracker.py`)**
   - Player tracking with **ByteTrack** (`bytetrack.yaml`) and **BoT-SORT** (`botsort.yaml`) plus `TemporalTeamStabilizer` (confidence-weighted voting, soft label locking, sustained-contradiction unlocking, and spatial-jump identity resets).
   - Dedicated `KalmanBallTracker` (4D state `[x, y, vx, vy]`) with confidence-distance gating, implausible jump rejection, multi-frame re-acquisition for long passes, camera-cut detection resets, and explicit provenance (`observed` vs `interpolated` vs `lost`).

4. **Ball-to-Player Ownership & Team Possession Analytics (`src/possession/`)**
   - Scale-normalized foot-to-ball distance ($d_{\text{norm}} = d_{\text{px}} / \max(30, h_{\text{player}})$) accounting for broadcast perspective scale.
   - Explicit 6-state `PossessionStateMachine`: `TEAM_A_CONTROL`, `TEAM_B_CONTROL`, `CONTESTED`, `LOOSE_BALL`, `UNKNOWN`, and `OUT_OF_PLAY`.
   - Temporal hysteresis (`min_control_frames`), same-team pass continuity (`max_pass_retention_frames`), referee exclusion, and conservative goalkeeper/`Unknown` handling.
   - Separates **Attributed Possession Share** ($\frac{T_A}{T_A + T_B}$) from **Total Analysed Time Distribution** across all 6 states, with live broadcast scoreboard overlays on the output MP4.

---

## Project Structure

```text
footballengineering/
├── configs/
│   ├── chelsea_sunderland.yaml
│   └── norway_wales.yaml
├── data/
│   ├── annotations/
│   └── crops/
├── docs/
│   ├── EVALUATION_PROTOCOL.md
│   └── TECHNICAL_AUDIT.md
├── notebooks/
│   └── football_engineering.ipynb
├── outputs/
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── classification/
│   ├── detection/
│   ├── evaluation/
│   │   ├── classification_eval.py
│   │   ├── detection_eval.py
│   │   ├── possession_eval.py
│   │   ├── report.py
│   │   └── tracking_eval.py
│   ├── pipeline/
│   │   ├── cli.py
│   │   ├── legacy_migration.py
│   │   └── runner.py
│   ├── possession/
│   │   ├── ball_tracker.py
│   │   ├── metrics.py
│   │   ├── ownership.py
│   │   ├── possession_engine.py
│   │   └── state_machine.py
│   ├── tracking/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

---

## Fresh-Clone Setup & Reproducibility

1. **Install Dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Place Match Videos:**
   Because raw broadcast `.mp4` files and extracted crop PNGs (`data/crops/*/images/`) are excluded via `.gitignore`, place the match videos in the project root or `data/raw/`:
   - `Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4`
   - `27.mp4` (Chelsea vs Sunderland)

   On the first run after cloning, `JerseyCropDataset.regenerate_missing_crops` automatically validates `manifest.json`, slices the exact `torso_bbox` from each recorded `frame_number`, verifies crop dimensions and appearance consistency, and restores `data/crops/<match_id>/images/*.png` while preserving all manually verified labels.

3. **Run the Offline Unit Test Suite:**
   ```bash
   python -m pytest -v
   ```

4. **Run the End-to-End Pipeline with Possession Analytics:**
   ```bash
   # Norway vs Wales
   python -m src.pipeline.cli \
     --config configs/norway_wales.yaml \
     --max-frames 300 \
     --enable-possession \
     --compare-trackers

   # Chelsea vs Sunderland
   python -m src.pipeline.cli \
     --config configs/chelsea_sunderland.yaml \
     --max-frames 300 \
     --enable-possession \
     --compare-trackers
   ```
