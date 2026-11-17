# Football AI Engineering — Detection, Classification, Tracking & Possession Pipeline

An end-to-end computer vision and sports analytics pipeline for broadcast football footage using **Ultralytics YOLO11**, **OpenCV**, **Scikit-learn**, **ByteTrack**, **BoT-SORT**, and a **2D Kalman Ball Tracker + Possession State Machine**.

---

## Vision: accessible grassroots football intelligence

The long-term goal is a reliable local-first system that can analyse a **previously unseen grassroots match** from a fixed camera without manually training jersey labels for that match.

```text
Fixed tactical camera / recorded match
    -> player and ball observations
    -> automatic Team_A / Team_B visual discovery
    -> stable anonymous player tracks
    -> calibrated pitch positions in metres
    -> confidence-aware possession and spatial analysis
    -> coach-friendly post-match evidence
```

The project deliberately starts with post-match analysis rather than promising live tactical intelligence too early. A stable elevated camera, clear footage and measured accuracy are more valuable than a large dashboard full of unsupported statistics.

### What “good” looks like

- Upload an unfamiliar match video without a manually labelled crop gallery or hard-coded kit colours.
- Discover two dominant playing-team appearances as `Team_A` and `Team_B`; do **not** pretend clustering can identify club names.
- Keep officials, goalkeepers, substitutes, poor crops and ambiguous observations as `Unknown` unless there is sufficient evidence.
- Track real players and the real ball, explicitly distinguishing `observed`, short `interpolated`, and `lost` ball states.
- Estimate possession only over confidently attributed intervals and expose coverage alongside every statistic.
- Convert fixed-camera player foot positions from pixels to pitch metres when a valid venue calibration is supplied.
- Measure improvements against independently reviewed reference data; never use model outputs as their own ground truth.

### Delivery plan

| Phase | Outcome | Evidence required before moving on |
|---|---|---|
| 1. Recorded-match foundation | Unseen-match team discovery, anonymous track labels, ball tracking, annotated video, confidence-aware possession | Visual review plus a small independently labelled sample per match |
| 2. Venue-calibrated analysis | Pitch-coordinate exports, distance/speed, territory, team width/depth and compactness for fixed cameras | Calibration reprojection checks and position-error measurements |
| 3. Grassroots robustness | Models and thresholds validated across local pitches, lighting, kits, weather and camera placements | Match-held-out ball/team/possession evaluation |
| 4. Near-live operation | Local/edge inference with an intentional delay and health telemetry | Stable end-to-end reliability at a real venue |
| 5. Live intelligence | Low-latency sideline support and optional roster integration | Proven accuracy, privacy controls and operational monitoring |

### Accuracy and privacy principles

- Coverage is not recall. A frame containing *something* ball-like is not proof that the real ball was detected.
- `Unknown` is a valid answer. Rejecting unreliable observations is preferable to fabricating possession or player identity.
- The system maintains identities primarily within continuous camera segments. Cross-cut and cross-match re-identification are separate, harder problems.
- Player names and face recognition are not requirements for the initial product. Anonymous per-match tracks deliver useful analysis with less risk.
- Obtain appropriate consent and apply the applicable child-data, privacy and venue-recording requirements before recording grassroots matches.

---

## Key Capabilities

1. **Player & Football Detection (`src/detection/`)**
   - Pre-ByteTrack raw ball capture hook that prevents ByteTrack's two-frame IoU activation gate (`is_activated`) from discarding fast-moving $14 \times 14\text{ px}$ footballs.
   - Multi-strategy football detection benchmark (`benchmark_ball_detection_strategies`) comparing full-frame YOLO11n (`imgsz=1088`), high-resolution (`imgsz=1280`), decoupled low-confidence with geometric/torso false-positive filtering, 2-tile horizontal pitch slicing (`detect_balls_tiled`), and Kalman ROI recovery (`detect_ball_in_roi`).
   - Ground-truth bounding-box annotation store (`DetectionAnnotationStore`) with 45+ representative frame sampling (`sample_representative_frame_numbers`), interactive HTML5 canvas annotator (`detection_annotator.html`), and strict manual provenance enforcement (`verified=True, source="manual"`).

2. **Match-Agnostic Jersey Classification (`src/classification/`)**
   - Generic internal labels (`Team_A`, `Team_B`, `Referee`, `Unknown`) mapped via YAML configurations (`configs/norway_wales.yaml`, `configs/chelsea_sunderland.yaml`) with cross-match safety guards.
   - **Fresh-Clone Dataset Integrity & Deterministic Regeneration (`validate_integrity`, `regenerate_missing_crops`):** Automatically detects missing, corrupted, and dimension-mismatched gitignored crop PNGs and deterministically regenerates them by seeking to `frame_number` and slicing `torso_bbox` from the source video, verifying dimensions and appearance metrics (`grass_ratio`, `blur_score`) before preserving verified labels.
   - Six benchmarked feature extractors (`baseline_hsv`, `legacy_enhanced`, `hsv_lab`, `foreground_masked`, `dominant_colour`, `spatial_embedding`) and validation-only Unknown threshold calibration.

3. **Persistent Player & Ball Tracking (`src/tracking/`, `src/possession/ball_tracker.py`)**
   - Player tracking with **ByteTrack** (`bytetrack.yaml`) and **BoT-SORT** (`botsort.yaml`) plus `TemporalTeamStabilizer` (confidence-weighted voting, soft label locking, sustained-contradiction unlocking, and spatial-jump identity resets).
   - Dedicated `KalmanBallTracker` (4D state `[x, y, vx, vy]`) with player-proximity candidate scoring, static isolated clutter false-track breaking & spatial blacklisting (`false_track_associations`), player-scale relative box filtering (`dynamic_max_size = 0.38 * median_player_height`), camera-cut detection resets (combining HSV histogram correlation, L1 diff, and player-height scale jumps), and explicit provenance (`observed` vs `interpolated` vs `lost`).

4. **Ball-to-Player Ownership & Team Possession Analytics (`src/possession/`)**
   - Scale-normalized foot-to-ball distance ($d_{\text{norm}} = d_{\text{px}} / \max(30, h_{\text{player}})$) accounting for broadcast perspective scale, plus automatic rejection of non-tactical player close-up shots (`UNKNOWN`).
   - Explicit 6-state `PossessionStateMachine`: `TEAM_A_CONTROL`, `TEAM_B_CONTROL`, `CONTESTED`, `LOOSE_BALL`, `UNKNOWN`, and `OUT_OF_PLAY`.
   - Temporal hysteresis (`min_control_frames`), same-team pass continuity (`max_pass_retention_frames`), referee exclusion, conservative goalkeeper/`Unknown` handling, and cut-separated continuous camera segment breakdown (`camera_segments`).
   - Separates **Attributed Possession Share** ($\frac{T_A}{T_A + T_B}$) from **Total Analysed Time Distribution** across all 6 states, with interactive HTML5 interval annotation (`possession_annotator.html`) and live broadcast scoreboard overlays on the output MP4.

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

5. **Analyse an unfamiliar match with automatic team discovery:**
   ```powershell
   python -m src.pipeline.cli `
     --video "data/raw/new_match.mp4" `
     --auto-teams `
     --enable-possession `
     --max-frames 1500
   ```

   This mode adapts only from the supplied video: it collects quality-valid torso observations, aggregates appearance by player track, discovers two dominant visual groups and keeps ambiguous players as `Unknown`. It does not use a match crop manifest, team names or configured kit colours.

   For a fixed grassroots camera, add a calibration file to export player positions in pitch metres:

   ```powershell
   --pitch-calibration "configs/venue_pitch_calibration.json"
   ```

   A calibration file contains matching image pixels and real pitch landmarks:

   ```json
   {
     "pitch_length_m": 105.0,
     "pitch_width_m": 68.0,
     "image_points_px": [[120, 180], [1800, 220], [1900, 920], [40, 900]],
     "pitch_points_m": [[0, 0], [105, 0], [105, 68], [0, 68]]
   }
   ```

   Use a calibration only while the camera position and framing are unchanged. It is not valid across broadcast cuts, zooms or a moved tripod.
