# Technical Audit — Football AI Engineering Pipeline

## 1. Workspace Inventory & Initial State

An inspection of the project root (`c:\Users\jmorrison\projects\footballengineering`) identified three initial files:

| File | Size | Role |
|---|---|---|
| `footballengineering (2).py` | 49,936 bytes (2,186 lines) | Exported Google Colab script combining two match experiments (Chelsea vs Sunderland and Norway vs Wales) into a single procedural script. |
| `27.mp4` | 27,934,942 bytes | Broadcast footage for Match 1 (Chelsea vs Sunderland). |
| `Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4` | 22,824,349 bytes | Broadcast highlights footage for Match 2 (Norway vs Wales). |

Prior to refactoring, no modular package structure (`src/`, `configs/`, `tests/`, `data/`), serialized model checkpoints, or persisted evaluation manifests existed. All crops and model state lived ephemerally in notebook RAM.

---

## 2. Entry Points & Colab Notebook Structure

`footballengineering (2).py` is a linear concatenation of two notebook sessions:

1. **Part 1 — Chelsea vs Sunderland (Lines 1–947):**
   - **Cells 1–3 (Lines 13–246):** Interactive video upload (`google.colab.files.upload()`), YOLO11n person detection (`classes=[0]`), torso extraction (`extract_jersey`), median BGR colour calculation, and 3-cluster `KMeans` unsupervised clustering.
   - **Cells 4A–4B (Lines 247–370):** Display of an 80-crop reference gallery (`candidate_crops`) and supervised training of a 3-NN cosine classifier (`KNeighborsClassifier`) using HSV 2D histograms (`16x8` Hue-Saturation bins) mapped via hardcoded gallery indices (`label_to_indices`).
   - **Cells 5–7 (Lines 371–683):** End-to-end video tracking with `bytetrack.yaml` (`classes=[0, 32]`), 7-frame majority voting (`player_history`), OpenCV annotation (`soccer_analyzed.mp4`), and detection count plots.
   - **Cells 8–11 (Lines 684–947):** Collection of `eval_crops` from a second uploaded video (`Norway vs Wales`), manual index labelling (`true_labels` with keys `"Norway"`, `"Wales"`, `"REF"`, `"Other"`), and evaluation against the still-in-memory `team_model` trained on `"Chelsea"` and `"Sunderland"`.

2. **Part 2 — Norway vs Wales (Lines 948–2186):**
   - **Cells 12A–12D (Lines 952–1225):** Re-imports libraries, uploads the Norway vs Wales video, redefines `extract_jersey` and `jersey_features` (adding an HSV green grass mask), extracts `train_crops` (first 40% of video duration) and `eval_crops` (after 55% of duration), and trains a 3-NN cosine classifier with an uncalibrated distance rejection threshold (`UNKNOWN_DISTANCE = 0.45`).
   - **Cells 12E–12F (Lines 1227–1413):** Runs ByteTrack and writes `norway_wales_analyzed.mp4`, then prints classification accuracy from a stale `accuracy` variable computed earlier in Cell 10.
   - **Cells 12G–13 (Lines 1414–1637):** Samples 12 frames from the final 45% of the match, displays plotted YOLO predictions, and computes precision/recall/F1 from a hardcoded dictionary of visual counts (`GROUND_TRUTH`) rather than bounding-box IoU matching.
   - **Cells 14–15 (Lines 1638–1874):** Compares `baseline_features` (18x8 HS histogram) against `enhanced_jersey_features` (weighted concatenation of 18x8 HS histogram, 12x12 LAB `ab` histogram, and 12-bin V histogram) on `train_labels` (24 samples) and `eval_labels` (30 samples).
   - **Cells 16–17 (Lines 1875–2186):** Runs a 45-second tracking diagnostic pass (`tracking_diagnostic.mp4`) with 9-frame history (`MIN_VOTES = 3`, `MIN_AGREEMENT = 0.67`) and provides an empty template (`PLAYER_ID_AUDIT`) for manual tracking ID continuity checks.

---

## 3. Duplicate Functions, Stale Variables & Inconsistent Configurations

1. **`extract_jersey` (Defined twice with different crop ratios and height thresholds):**
   - **Line 66:** `MIN_PLAYER_HEIGHT = 25`, horizontal crop `20%–80%`, vertical crop `20%–55%`.
   - **Line 996:** Minimum height `30`, horizontal crop `25%–75%`, vertical crop `18%–55%`.
   - *Weaknesses:* Neither version checks for image blur, excessive pitch-green background ratio, or inter-player torso occlusion.
2. **`jersey_features` / `enhanced_jersey_features` (Defined three times):**
   - **Line 298:** 2D HSV histogram (`[16, 8]` bins over H and S), no grass mask, no brightness channel (`V`).
   - **Line 1019 (`baseline_features`):** 2D HSV histogram (`[18, 8]` bins over H and S) with a fixed green mask. Ignores Value (`V`) and Lightness (`L`).
   - **Line 1693 (`enhanced_jersey_features`):** Center sub-crop resized to `48x64`, combining L1-normalized `0.50 * hs_hist + 0.35 * ab_hist + 0.15 * brightness_hist`.
3. **`TEAM_COLOURS` (Defined three times with conflicting mappings):**
   - **Line 361:** Chelsea (Magenta), Sunderland (Cyan), REF (Yellow), Other (Grey).
   - **Line 1220:** Norway (Red: `0, 0, 255`), Wales (Blue: `255, 0, 0`), REF (Yellow: `0, 255, 255`), Unknown (Grey).
   - **Line 1893:** Swaps team colours in Cell 16 — `"Other": (0, 0, 255)`, `"REF": (255, 0, 0)`, `"Wales": (0, 255, 255)`, `"Norway": (160, 160, 160)` (rendering Norway in grey, identical to `Unknown`).
4. **Stale Variables Across Notebook Cells:**
   - Cell 12F (`Line 1391`) prints `f"Accuracy: {accuracy:.2%}"` and `len(y_true)` directly after running the Norway vs Wales video tracker, but `accuracy` and `y_true` were last assigned in Cell 10 (`Line 865`), reporting stale cross-match evaluation numbers.

---

## 4. Empirical Root-Cause Diagnosis of the Norway–Wales Classification Failure (16.67% / 10.00%)

By extracting the exact crops from `Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4` (`data/crops/norway_wales/legacy_train_gallery.png` and `legacy_val_gallery.png`), we verified five compounding root causes that produced the **16.67% baseline / 10.00% enhanced** failure and its specific confusion pattern (**8/10 Norway → Unknown, 13/14 Wales → REF, 4/4 REF → Unknown**):

### Cause 1: Pre-Match Anthem Jacket Contamination in `train_crops[0..29]`
- The first ~10 seconds (`frames 0–256`) of `Norway vs Wales _ HIGHLIGHTS _ UEFA Nations League.mp4` show the pre-match tunnel walkout and national anthem line-up, during which Norway players wear **black `KIWI` anthem jackets** over their kits.
- Actual match play—where Norway wears **red shirts with a dark-blue/white Nordic cross** (`#30 ØDEGAARD` at `frame 272` onwards)—begins on `page=1` (`indices 30–89`) of `train_crops`.
- Because `show_gallery(train_crops, page=0)` (`Line 1125`) only displayed indices `0–29`, all 15 `"Norway"` training indices (`#1, #2, #3, #5, #12, #15, #18, #20, #21, #23`, etc.) were sampled from the pre-match segment where Norway wore **black jackets**.
- During validation (`eval_crops`, `frames 1632+`), Norway wore **red match shirts**, which had a cosine distance $> 0.45$ from the black anthem jackets, yellow Wales kits, and cyan referee kits—causing **Norway crops to be systematically rejected as `Unknown`**.

### Cause 2: Inverted Class Labels & Volatile RNG Permutation Indices
- In `Norway vs Wales`, Wales wears **yellow away kits** (e.g. `#29` visibly reads `"DAVIES 11"` on a yellow shirt; `#10, #17, #26` display the Welsh crest on yellow shirts), while the UEFA Referee wears a **cyan / light-blue Macron official shirt** (`#0, #4, #6, #7, #28` with the UEFA badge and holding the match ball).
- In `train_labels` (`Lines 1130–1143`):
  - `train_labels["Wales"] = [0, 4, 6, 7]` actually points to 4 crops of the **cyan Referee** holding the ball in the tunnel.
  - `train_labels["REF"] = [19, 24, 26, 28, 29]` actually points to 4 crops of **yellow Wales players** (`#19, #24, #26, #29 DAVIES`) plus 1 cyan Referee (`#28`).
- Because `rng.permutation(len(boxes))[:3]` (`Line 1083`) was executed inside the live detection loop without saving crops to disk, `eval_labels` (`Lines 1145–1161`) suffered severe index drift: 18 of the 30 validation indices pointed to a different class than their dictionary key. Combined with yellow Wales kits being trained as `"REF"`, validation Wales crops matched `"REF"` templates.

### Cause 3: Cross-Match Label Incompatibility (Cells 8–10)
- In Cell 10 (`Lines 796–860`), `team_model` trained on `["Chelsea", "Sunderland", "REF", "Other"]` was evaluated directly against `true_labels` keyed by `["Norway", "Wales", "REF", "Other"]`.

### Cause 4: Feature Normalization & LAB Gamut Compression
- In `enhanced_jersey_features` (`Line 1693`), `hs_hist` (144 bins), `ab_hist` (144 bins), and `brightness_hist` (12 bins) were L1-normalized (`sum = 1`) rather than L2-normalized (`||v||_2 = 1`) before computing L2 cosine distance.
- Furthermore, `ab_hist` binned OpenCV 8-bit LAB `a` and `b` channels uniformly across `[0, 256]`, where neutral `128` sits in the center and >90% of natural pixels fall inside `[106, 149]`, collapsing the 144-bin LAB histogram into 4 central bins.

### Cause 5: Uncalibrated Fixed Rejection Threshold (`UNKNOWN_DISTANCE = 0.45`)
- Applying a static cosine threshold of `0.45` across feature spaces of different dimensionality (144-d vs 300-d) without validation calibration caused either excessive false rejections (`Unknown`) or false acceptances of goalkeepers and occluded background crops.

---

## 5. Quantitative Impact of Fixes

| Match & Split | Pipeline Configuration | Overall Accuracy | Selective Accuracy | Macro F1 (All) | Positive Macro F1 | Coverage |
|---|---|---|---|---|---|---|
| **Norway vs Wales (Val, n=30)** | Legacy Uncorrected (`baseline_hsv`, `t=0.45`) | 10.00% (16.67% Colab) | 9.09% | 10.71% | 7.62% | 36.67% |
| **Norway vs Wales (Val, n=30)** | Legacy Uncorrected (`legacy_enhanced`, `t=0.45`) | 13.33% (10.00% Colab) | 18.18% | 14.55% | 9.33% | 36.67% |
| **Norway vs Wales (Val, n=30)** | **Audited + `foreground_masked` (Calibrated `t=0.294`)** | **93.33%** | **100.00%** | **91.86%** | **93.90%** | **73.33%** |
| **Norway vs Wales (Val, n=30)** | **Audited + `spatial_embedding` (`t=0.350`)** | **93.33%** | **92.31%** | **90.00%** | **93.33%** | **86.67%** |
| **Norway vs Wales (Test, n=48)** | **Audited + `foreground_masked` (Held-out Test)** | **87.50%** | **88.64%** | **77.83%** | **90.30%** | **91.67%** |
| **Chelsea vs Sunderland (Val, n=47)** | **Audited + `spatial_embedding` (Calibrated `t=0.206`)** | **91.49%** | **100.00%** | **92.92%** | **97.22%** | **74.47%** |

---

## 6. Sprint 2.5 Empirical Diagnosis of Sprint 2 Ball Tracking & Possession Anomalies

A frame-by-frame audit of the Sprint 2 telemetry (`outputs/norway_wales_sprint2/possession_timeline.csv`, `ball_trajectory.csv`, `frame_observations.csv` and `outputs/chelsea_sunderland_sprint2/`) established the exact root causes of the two Sprint 2 possession anomalies:

### 6.1 Why Norway vs Wales (`frames 1620–1738`, `4.80s`) Reported Wales 100.0% / Norway 0.0%
1. **Unflagged Wide-to-Closeup Broadcast Camera Cut at `frame 1640` (`65.60s`):**
   - In `frames 1620–1638`, the broadcast camera showed wide-angle tactical play (`11` players visible, median player height `~105 px`), during which the ball tracker had locked onto a stationary pitch-side false positive at `bbox=[1209, 488, 1228, 506]` (`speed = 0.0 px/s`, `2.34` player heights from the nearest player), producing `LOOSE_BALL`.
   - At `frame 1640`, the broadcast cut to a tight full-screen close-up of **Wales Player `#29` (`bbox=[248, 4, 1037, 709]`, height `705 px` out of `720 px`)**. Because the background remained green grass and yellow kit, global HSV histogram correlation stayed above `0.58`, so `detect_camera_cut()` did not fire.
   - Inside that close-up (`frames 1640–1706`), YOLO detected an `82x82 px` patch near the bottom of the player's shorts/legs (`bbox=[582, 636, 665, 718]`) as `sports ball`. Because Player `#29`'s bounding box was `705 px` tall, the normalized distance `d_px / 705.0` was `0.036–0.289` (`< 0.70`), attributing `20 frames` (`1.60s`) of `TEAM_B_CONTROL` (Wales) during a non-tactical player close-up!

### 6.2 Why Chelsea vs Sunderland (`frames 0–88`, `3.60s`) Had `8.89%` Attribution Coverage & `55.56%` Loose Ball
1. **Stationary Touchline False Positive Lock (`frames 68–88`):**
   - In `frames 0–46`, the ball was only intermittently detected (`frames 4, 16, 40–46`), with `TEAM_A_CONTROL` (Chelsea) confirmed for `4 frames` (`0.32s = 8.89%`).
   - Starting at `frame 68` (`2.72s`), tiled low-confidence search (`c=0.18`) detected a stationary bright object near the left touchline (`bbox=[157, 313, 173, 325]`, `speed = 0.0–5.9 px/s`, `2.70` player heights away from all players). Because it never moved, the Kalman filter gated and locked onto `[157, 313, 173, 325]` for `11 consecutive frames` (`0.88s`), inflating `observed_coverage_pct` and `LOOSE_BALL` (`55.56%`).

### 6.3 Sprint 2.5 Engineering Fixes Implemented
1. **Player-Scale Relative Ball Box Filtering (`src/possession/ball_tracker.py`):**
   - `filter_ball_candidates()` now bounds maximum ball box dimensions to `dynamic_max_size = min(max_size, max(26, round(0.38 * median_player_height)))`, rejecting oversized `82x82 px` false positives on player shorts/kits.
2. **Wide-to-Closeup Camera Cut Detection & Non-Tactical Frame Rejection (`src/possession/ball_tracker.py`, `src/possession/ownership.py`):**
   - `detect_camera_cut()` now triggers on abrupt player-height scale jumps (`scale_ratio > 2.25` or `< 0.44`) in addition to HSV correlation and joint correlation/L1 difference.
   - `evaluate_frame_ownership_candidates()` explicitly classifies non-tactical close-up shots (`median_player_height > 0.44 * frame_height` or `max_player_area / frame_area > 0.18`) as `UNKNOWN` (`reason="non_tactical_closeup_frame"`).
3. **Static Isolated Clutter Detection & Spatial Blacklisting (`KalmanBallTracker`):**
   - Tracks `static_isolated_streak` whenever an observed ball candidate moves `< 2.5 px` while remaining `> 1.35` player heights away from all players. At `streak >= 6`, the tracker breaks the false track (`false_track_associations`), transitions to `LOST`, and blacklists the static coordinate (`blacklisted_static_zones`).
4. **Cut-Separated Continuous Camera Segment Reporting (`src/possession/metrics.py`):**
   - Partitions multi-shot highlight sequences at every detected camera cut into `camera_segments`, reporting per-shot start/end frames, duration, and team control without carrying state across cuts.

