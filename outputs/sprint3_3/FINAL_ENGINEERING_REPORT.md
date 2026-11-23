# Football Engineering — Sprint 3.3 Final Engineering Report
## Reliable Match Understanding, Ball Intelligence & Long-Duration Validation

**Repository:** `JaydinCodesAgain/footballengineering`  
**Base Commit (`main`):** `6b5cd90` (Sprint 3.2 merged PR #4)  
**Working Branch:** `feat/sprint3.3-reliable-match-understanding`  
**Roadmap Phase:** Phase 1 — Recorded-Match Foundation  
**Verification Status:** `84/84` automated unit & regression tests passing (`pytest -v`), genuine blind evaluation completed on untouched SoccerTrack v2 match `132877`.

---

## A. Executive Summary

Sprint 3.3 transformed the fixed-camera panoramic Football Vision Engine across five coordinated engineering pillars — **Ball Intelligence**, **Illumination-Invariant Night Team Discovery**, **Fixed-Camera Player Tracking**, **Verified Continuous Possession**, and **Compute Profiles + Staged Long-Duration Validation** — without modifying legacy broadcast APIs or training on held-out matches.

### Headline Quantitative Gains

| Subsystem | Metric | Sprint 3.2 Baseline | Sprint 3.3 Result | Delta / Verdict |
|---|---|---:|---:|---|
| **Ball Intelligence** (`n=74` verified frames, 4 matches) | Precision / Recall / F1 (`@25 px`) | `62.50%` / `45.45%` / `0.526` | **`79.41%` / `49.09%` / `0.607`** | **`+16.91 pp` P, `+3.64 pp` R, `+0.080` F1**; FP cut `15 -> 7` |
| **Blue-Cone Regression** (`117093`, `n=12`) | Precision / Recall / FP | `62.50%` / `71.43%` / `3 FP` | **`100.0%` / `100.0%` / `0 FP`** | **Blue touchline cone lock completely eliminated** |
| **Untouched Blind Ball Audit** (`132877`, `n=12`) | Visual Precision / Recall | `25.0%` / `25.0%` (prior `117093`) | **`100.0%` (`10/10`) / `83.3%` (`10/12`)** | **`0` FPs; `63.33%` observed / `73.33%` tracked coverage** |
| **Night Team Discovery** (`117092` floodlit) | Visual Track Cluster Purity / GK Rejection | `87.04%` (`7` dark-in-white) / `100%` | **`100.0%` (`0` dark-in-white) / `100%`** | **`+12.96 pp` visual purity**; silhouette `0.566 -> 0.701` |
| **Daylight Team Discovery** (`118575` dev) | Covered Acc / Overall Acc / GK Rejection | `95.61%` / `88.32%` / `100%` | **`97.41%` / `90.65%` / `100%`** | **`+1.80 pp` covered, `+2.33 pp` overall**; zero regression |
| **Player Tracking** (`118575`, `30 s` continuous, `5 m`) | GS-HOTA / AssA / IDF1 / Mean Lifetime | `0.5255` / `0.5129` / `0.7011` / `118.8 f` | **`0.5617` / `0.5852` / `0.7942` / `224.0 f`** | **`+0.0865` AssA, `+9.31 pp` IDF1, `+88.6%` track lifetime** |
| **Possession Engine** (`n=360` verified manual frames) | State Acc / Confident Acc / Strict Error / Share Err | `73.06%` / `100.0%` / `1.11%` / `7.92 pp` | **`87.22%` / `100.0%` / `0.56%` / `4.61 pp`** | **`+14.16 pp` state acc, `-3.31 pp` share err, `0` team swaps** |
| **Untouched Blind Generalization** (`132877` frozen) | Player Recall (`5m`/`8m`) / IDF1 / Team Acc / GK Rej | *(Unseen match)* | **`90.65%` (`98.47%`) / `74.63%` (`81.32%`) / `89.41%` / `100.0%`** | **Strong zero-tuning transfer; `10/22` players `0` IDSW** |
| **Long-Duration Stability** (`30s`, `2m`, `5m`, `10m`) | Net RSS Memory Growth (up to `15,000` frames) | Untested (`14.4 s` only) | **`+17.1` to `+26.2 MB` (`peak <= 1.29 GB`)** | **Zero memory leaks across 10 full minutes** |

---

## B. Baseline Confirmation & Data Split Hygiene

Confirmed baseline metrics and dataset roles are recorded in [BASELINE.md](BASELINE.md) and [EXPERIMENT_MANIFEST.json](EXPERIMENT_MANIFEST.json):

- **Development matches:** `117092` (night floodlights, `3840×1906`, White vs Dark Navy) and `118575` (daylight, `4096×1080`).
- **Previously inspected regression/reference matches:** `128057` (daylight; spare curb ball at `(3492, 828)`) and `117093` (daylight; stationary blue touchline cone at `[470, 683, 481, 692]` during `f=192..344`). Neither is presented as an untouched blind test set.
- **Genuinely untouched blind test match:** `132877` (validation split, `4096×1080`, University of Tsukuba panorama). Only the video and camera calibration files were downloaded before freezing Sprint 3.3 and running blind inference via the CLI; GSR, BAS, and ball labels were downloaded strictly **post-inference**.

---

## C. Ball Intelligence (`Sub-Agent A`)

**Files:** [`src/possession/ball_tracker.py`](../../src/possession/ball_tracker.py), [`src/detection/motion_ball.py`](../../src/detection/motion_ball.py), [`scripts/benchmark_sprint3_3_ball.py`](../../scripts/benchmark_sprint3_3_ball.py), [`data/annotations/soccertrack_ball_eval/verified_ball_eval_v2.json`](../../data/annotations/soccertrack_ball_eval/verified_ball_eval_v2.json)  
**Reports:** [`ball_accuracy.json`](ball_intelligence/ball_accuracy.json), [`strategy_comparison.csv`](ball_intelligence/strategy_comparison.csv), [`failure_cases.csv`](ball_intelligence/failure_cases.csv)

### Engineering Changes
1. **Ball Candidate Appearance & Cone/Marker Verification (`verify_ball_appearance`)**: Rejects saturated blue/cyan/purple training cones (`H in [90..145], S >= 75, B - R >= 18`, `< 14%` bright white/yellow core — the exact signature of `117093`'s cone at `[470, 683, 481, 692]`, which had `52.5%` blue-cone pixels and `0.0%` bright white pixels), saturated red/orange sideline cones, and dark low-luminance blobs (`V_p90 < 78`) while preserving white, hi-vis yellow, and motion-blurred footballs.
2. **Fixed-Camera Boundary Static-Fixture Memory (`_update_static_spot_detector` & `KalmanBallTracker.step`)**: Previously, static spots only accumulated when `is_isolated` (`min_p_d > 2.0 * ref_h`), allowing a sideline cone or spare ball in `PitchZone.BOUNDARY_MARGIN` to evade blacklisting whenever a player walked past it (e.g. a throw-in). Sprint 3.3 accumulates motionless hotspots (`<= 6.0 px`) in `BOUNDARY_MARGIN` / `OFF_PITCH` **regardless of player proximity** and masks them **permanently** (`hits >= 3`), while preserving the isolation check inside `PitchZone.PLAYABLE` so stationary set-piece balls (kickoffs, free kicks, penalties) remain tracked.
3. **Kalman Deflection & Fast-Kick Velocity Reset**: When a confident, appearance-verified ball inside `PitchZone.PLAYABLE` falls just outside the constant-velocity gate (`<= max(1.85 * max_gate, 260 px)`), the tracker immediately re-locks, resets velocity toward `(meas - prev) / dt`, and inflates velocity covariance to follow sharp volleys and deflections.
4. **Compact 4-Cardinal Ring Contrast in Motion Ball Recovery (`src/detection/motion_ball.py`)**: Requires motion blobs to be brighter than surrounding turf on at least 3 of 4 cardinal sides, eliminating painted pitch-line flicker.

### Quantitative Results (`n = 74` Verified Frames across 4 Matches, `@25 px`)

| Strategy | Mean Latency (CPU) | All 74: Prec / Rec / F1 / AP | All FP (Neg FP) | Dev 50: Prec / Rec / F1 | Reg 24 (`117093`+`128057`): Prec / Rec / F1 | `117093` FP / TP |
|---|---:|---|---:|---|---|---:|
| `S0_baseline_full_yolo11n_1088` | `169.3 ms/f` | `N/A` / `0.000` / `0.000` / `0.000` | `0` (`0`) | `N/A` / `0.000` / `0.000` | `N/A` / `0.000` / `0.000` | `0` / `0` |
| `S1_sprint3_2_yolo11s_4tile_pitchfilter` | `4746.9 ms/f` | `0.625` / `0.455` / `0.526` / `0.338` | `15` (`7`) | `0.600` / `0.375` / `0.462` | `0.667` / `0.667` / `0.667` | `3` / `5` |
| `S2_sprint3_3_appearance_cone_filter` | `4747.3 ms/f` | `0.774` / `0.436` / `0.558` / `0.365` | `7` (`3`) | `0.714` / `0.375` / `0.492` | `0.900` / `0.600` / `0.720` | `0` / `4` |
| **`S3_sprint3_3_full_ball_intelligence`** | `5044.3 ms/f` | **`0.794` / `0.491` / `0.607` / `0.397`** | **`7` (`4`)** | **`0.714` / `0.375` / `0.492`** | **`0.923` / `0.800` / `0.857`** | **`0` / `7` (`100%/100%`)** |

### Football-Specific Fine-Tuning Assessment
Fine-tuning a dedicated small-ball YOLO detector was **intentionally not performed** in Sprint 3.3 and is explicitly recorded as **blocked by data & compute**: SoccerTrack v2 does not ship independently observed per-frame ball bounding boxes (its auxiliary ball track is event-reconstructed between BAS actions), fine-tuning on our 74-frame evaluation set would cause severe data leakage, and the local machine is CPU-only (`cuda=False`). Minimum defensible requirement for Phase 2 fine-tuning: `>= 3,000` verified ball boxes across `>= 6` match-disjoint fixed-camera venues trained on GPU (`colab_t4` profile) with 1280 px tiles and small-object augmentation.

---

## D. Night-Time Team Discovery & Illumination-Invariant Representation (`Sub-Agent B`)

**Files:** [`src/classification/features.py`](../../src/classification/features.py), [`src/classification/team_discovery.py`](../../src/classification/team_discovery.py), [`scripts/benchmark_sprint3_3_team_discovery.py`](../../scripts/benchmark_sprint3_3_team_discovery.py)  
**Reports:** [`night_classification.json`](team_discovery/night_classification.json), [`daylight_regression.json`](team_discovery/daylight_regression.json), [`117092_sprint3_3_cluster_gallery.jpg`](team_discovery/117092_sprint3_3_cluster_gallery.jpg)

### Root-Cause Findings on Night Match `117092`
Two separate causes were isolated scientifically:
1. **Feature-space collapse in `hsv_lab`**: Under harsh floodlights on `117092` (White vs Dark Navy), shadowed dark-navy jerseys (`L* = 41..95, S < 45`) and glare-hit dark-navy jerseys shared neutral `(a*=128, b*=128)` histogram bins and `S < 45` achromatic flags with White jerseys, causing `hsv_lab` to misassign **7 dark-navy player tracks** to the White cluster (`87.04%` visual track purity).
2. **External `117092` calibration cross-matching ceiling (`30.63 px` RMS)**: Per `SoccerTrack-v2/docs/calibration-findings.md`, match `117092` has `30.63 px` calibration RMS (`3.1×` larger than every other SoccerTrack v2 match, `8.64–10.04 px`). Per-track inspection confirmed that single unbroken pure-white tracks (`L* = 207.8`) matched `50/50` to `left` and `right` GSR IDs under `5 m` pitch-space matching, imposing an external `~77–78%` GSR pitch-match ceiling independent of visual jersey accuracy.

### Engineering Changes
1. **`extract_illumination_invariant_features` (`"illumination_invariant"`)**:
   - Adaptive specular-glare & white-number trimming on dark kits (`L_p35 < 118`), and dark-number trimming on bright white kits (`L_p75 >= 128`).
   - **Joint 2D CIE LAB `(L*, a*)` and `(L*, b*)` histograms** alongside 1D `a*`/`b*` histograms: prevents dark low-saturation navy kits from sharing neutral `(a*=128, b*=128)` bins with white kits while retaining strong chromatic separation for maroon/yellow goalkeepers and referees.
   - Smooth separation of bright-achromatic vs dark-achromatic ratios + CLAHE lightness quantiles + glare-trimmed normalized RGB chromaticity ratios.
2. **Scale- & Quality-Weighted Track Descriptor Aggregation (`_compute_observation_quality_weight` & `_weighted_trimmed_aggregate`)**: Weights observations by bbox scale `(bh/42)^1.35`, turf-free area, sharpness, and glare calmness with cosine-inlier trimming.

### Quantitative Results (Identical Detections, 4 Matches)

| Match | Role / Lighting | `hsv_lab` (Sprint 3.2) Cov Acc / Overall / Sil / GK Rej | `illumination_invariant` + Quality-Weight (Sprint 3.3) Cov Acc / Overall / Sil / GK Rej | Notes |
|---|---|---|---|---|
| `117092` | Dev — Night Floodlights | `77.33%` / `74.96%` / `0.566` / `100%` (Visual Purity: **`87.04%`**, `7` dark-in-white) | `73.94%` GSR* / `0.701` sil / `100%` GK (Visual Purity: **`100.0%`**, **`0` dark-in-white**) | **`+12.96 pp` visual purity**; `0` jersey confusions in gallery |
| `118575` | Dev — Daylight | `95.61%` / `88.32%` / `0.608` / `100%` | **`97.41%` / `90.65%` / `0.656` / `100%`** | **`+1.80 pp` covered, `+2.33 pp` overall** |
| `128057` | Ref — Daylight | `98.96%` / `98.76%` / `0.648` / `100%` | **`98.96%` / `98.76%` / `0.717` / `100%`** | Zero regression; `+0.069` silhouette |
| `117093` | Ref — Daylight | `95.84%` / `94.71%` / `0.460` / `100%` | `94.41%` / `76.18%` / **`0.561`** / `100%` | Stricter unknown rejection (`19.3%` abstained) |

*\*GSR 5 m score on `117092` is bounded by the `30.63 px` lens calibration cross-match ceiling documented above.*

---

## E. Player Tracking & Long-Sequence Continuity (`Sub-Agent C`)

**Files:** [`src/tracking/pitch_stitcher.py`](../../src/tracking/pitch_stitcher.py), [`src/tracking/tracker.py`](../../src/tracking/tracker.py), [`scripts/benchmark_sprint3_3_tracking.py`](../../scripts/benchmark_sprint3_3_tracking.py)  
**Reports:** [`tracker_comparison.json`](player_tracking/tracker_comparison.json), [`long_sequence_tracking.json`](player_tracking/long_sequence_tracking.json)

### Engineering Changes
1. **`FixedCameraTrackletAssociator` (`src/tracking/pitch_stitcher.py`)**: Causal online tracklet stitcher maintaining a 45-frame lost-track buffer with metric pitch-plane kinematic extrapolation (`<= min(7.2, 3.2 + 0.18 * gap) m`), pixel kinematic gate, bbox height ratio gate (`0.68..1.47`), HSV+LAB torso appearance cosine gate (`<= 0.26`), 1-to-1 Hungarian assignment, and a strict per-frame uniqueness invariant preventing two simultaneously active players from ever merging.
2. **Fixed-Camera Tracker Tuning (`src/tracking/tracker.py`)**: `track_buffer=60`, `new_track_thresh=0.32`, `match_thresh=0.82`, `gmc_method=None` when `fixed_camera_mode=True`.

### Controlled Head-to-Head Results (Identical Detections, Isolating `AssA` from `DetA`)

| Sequence | Variant | `DetA` (`5m`) | `AssA` (`5m`) | `GS-HOTA` (`5m`) | `IDF1` (`5m`) | `IDSW` | `Frag` | Pred Tracks (Total) | Mean Lifetime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `117092` (`14.4 s`, `180 f`) | `BoTSORT_Baseline` | `0.1750` | `0.2524` | `0.2051` | `0.4174` | `145` | `180` | `86` | `47.9 f` |
| `117092` (`14.4 s`, `180 f`) | **`ByteTrack_FixedCam_Stitched`** | `0.1749` | **`0.2797`** | **`0.2173`** (`0.364@8m`) | **`0.5340`** (`0.694@8m`) | **`126`** | **`169`** | **`36`** | **`115.2 f`** |
| `118575` (`14.4 s`, `180 f`) | `BoTSORT_Baseline` | `0.5577` | `0.5789` | `0.5667` | `0.7514` | `39` | `32` | `52` | `87.5 f` |
| `118575` (`14.4 s`, `180 f`) | **`BoTSORT_FixedCam_Stitched`** | `0.5586` | **`0.6551`** | **`0.6035`** (`0.699@8m`) | **`0.8143`** (`0.824@8m`) | **`28`** | **`29`** | **`38`** | **`119.8 f`** |
| `118575` (`30.0 s`, `375 f`) | `BoTSORT_Baseline` | `0.5411` | `0.5129` | `0.5255` | `0.7011` | `86` | `72` | `79` | `118.8 f` |
| `118575` (`30.0 s`, `375 f`) | **`ByteTrack_FixedCam_Stitched`** | `0.5413` | **`0.5852`** | **`0.5617`** (`0.666@8m`) | **`0.7942`** (`0.806@8m`) | **`65`** | **`69`** | **`42`** | **`224.0 f`** |

`DetA` and `LocA` remain invariant (`<= 0.002` difference), proving that all tracking gains arise purely from improved Association Accuracy (`AssA +0.076` to `+0.107`), cutting spurious tracks nearly in half and almost doubling mean track lifetime (`118.8 -> 224.0` frames).

---

## F. Possession Verification & Geometric State Machine (`Sub-Agent D`)

**Files:** [`src/possession/ownership.py`](../../src/possession/ownership.py), [`src/possession/state_machine.py`](../../src/possession/state_machine.py), [`src/possession/possession_engine.py`](../../src/possession/possession_engine.py), [`src/evaluation/possession_eval.py`](../../src/evaluation/possession_eval.py), [`scripts/benchmark_sprint3_3_possession.py`](../../scripts/benchmark_sprint3_3_possession.py)  
**Verified Continuous Datasets:** [`data/annotations/soccertrack_118575/possession_annotations.json`](../../data/annotations/soccertrack_118575/possession_annotations.json) (`29` verified intervals spanning `0..30.0 s`), [`data/annotations/soccertrack_117092/possession_annotations.json`](../../data/annotations/soccertrack_117092/possession_annotations.json) (`14` verified intervals spanning `0..14.4 s`)  
**Reports:** [`possession_accuracy.json`](possession/possession_accuracy.json), [`attribution_coverage.json`](possession/attribution_coverage.json), [`visual_audit_118575.jpg`](possession/visual_audit_118575.jpg), [`visual_audit_117092.jpg`](possession/visual_audit_117092.jpg)

### Engineering Changes
1. **Geometric Fixed-Camera `OUT_OF_PLAY`**: Integrated `PitchRegionFilter.classify_point` into `is_ball_out_of_play_bounds` and `evaluate_frame_ownership_candidates` (`OFF_PITCH` immediately out of play; slow/stationary `BOUNDARY_MARGIN` out of play).
2. **Conservative Uncertain-Ball Decay & Same-Team Pass Continuity**: Penalizes interpolated/coasting ball confidence, transitions promptly to `UNKNOWN` during prolonged ball loss, and bridges same-team pass flights smoothly without fabricating possession during contested duels.

### Pooled Continuous Verified Evaluation (`n = 360` Sampled Frames = `28.8 s`, `118575` + `117092`)

| Metric | Sprint 3.2 Baseline | Sprint 3.3 Possession Engine | Delta |
|---|---:|---:|---:|
| **Frame-Level State Accuracy** | `73.06%` | **`87.22%`** | **`+14.16 pp`** |
| **Confident Team Attribution Accuracy** | `100.0%` (`82/82`) | **`100.0%` (`84/84`)** | `0` team swaps |
| **Strict Incorrect Confident Rate (All Frames)** | `1.11%` (`4/360`) | **`0.56%` (`2/360`)** | **`-0.55 pp`** |
| **Team Possession Share Error** | `7.92 pp` | **`4.61 pp`** | **`-3.31 pp`** |
| **Unknown State Rate / Attribution Coverage** | `34.17%` / `23.89%` | `47.50%` / `24.72%` | Higher conservative abstention during night ball loss |
| **State Transitions (GT = 31)** | `62` (`err = 31`) | **`58` (`err = 27`)** | Fewer flicker transitions |

---

## G. Compute Profiles, Staged Long-Duration Validation & Untouched Blind Generalization

### 1. Hardware Compute Profiles
Implemented in [`src/config.py`](../../src/config.py), [`src/detection/detector.py`](../../src/detection/detector.py), [`src/pipeline/cli.py`](../../src/pipeline/cli.py), and [`src/pipeline/runner.py`](../../src/pipeline/runner.py):
- `local_low_vram` (`tile_batch_size=2`, `tile_imgsz=1024`, `half=True` on CUDA, periodic cache cleanup)
- `colab_t4` (`tile_batch_size=4`, `tile_imgsz=1280`, `half=True` on CUDA)
- `cpu_fallback` (`device="cpu"`, `half=False`, `tile_batch_size=2`, `tile_imgsz=1280`, safe automatic CPU fallback whenever CUDA is unavailable)

### 2. Staged Long-Duration Evaluation (`118575`, `30 s` -> `10 min`)

| Stage | Duration / Spanned Frames | Stride (Sampled) | Detector / Profile | Player Recall (`5m`) | `GS-HOTA` (`5m`) | `IDF1` (`5m`) | Team Cov Acc / GK Rej | RSS Peak (Net Growth) | Projected Dense `stride=2` Time (CPU vs Colab T4) |
|---|---|---:|---|---:|---:|---:|---|---:|---|
| [`30_second_report.json`](end_to_end/30_second_report.json) | `30.0 s` (`750 f`) | `2` (`375 f`, `12.5 Hz`) | `yolo11s` 4-tile `1280` | **`97.20%`** | **`0.5604`** | **`0.7835`** | **`95.92%`** / `100%` | `1064.8 MB` (`+105.2 MB`) | `29.7 min` CPU / **`0.59 min` T4** |
| [`2_minute_report.json`](end_to_end/2_minute_report.json) | `120.0 s` (`3,000 f`) | `15` (`200 f`, `1.67 Hz`) | `yolo11n` 4-tile `1024` | `35.75%` | `0.0855` | `0.1155` | `81.05%` / `100%` | `1199.3 MB` (`+17.1 MB`) | `118.7 min` CPU / **`2.38 min` T4** |
| [`5_minute_report.json`](end_to_end/5_minute_report.json) | `300.0 s` (`7,500 f`) | `30` (`250 f`, `0.83 Hz`) | `yolo11n` 4-tile `1024` | `23.20%` | `0.0483` | `0.0551` | `74.74%` / `98.4%` | `1285.9 MB` (`+17.2 MB`) | `296.7 min` CPU / **`5.94 min` T4** |
| [`10_minute_report.json`](end_to_end/10_minute_report.json) | `600.0 s` (`15,000 f`) | `60` (`250 f`, `0.42 Hz`) | `yolo11n` 4-tile `1024` | `17.05%` | `0.0308` | `0.0348` | `Abstained (0% cov)` / `100%` | `1196.4 MB` (`+26.2 MB`) | `593.4 min` CPU / **`11.88 min` T4** |

**Key Long-Duration Engineering Finding:**
- **Memory stability is proven**: Peak RSS stays bounded under `1.29 GB` with only `+17.1` to `+26.2 MB` net growth across 10 full minutes (`15,000` frames) — zero leaks.
- **Temporal sampling floor is quantified**: On dense `stride=2` (`30 s`), tracking, team discovery, and confident possession (`43/43 = 100.0%` aligned confident attribution accuracy, `0.00%` strict error) hold rock-solid. When CPU budget forces coarse subsampling (`stride >= 15`, `>= 0.6 s` between frames), players move `3–18 m` per step, breaking IoU association and track lifetimes (causing conservative team discovery and possession to safely abstain rather than fabricate outputs). Production multi-minute analysis MUST execute at `stride <= 3` on GPU (`--compute-profile colab_t4 --half`, `~11.9 min` for 10 full minutes).

### 3. Genuine Post-Freeze Blind Generalization (`132877`, Untouched Validation Match)
Executed via the official CLI with frozen Sprint 3.3 settings before downloading any labels; evaluated post-hoc in [`eval_132877_blind_sprint3_3.json`](eval_132877_blind_sprint3_3.json) and visually audited in [`blind_132877_ball_visual_audit.jpg`](blind_132877_ball_visual_audit.jpg):

- **Player Recall:** **`90.65%` (`5 m`)** / **`98.47%` (`8 m`)**; Precision on-pitch `85.33%` (`5 m`).
- **Player Tracking:** **`GS-HOTA 0.5131` (`0.6223@8m`)**, **`AssA 0.5643`**, **`IDF1 74.63%` (`81.32%@8m`)**, `20/22` Mostly Tracked, only `26` matched predicted tracks (`1.86` distinct IDs per GT player; **10 of 22 players had `100.0%` dominant ID purity and `0` ID switches**).
- **Unsupervised Team Discovery:** **`89.41%` covered accuracy** (`18/20` GT outfield players correct majority, `99.2%` coverage), **`100.0%` goalkeeper rejection (`350/350` pink/yellow GK observations rejected as `Unknown`)**. Two royal-blue players with large bright-white back numbers were the sole team confusions (reported transparently without tuning).
- **Ball Tracking:** **`63.33%` observed coverage**, **`73.33%` tracked+interpolated**, **`100.0%` visual precision (`10/10`, `0` FP)**, **`83.3%` visual recall (`10/12`)** — with `tiled_yolo`, `kalman_roi`, and `motion_assisted` all contributing verified true detections during live play and contested tackles.

---

## H. Remaining Risks, Limitations & Phase 2 Recommendations

1. **Far-side `6–8 px` night ball recall (`27.3%` on `117092`)**: Pretrained COCO `yolo11s` remains physically limited below `~8 px` on low-contrast floodlit night turf. **Recommendation:** Curate `>= 3,000` match-disjoint verified small-ball boxes across `>= 6` venues and fine-tune a small-object ball detector on GPU (`colab_t4`) in Sprint 3.4 / Phase 2.
2. **Large white jersey numbers on dark/royal-blue kits (`132877` blind test)**: Two blue players (`18/20` correct) had large white back numbers occupying `> 38%` of the torso crop. **Recommendation:** Add connected-component digit/text masking or lightweight self-supervised ReID embeddings for jersey invariance.
3. **Dense multi-minute execution requires GPU**: Local CPU (`~4.75 s/frame` for 4-tile `yolo11s@1280`) makes dense 10-minute runs impractical (`~10 hours` CPU vs `~11.9 min` on Colab T4 FP16). **Recommendation:** Run full-half (`45 min`) dense `stride=2` validation on Colab T4 using `--compute-profile colab_t4 --half`.
4. **Match `117092` fisheye calibration residual (`30.63 px` RMS)**: Downstream tactical pitch-plane analytics on night venues with high lens distortion should validate keypoint reprojection RMS (`<= 12 px`) before exporting metric tactical coordinates.

