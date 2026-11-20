# Sprint 3 vs Sprint 3.1 — Baseline Comparison Report

**Branch:** `feat/sprint3.1-accuracy-reliability` (branched from Sprint 3 commit `4c91813`)  
**Evaluation Rule:** Ground-truth accuracy metrics (computed strictly against independently human-verified labels) are separated cleanly from descriptive telemetry and coverage diagnostics. Missing ground-truth benchmarks are reported explicitly as validation blockers rather than substituted with coverage or model self-consistency.

---

## 1. Automatic Team Discovery (Independently Human-Verified Jersey Crops)

All clustering runs are strictly unsupervised: ground-truth labels are never exposed during fitting and are used solely post-hoc via optimal `Team_A`/`Team_B` permutation alignment.

### 1.1 Norway vs Wales (`n = 147` verified crops: 50 Norway, 49 Wales, 21 Referee, 27 Unknown/GK)

| Metric | Sprint 3 Baseline (`spatial_embedding` + `kmeans`, K=3) | Sprint 3.1 Selected (`hsv_lab` + `kmeans`, auto-K=4) | Delta |
|---|---:|---:|---:|
| Macro-F1 (covered Team_A/Team_B) | 0.9600 | **0.9748** | +0.0148 |
| Overall 3-Way Accuracy (incl. Unknown rejection) | 0.9184 | **0.9660** | +0.0476 |
| Macro-F1 (3-Way, incl. Unknown) | 0.9180 | **0.9658** | +0.0478 |
| Playing-Team Accuracy (incl. rejected misses, `n=99`) | 0.9192 | **0.9798** | +0.0606 |
| Playing-Team Coverage (`n=99`) | 0.9394 | **0.9899** | +0.0505 |
| Referee Misclassification Rate (`n=21`) | 0.0000 | **0.0000** | 0.0000 |
| Unknown/GK Misclassification Rate (`n=27`) | 0.2222 | **0.1111** | **-0.1111 (-50%)** |
| Adjusted Rand Index (3-Way) | 0.7690 | **0.9000** | +0.1310 |
| Seed Stability (5 seeds, 3-Way Macro-F1 std) | 0.0000 | **0.0000** | 0.0000 |
| Feature Extraction Latency (CPU) | 2.76 ms/crop | **1.73 ms/crop** | **-37.3% faster** |

### 1.2 Chelsea vs Sunderland (`n = 80` verified crops: 36 Sunderland, 29 Chelsea, 3 Referee, 12 Unknown/GK)

| Metric | Sprint 3 Baseline (`spatial_embedding` + `kmeans`, K=3) | Sprint 3.1 Selected (`hsv_lab` + `kmeans`, auto-K=3) | Delta |
|---|---:|---:|---:|
| Macro-F1 (covered Team_A/Team_B) | 0.9500 | **0.9898** | +0.0398 |
| Covered Accuracy (Team_A/Team_B predictions) | 0.9231 | **0.9833** | +0.0602 |
| Overall 3-Way Accuracy (incl. Unknown rejection) | 0.8875–0.9375 *(seed-sensitive)* | **0.9125** | Stabilised |
| Macro-F1 (3-Way, incl. Unknown) | 0.8250–0.9018 *(seed-sensitive)* | **0.8957** | Stabilised |
| Referee Misclassification Rate (`n=3`) | 0.0000 | **0.0000** | 0.0000 |
| Unknown/GK Misclassification Rate (`n=12`) | 0.3333 | **0.0833** | **-0.2500 (-75%)** |
| Cross-Team (`Team_A` ↔ `Team_B`) Swaps | 0 | **0** | 0 |
| Seed Stability (5 seeds, 3-Way Macro-F1 std) | Unstable (dropped to 0.825 on seeds 0/99) | **0.0000** | **100% deterministic** |
| Temporal Subset Stability (Half 1 / Half 2 3-Way F1) | Half 2 degraded under small sample | **0.8489 / 0.8546** | Balanced |
| Feature Extraction Latency (CPU) | 5.31 ms/crop | **0.78 ms/crop** | **-85.3% faster** |

### 1.3 Clustering Algorithm & Feature Ablation Highlights (`outputs/sprint3_1/team_discovery/team_metrics.csv`)

- **Agglomerative (`ward`) and GMM (PCA + full covariance):** Fixed severe Sprint 3 collapses (where average-linkage agglomerative scored ARI `0.08–0.34` and 519-d diagonal GMM misclassified `67–76%` of referees and `92–93%` of unknowns). With Sprint 3.1 `hsv_lab`, Ward agglomerative (`0.9658` / `0.8636`), PCA+full GMM (`0.9658` / `0.8812`), and HDBSCAN (`0.9590` / `0.8779`) all achieve strong separation; `kmeans` (`n_init=30`, auto-K) won overall on combined 3-way F1, zero seed variance, and Unknown rejection (`0.0833` Chelsea, `0.1111` Norway).
- **MobileNetV3-Small (`torchvision`, 576-d) Ablation:** Rejected after proving both **~100× slower** (`~81 ms/crop` CPU) and far less accurate (`ARI = 0.099` Norway, `ARI = -0.002` Chelsea) because ImageNet features prioritise pose/object invariance over jersey colour.

---

## 2. Ball Detection Filtering & Detector Benchmark

> **Validation Blocker:** `0` manually verified ground-truth ball bounding boxes (`verified=true, source=manual`) exist in `data/annotations/*`. Formal IoU AP@50, precision, recall, and pixel center-error metrics remain `metrics_available = False`.

### 2.1 Curated Failure-Case Static Filter Audit (`outputs/sprint3_1/ball_tracking/false_positive_examples.csv`)

| Case Set | Sprint 3 Baseline Filter | Sprint 3.1 Improved Filter | Notes |
|---|---:|---:|---|
| Crowd / High-Vis Stewards / Pure Turf / Close-Up Shorts FPs (`n=6`) | 0 / 6 rejected (all kept) | **6 / 6 rejected (100%)** | Rejects `f=162`, `f=296`, `f=322` (`conf=0.74`), `f=446` (`conf=0.65`), `f=98`, `f=1640` |
| White-Sock Jump Outliers (`f=14`, `f=166`, `n=2`) | 0 / 2 rejected | Delegated to Kalman motion gate + 2-frame init confirmation | Intentionally not suppressed statically to avoid killing real balls at feet |
| True Match Footballs (`f=18` open pitch, `f=650` at player's feet, `n=2`) | 2 / 2 preserved | **2 / 2 preserved (100%)** | Zero false suppression on true balls |

### 2.2 Detector Model & Strategy Comparison (`outputs/sprint3_1/ball_tracking/detection_metrics.json`)

- **Legacy 11-Frame Count-Only Reference (Norway vs Wales, `iou_verified=False`):**
  - `yolo11n.pt`: Ball count F1 = `0.5333` (`P=0.400`, `R=0.800`), Player count F1 = `0.9935` (`P=0.9872`, `1` FP), `775.1 ms/frame`.
  - `yolo11s.pt`: Ball count F1 = `0.5333` (`P=0.400`, `R=0.800`), Player count F1 = `0.9325` (`P=0.8837`, `10` crowd-person FPs), `1401.4 ms/frame`.
- **Chelsea 13 Tactical Frames (Coverage vs Latency, not recall):**
  - `yolo11n` hybrid 1088 + tiled fallback: `53.85%` coverage at `1827.8 ms/f` (`kalman_roi_recovery`: `46.15%` at `815.7 ms/f`).
  - `yolo11s` hybrid 1088 + tiled fallback: `61.54%` coverage at `3014.3 ms/f`.
- **Frozen Decision:** Retain `yolo11n.pt` (`imgsz=1088`, pre-filtered tiled + Kalman ROI) as default for CPU efficiency and cleaner player precision; keep `yolo11s.pt` as a selectable GPU option (`--model yolo11s.pt`) once verified IoU GT is annotated.

---

## 3. Player Tracking — ByteTrack vs BoT-SORT on Continuous Wide-Angle Shots

> **Validation Blocker:** Formal IDF1/HOTA require manual multi-object identity GT (`metrics_available = False`). Within-shot fragmentation (`IDs / player slot`), cross-cut ID leaks, lifetime, and team consistency are measured with identical detections and track-level team discovery.

| Segment | Tracker | Distinct IDs | Cross-Cut ID Leaks | Within-Shot IDs / Player Slot | Mean Track Lifetime (obs) | Team Consistency | Latency (ms/f) |
|---|---|---:|---:|---:|---:|---:|---:|
| Chelsea vs Sunderland (`start=840`, 150f) | `bytetrack.yaml` | 57 | **0** | 4.057 | 19.39 | 97.13% | 328.5 |
| Chelsea vs Sunderland (`start=840`, 150f) | **`botsort.yaml`** | 57 | **0** | **3.933 (-3.1%)** | 18.68 | 94.76% | 341.7 |
| Norway vs Wales (`start=1000`, 120f) | `bytetrack.yaml` | 29 | **0** | 2.849 | 29.97 | 100.00% | 311.5 |
| Norway vs Wales (`start=1000`, 120f) | **`botsort.yaml`** | **25 (-13.8%)** | **0** | **2.552 (-10.4%)** | **33.68 (+12.4%)** | **100.00%** | 332.1 |

---

## 4. Held-Out Unseen Match (`Man Utd 4-4 Aston Villa`, 1,500 frames, Frozen Config)

Apples-to-apples comparison on identical frames (`0..2998`, `frame_stride=2`, `120.0s` analysed) between Sprint 3 (`outputs/man_utd_4_4_aston_villa_highlights_20261117_104302`) and Sprint 3.1 (`outputs/sprint3_1/man_utd_villa_run`). Ground-truth accuracy on this unseen match is `pending_human_review` (review templates exported to `data/annotations/man_utd_4_4_aston_villa_highlights/`).

| Subsystem / Diagnostic Metric | Sprint 3 Baseline | Sprint 3.1 Improved (Frozen) | Delta / Interpretation |
|---|---:|---:|---|
| **Team Discovery & Consistency** | | | |
| Reliable Appearance Tracks Discovered | 427 (178 A / 172 B / 77 Unk) | 402 (168 A / 173 B / 61 Unk) | Balanced 2-team split (`inter_team_dist = 0.8751`, `sil = 0.6135`) |
| Pass-1 vs Pass-2 Raw Track Agreement | 362 / 427 (84.78%) | **402 / 402 (100.00%)** | **+15.22 pp** — architectural two-pass mismatch fixed |
| Pass-1 vs Pass-2 Stabilized Track Agreement | 362 / 427 (84.78%) | **365 / 402 (90.80%)** | 37 diffs are 3–4f stabilizer warmups (`0` Team_A↔Team_B swaps) |
| Raw Mid-Track Team Label Flips | 79 | **1** | **-98.7%** |
| Smoothed Mid-Track Team Label Flips | 26 | **0** | **-100.0%** |
| Within-Shot Track Team Consistency | 97.40% | **100.00%** | **+2.60 pp** across all 33 shots |
| **Player Tracking** | | | |
| Cross-Cut Player ID Leaks | 30 | **0** | **-100.0%** via cut reset + monotonic `track_id_offset` |
| Distinct Tracking IDs | 549 | 681 | +132 (30 leaks removed + BoT-SORT GMC fast-pan splits) |
| Within-Shot Mean IDs / Player Slot | 3.078 | 3.764 | Honest trade-off: BoT-SORT GMC degrades on rapid highlight whip-pans at stride=2 |
| Within-Shot Short-Track Fraction (`< 5 obs`) | 29.39% | 37.44% | ByteTrack recommended if processing fast-cut highlight reels |
| **Ball Detection & Tracking** | | | |
| Top-Band Crowd/Steward Ball FPs (`y < 140px`) | 52 | **0** | **-100.0%** elimination of stand/steward FPs |
| Consecutive Camera-Cut Wipe Double-Fires | 16 | **0** | **-100.0%** via 6-frame cooldown |
| Camera Cuts Detected | 49 | **32** | -34.7% spurious replay-wipe triggers |
| Extreme-Speed Ball Jumps (`> 1500 px/s`) | 36 | **18** | **-50.0%** |
| Large Observed Jumps (`> 150 px`) | 114 | **81** | **-28.9%** |
| P95 Observed Ball Step (`px`) | 396.19 px | **253.00 px** | **-36.1%** tail jump reduction |
| Mean Continuous Ball Track Length | 14.68 frames | **23.20 frames** | **+58.0%** longer continuous ball tracks |
| Max Continuous Ball Track Length | 94 frames | **108 frames** | **+14.9%** |
| Fragmented Active Ball Track Segments | 76 | **44** | **-42.1%** |
| False Track Associations Logged | 1 | **0** | -100% |
| Kalman ROI Recovery Hits (Auto-Team Mode) | 0 | **10** | Restored in auto-teams mode |
| Ball Visibility Provenance (Obs / Interp / Lost) | 54.0% / 20.4% / 25.6% | 52.8% / 15.27% / 31.93% | Cleaner on-pitch tracks (-3.47pp crowd FPs stripped, honest Lost) |
| **Possession (Descriptive — GT Pending)** | | | |
| Attributed Possession Coverage | 17.40% (20.88 s) | **19.13% (22.96 s)** | **+1.73 pp** (Tactical Shot 1: `30.53% → 43.16%`) |
| Aligned Red (Man Utd) / White (Villa) Share* | 58.24% / 41.76% | **56.10% / 43.90%** | Consistent across runs despite unordered cluster swap |
| Full 6-State Distribution (`A / B / Cont / Loose / Unk / Out`) | 10.13 / 7.27 / 3.60 / 31.40 / 46.20 / 1.40% | 8.40 / 10.73 / 3.87 / 29.13 / 47.07 / 0.80% | Explicit `Unknown` preserved; no forced overrides |

*\*Cluster ID alignment verified directly from torso crop RGB means: in Sprint 3 Baseline `Team_A` = Red (`RGB 180,121,106`) and `Team_B` = White (`RGB 176,182,174`); in Sprint 3.1 `Team_B` = Red (`RGB 178,128,112`) and `Team_A` = White (`RGB 173,180,171`).*

---

## 5. Continuous-Shot Proxy (`Chelsea vs Sunderland`, `27.mp4`, start=840, 350 frames)

Because no static elevated tripod grassroots match video exists in the workspace, `27.mp4` (`start_frame=840`, 350 frames / 28.0s) serves as the strongest available continuous tactical wide-shot proxy:

- **Team Discovery:** `selected_k = 4`, `82` reliable tracks (`21 Team_A`, `29 Team_B`, `32 Unknown`), `inter_team_cosine_distance = 0.7338`, `silhouette = 0.4746`.
- **Player Tracking:** `0` cross-cut leaks, `0` raw/smoothed team flips, `100.0%` within-shot team consistency.
- **Ball Tracking:** `59.14%` observed, `7.14%` interpolated, `33.71%` lost, `mean_continuous_track_frames = 33.14`, `max_continuous_track_frames = 135` (`10.8 s` uninterrupted), `p95_observed_step_px = 79.06 px` (3.2× tighter than highlight reel), `0` false track associations.
- **Possession:** `8.86%` overall coverage across the 28s clip (`43.33%` during active tactical Segment 5), `UNKNOWN 49.14%`, `LOOSE_BALL 37.43%`, `CONTESTED 2.00%`, `OUT_OF_PLAY 2.57%`, attributed share `54.84% / 45.16%`.
- **Fixed-Camera Generalisation Status:** **Pending genuine fixed-camera grassroots footage.**
