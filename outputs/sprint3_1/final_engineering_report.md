# Football Vision Engine — Sprint 3.1 Final Engineering Report
## Model Accuracy, Generalisation & Fixed-Camera Reliability

**Repository:** `https://github.com/JaydinCodesAgain/footballengineering`  
**Branch:** `feat/sprint3.1-accuracy-reliability` (branched from Sprint 3 commit `4c91813`)  
**Phase Focus:** Phase 1 — Recorded-Match Accuracy & Reliability Foundation  
**Date:** 2026-11-19

---

## 1. Executive Summary

Sprint 3.1 systematically audited, fixed, and benchmarked the core vision and analytics subsystems of the Football Vision Engine across five engineering priorities plus an accuracy-focused evaluation harness (`src/evaluation/harness.py`). Every model choice was benchmarked on existing human-verified datasets (`Norway vs Wales`, `n=147`; `Chelsea vs Sunderland`, `n=80`) and **frozen prior to evaluating** the unseen held-out match (`Man Utd 4-4 Aston Villa`, 1,500 frames) and continuous-shot proxy (`Chelsea vs Sunderland`, 350 frames).

### Distinction Between Validated Accuracy and Implemented Capabilities

| Subsystem | Independently Validated Against Human GT? | Key Quantified Result |
|---|---|---|
| **Automatic Team Discovery** | **YES** (2 matches, `n=227` verified crops) | Norway vs Wales: **0.9748** covered Macro-F1, **0.9658** 3-way Macro-F1, **0.00%** referee misclass (`0/21`), **11.1%** unknown misclass (-50%), **0.9000** 3-way ARI. Chelsea vs Sunderland: **0.9898** covered Macro-F1, **0.00%** referee misclass (`0/3`), **8.33%** unknown misclass (-75%), **0** cross-team swaps, **0.0000** seed std across 5 seeds. Unseen match GT: *pending human review*. |
| **Two-Pass Track Integration** | **Verified structurally & on unseen footage** | Pass-1 vs Pass-2 raw track agreement rose from **84.78% (362/427)** to **100.00% (402/402)**; mid-track smoothed team flips fell **26 → 0**; within-shot team consistency hit **100.0%** across all 33 unseen shots. |
| **Ball Detection Filtering & Tracking** | **Curated failure-frame audit (YES); Full-match IoU mAP (PENDING)** | Static FP filter rejects **6/6** crowd/steward/turf/close-up FPs (incl. conf `0.74` yellow stewards) while preserving **2/2** true balls. On unseen 1,500-frame match: top-band crowd FPs **52 → 0**, replay-wipe double cuts **16 → 0**, extreme jumps >1500px/s **36 → 18 (-50%)**, p95 step **396 → 253 px (-36%)**, mean continuous track **14.68 → 23.20 frames (+58%)**, Kalman ROI recovery **0 → 10**. *Formal IoU AP@50/recall blocked by 0 verified manual ball boxes.* |
| **Player Tracking** | **Within-shot continuity (YES); Formal IDF1/HOTA (PENDING)** | Cross-cut player ID leaks eliminated (**30 → 0** on unseen match; **0** across all benchmarks). BoT-SORT reduced within-shot fragmentation vs ByteTrack on both steady continuous shots (`-3.1%` Chelsea, `-10.4%` Norway, `+12.4%` lifetime) but fragmented more on fast whip-pan highlight cuts at `frame_stride=2` (`ids/slot 3.078 → 3.764`). *Formal IDF1/HOTA blocked by 0 verified identity tracks.* |
| **Confidence-Aware Possession** | **Unit/regression verified; Full-match GT (PENDING)** | Eliminated raw-fallback override of stabilizer-rejected `Unknown` players; perspective-scaled ball speed threshold. On unseen match: coverage **17.40% → 19.13%** (Shot 1 tactical play **30.53% → 43.16%**), aligned Red/White share consistent (`58.24/41.76%` → `56.10/43.90%`), explicit `UNKNOWN` preserved (`47.07%`). *Possession accuracy blocked by 0 verified manual intervals.* |
| **Fixed-Camera Generalisation** | **PENDING REAL FIXED-CAMERA FOOTAGE** | All 3 local videos are panning broadcast TV highlights. Continuous wide-shot proxy (`27.mp4`) achieved **135-frame** continuous ball tracking and **79.06 px** p95 ball step, but genuine static tripod grassroots validation remains pending footage capture. |

---

## 2. Root-Cause Diagnosis & Algorithmic Improvements

### 2.1 Priority 1 — Automatic Team Discovery (`src/classification/team_discovery.py`, `src/classification/features.py`, `src/pipeline/runner.py`)

**Baseline Weaknesses Diagnosed:**
1. **Two-pass architectural disconnect:** `_run_auto_team_pipeline()` ran YOLO twice (Pass 1 for discovery, Pass 2 for inference) and discarded Pass 1 track descriptors, classifying Pass 2 crops one frame at a time with synthetic `margin=0.0` and `second_nearest_distance=distance`. This caused **65/427 (15.22%)** track disagreements between discovery and inference and **26** smoothed mid-track label flips.
2. **Permissive radius & lost outlier awareness:** Cluster radii (`p90 * 1.35`) inflated to `0.4008` cosine distance and excluded outlier cluster centers were thrown away, allowing referees, goalkeepers, and crowd/jacket crops into playing-team clusters (`unknown_misclassification_rate = 0.22–0.33`, up to `0.50` on Norway Half 2).
3. **Broken alternative clusterers & seed instability:** `agglomerative` used `average` linkage (chaining collapse: `ARI = 0.08–0.34`), `gmm` used diagonal covariance in 519-d (`ref_mis = 0.67–0.76`), and `kmeans` had seed instability on Chelsea (`acc_3way` dropped from `0.9375` to `0.8250` on seeds 0/99).
4. **Crowd/close-up contamination:** Non-tactical close-ups and crowd/scoreboard frames entered discovery tracks unfiltered.

**Implemented Improvements:**
- **Eliminated redundant 2nd YOLO pass & fixed track integration:** `_discover_teams_with_frame_cache()` caches Pass-1 detections, torso features, quality weights, and shot boundaries, assigns verified Pass-1 track descriptors directly via `(shot_id, tracking_id)`, and uses a new online `TrackAppearanceBuffer` for short/unqualified tracks with camera-cut and spatial-jump resets.
- **Tactical discovery gating:** `is_tactical_discovery_frame()` and `filter_discovery_player_indices()` filter close-ups, low-turf graphics, and high-elevation crowd persons before clustering.
- **Robust multi-clusterer architecture:** Upgraded `AutomaticTeamDiscoverer` with `KMeans (n_init=30)`, Ward `AgglomerativeClustering`, PCA-projected full-covariance `GaussianMixture`, `HDBSCAN` (`copy=True`), and adaptive-eps `DBSCAN`; added `auto_select_k` (`K ∈ {2..4}`), multi-player-per-frame pair selection, retained outlier-cluster repulsion (`outlier_repulsion_ratio=1.05`), MAD/IQR cluster radii (`2.6σ` capped at `0.78 * inter_team_dist`), margin (`0.03`) and distance-ratio (`0.85`) rejection gates, and full `ClassificationPrediction` telemetry.
- **Feature benchmark & selection:** Benchmarked 6 handcrafted features (`baseline_hsv`, `hsv_lab`, `foreground_masked`, `dominant_colour`, `spatial_embedding`, `hybrid_hsv_lab_fg`) × 5 clusterers (`30` configs per match) plus `torchvision` MobileNetV3-Small. Selected `hsv_lab` + `kmeans` empirically: highest combined accuracy, lowest Unknown contamination (`0.0833` / `0.1111`), `0.0000` seed variance, and `0.78–1.73 ms/crop` CPU speed (~3–6× faster than `spatial_embedding`, ~100× faster than MobileNetV3-Small).

### 2.2 Priority 2 & 3 — Ball Detection Filtering & Tracking (`src/possession/ball_tracker.py`, `src/detection/detector.py`)

**Baseline Failure Taxonomy (Verified Visually on Real Frames):**
1. **High-vis yellow stadium stewards in stands** (`f=162`, `f=322 conf=0.74`, `f=446 conf=0.65`) — **52** observed top-band (`y < 140px`) FPs in 1,500 frames.
2. **Uniform green turf phantoms** (`f=98 conf=0.21`) when the true ball is elsewhere on the pitch.
3. **White-sock/boot teleports** (`f=14` 341px jump `conf=0.46`; `f=166` 601px teleport `conf=0.20`).
4. **Close-up oversized clothing boxes** (`Norway f=1640`, 83×82px on player shorts).
5. **Replay-wipe camera-cut double-fires** (16 consecutive cut triggers out of 49 total cuts).
6. **Disabled Kalman ROI recovery** in `_run_auto_team_pipeline()` (`0` hits in Sprint 3).

**Implemented Improvements:**
- **Pitch-horizon & visual appearance FP filters:** Added `estimate_pitch_horizon_y()` (turf row density), `_local_surrounding_green_ratio()` (excluding box interior with strict turf Hue `36..92` so neon-yellow stewards never pass as pitch grass), `_is_high_vis_steward_patch()`, `_is_pure_turf_patch()`, corner watermark/scoreboard rejection, close-up oversized clothing rejection, and centered low-conf white-sock rejection.
- **Set-piece-safe KalmanBallTracker:** Added 6-frame camera-cut wipe cooldown, 2-frame low-confidence (`< 0.45`) track initialization confirmation (eliminating 1-frame FP hijacks when LOST), motion-direction and size-consistent association scoring, bounded confidence decay during interpolation, and TTL-expiring + player-proximity-released static blacklists so legitimate stationary free-kick/corner/penalty balls are preserved.
- **Restored Kalman ROI recovery & pre-filtered tiled fallback:** Both `tracker.py` and `runner.py` now pre-filter full-frame candidates before skipping tiled fallback and invoke `detect_ball_in_roi()` when the Kalman tracker is actively interpolating (`10` recovered detections on the unseen match).

### 2.3 Priority 4 — Player Tracking (`src/tracking/tracker.py`, `src/tracking/stabilizer.py`, `src/evaluation/tracking_eval.py`)

**Implemented Improvements:**
- **Zero cross-cut ID leaks:** Added HSV + player-scale-jump camera-cut resets inside `PlayerTracker.track_video()` with a strictly monotonic global `track_id_offset` so Ultralytics tracker resets never recycle IDs back to `1`, paired with `TemporalTeamStabilizer.on_camera_cut()`.
- **Within-shot fragmentation metrics:** Added `evaluate_within_shot_player_tracking()` to separate broadcast camera cuts from genuine within-shot fragmentation (`IDs per player slot`, short-track fraction, track lifetime, cross-cut leak count, within-shot team consistency).
- **ByteTrack vs BoT-SORT empirical finding:** BoT-SORT (`botsort.yaml`) won on both steady wide-angle shots (`Chelsea -3.1%`, `Norway -10.4%` fewer IDs/slot, `+12.4%` longer track lifetime, `0` leaks) and was set as default (`--tracker` CLI flag added for override). However, on the fast-cut `Man Utd 4-4 Aston Villa` highlight reel at `frame_stride=2`, BoT-SORT's sparse optical-flow GMC fragmented more during whip-pan transitions (`ws_ids/slot 3.764` vs Sprint 3 ByteTrack `3.078`). Both outcomes are documented transparently so operators can pick `--tracker bytetrack.yaml` for fast-panning highlight footage and `--tracker botsort.yaml` for steady tactical footage.

### 2.4 Priority 5 — Possession Reliability (`src/possession/ownership.py`, `src/possession/possession_engine.py`, `src/evaluation/possession_eval.py`)

**Implemented Improvements:**
- **Prevented false raw-classification overrides:** Previously, `infer_frame_ownership()` fell back to noisy single-frame `raw_team_id` whenever stabilized `team_id == Unknown`, overriding intentional `TemporalTeamStabilizer` rejections. Fixed so raw fallback applies *only* to warming-up tracks (`track_team_confidence <= 0.0` and `classification_confidence >= 0.72`).
- **Perspective-scaled ball speed threshold:** Scales `max_ball_speed_for_control_px_per_sec` by `median_player_height / 80.0` (clamped `[0.65, 1.75]`) so far-side passes and close-range dribbles are judged consistently across broadcast zoom levels.
- **Expanded evaluation metrics:** Added `incorrect_confident_attribution_rate`, `team_ownership_swap_frames`, and `per_state_duration_error_seconds` to `src/evaluation/possession_eval.py`, and exported `possession_timeline.csv` from auto-team pipeline runs for direct evaluation by `src/evaluation/harness.py`.

---

## 3. Fixed-Camera Reliability Protocol

Because all three available recordings (`Norway vs Wales`, `Chelsea vs Sunderland`, `Man Utd vs Aston Villa`) are broadcast TV highlights with panning/zooming cameras and frequent cuts, **fixed-camera generalisation cannot yet be claimed as validated**.

When elevated static tripod grassroots footage (`data/raw/fixed_grassroots_match.mp4`) is recorded, run the exact frozen protocol:

1. **Calibrate static pitch homography (optional for Phase 1, required for Phase 2):**  
   `configs/venue_pitch_calibration.json` (4+ pitch landmarks via `src/calibration/pitch.py`).
2. **Run end-to-end label-free inference:**
   ```bash
   python -m src.pipeline.cli --video "data/raw/fixed_grassroots_match.mp4" --auto-teams --enable-possession --tracker botsort.yaml --max-frames 3000
   ```
   *(If fixed camera has zero camera pan and low CPU budget, benchmark `--tracker bytetrack.yaml` alongside `--tracker botsort.yaml` to verify whether disabling GMC improves track lifetime further on static backgrounds.)*
3. **Annotate 45+ representative frames & possession intervals** using the exported HTML annotators in `data/annotations/<match_id>/` (`verified=true, source=manual`).
4. **Run the accuracy evaluation harness:**
   ```bash
   python -m src.evaluation.harness --run-dir "outputs/<exp_id>" --match-id "<match_id>" --annotations-root "data/annotations" --output-json "outputs/<exp_id>/harness_eval.json"
   ```

---

## 4. Remaining Failure Modes & Honest Limitations

1. **Ground-truth annotation gap (Critical Blocker):** Zero manually verified ground-truth ball boxes, possession intervals, and player identity tracks exist in `data/annotations/*`. Until human annotations are completed (templates are now pre-generated for all 3 matches including Man Utd vs Aston Villa), ball IoU mAP/AP@50, player IDF1/HOTA, and possession share MAE remain unmeasured.
2. **Crowd-cutaway & replay-graphic leakage in team discovery:** Visual inspection of `outputs/sprint3_1/man_utd_villa_run/team_cluster_gallery.png` confirms strong Red vs White separation and goalkeeper/referee rejection, but a handful of crowd cutaway shots (spectators' heads/jackets) and broadcast badge-wipe transitions still passed `is_tactical_discovery_frame()` and appeared among the 61 `Unknown`/tail tracks.
3. **High-speed whip-pan fragmentation under BoT-SORT GMC at `frame_stride=2`:** BoT-SORT improved steady tactical continuity (`-3.1%` to `-10.4%` IDs/slot) yet increased short tracks during rapid broadcast highlight whip-pans (`short_track_fraction 29.4% → 37.4%` on Man Utd vs Aston Villa).
4. **Small/occluded ball recall on `yolo11n.pt`:** Generic COCO `sports ball` weights still miss fast aerial balls and heavily occluded touches at 720p (`~52.8–59.1%` observed coverage on wide shots). A dedicated football-fine-tuned detector (with verified IoU GT evaluation) or GPU `yolo11s`/higher-resolution inference will likely be needed to push ball recall above 80% on elevated grassroots cameras.
5. **Stabilizer end-of-video lock counter telemetry:** Calling `TemporalTeamStabilizer.on_camera_cut()` resets `locked_label` on shot boundaries, which means `locked_tracks_count` reflects only tracks locked in the final camera shot (`1` on the 32-cut Man Utd highlight) and `identity_resets_triggered` includes cut-induced history clears (`524`). Predictions are unaffected (`smoothed_team_label_changes = 0`, `within_shot_team_consistency = 100.0%`), but separating `camera_cut_clears` from `mid_shot_identity_resets` is a clean follow-up improvement.
6. **CPU-only inference latency:** PyTorch in this workstation environment has `torch.cuda.is_available() == False` despite an RTX 500 Ada GPU in `nvidia-smi`. Full 1,500-frame processing took `1,222.37 s` (`1.23 FPS`) on CPU; GPU execution is expected to run 8–15× faster.

---

## 5. Phase 1 Accuracy Gate & Phase 2 Readiness Recommendation

**Recommendation: DO NOT start Phase 2 spatial analytics yet.**

**Rationale:**
- **Software & algorithmic Phase 1 foundation is now strong and well-tested** (57/57 automated tests passing, verified `0.975–0.990` covered Macro-F1 and `0.00%` referee misclassification on unsupervised team discovery, `100%` Pass-1/Pass-2 track agreement, `0` cross-cut player ID leaks, `0` top-band steward ball FPs, `0` replay-wipe double cuts, `+58%` longer continuous ball tracks).
- **However, the Phase 1 empirical accuracy gate is NOT yet closed**, because:
  1. Ball detection IoU AP@50/recall, player IDF1/HOTA, and possession share error have **0 manually verified ground-truth annotations** (`metrics_available = False`).
  2. Unseen-match team discovery (`Man Utd 4-4 Aston Villa`) is awaiting human verification of its exported review templates.
  3. **Zero elevated fixed-camera grassroots footage** has been evaluated.

**Exact Next Steps to Close Phase 1 Before Phase 2:**
1. Complete manual annotation of **30–50 frames per match** in `data/annotations/*/detection_annotator.html` and **5–10 minutes of intervals** in `possession_annotator.html`, then run `python -m src.evaluation.harness` to establish verified IoU ball AP@50 and possession MAE.
2. Record or acquire **one genuine 10–15 minute fixed elevated tripod grassroots match clip** and run the frozen Sprint 3.1 fixed-camera protocol (comparing `botsort.yaml` vs `bytetrack.yaml` on zero-pan footage).
3. If verified ball IoU recall on fixed grassroots footage falls below `75%`, fine-tune a compact YOLO11 football-only ball+player detector on annotated grassroots frames before building Phase 2 homography-projected heatmaps and passing networks.
