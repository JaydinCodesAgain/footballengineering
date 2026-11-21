# SoccerTrack v2 Fixed-Camera Baseline Experiment Report
## Unchanged Frozen Football Engineering Model (Sprint 3.1 Baseline)

- **Date:** 2026-11-20
- **Model state:** **100% unchanged** (`git diff --stat src/` empty; Sprint 3.1 frozen pipeline)
- **CLI invocation (exact unchanged defaults):**
  - Match `117092` (train split):
    `python -m src.pipeline.cli --video "SoccerTrack-v2/data/videos/117092/117092_panorama_1st_half.mp4" --auto-teams --enable-possession --experiment-name "soccertrack_v2_baseline/match_117092_unchanged"`
  - Match `128057` (designated SoccerTrack v2 **test** split):
    `python -m src.pipeline.cli --video "SoccerTrack-v2/data/videos/128057/128057_panorama_1st_half.mp4" --auto-teams --enable-possession --experiment-name "soccertrack_v2_baseline/match_128057_unchanged"`
- **Frozen runtime defaults:** `yolo11n.pt`, `imgsz=1088`, `person_conf=0.40`, `ball_conf=0.20`, `frame_stride=2`, `max_frames=180` (frames `0..358`, `14.4 s` from kickoff at `25 fps`), `tracker=botsort.yaml`, `team_discovery=hsv_lab + kmeans (auto_select_k)`, CPU inference (`torch 2.14.1+cu126`, `cuda=False`).
- **Zero-leakage discipline:** No SoccerTrack v2 reference labels were supplied to detection, tracking, team discovery, or possession. For held-out test match `128057`, its GSR/BAS/ball/calibration reference files were **not downloaded onto disk** until after the frozen pipeline finished producing all outputs.

---

## 1. Evaluation Methodology & Dataset Coordinate Verification

Visual overlay verification caught an essential SoccerTrack v2 dataset property (matching `SoccerTrack-v2/docs/format-gsr.md` and `CHANGES_v1_2.md`):

- On **match `117092`** (`3840×1906`, night floodlit dual-camera stitched fisheye panorama), the GSR `bbox_image` field lives in an undistorted/shrunk coordinate frame rather than raw fisheye video pixels (the reason `CHANGES_v1_2.md` notes *"117092 could not be measured"* against raw detector boxes). Evaluating image-space IoU directly against `bbox_image` on `117092` yields a spurious `0.006` recall even on exact player hits.
- On **match `128057`** (`4096×1080`, daylight panorama), `bbox_image` sits near the players in raw pixel space but uses small ~58×43 boxes derived from quantized `1.05 m × 0.68 m` pitch positions (`docs/format-gsr.md`: *"GS-HOTA is scored in pitch space only... `bbox_image` must not be used for scoring"*).
- **Authoritative scoring used below:** Predicted player foot points `((x1+x2)/2, y2)` are projected onto the `105 m × 68 m` metric pitch using each match's exact fisheye ray-plane inverse (`raw/<match>/<match>_camera_intrinsics.npz` via `cv2.fisheye.undistortPoints` + `(R, t)` ground-plane intersection) and matched 1-to-1 (Hungarian assignment) against GSR's authoritative `(pitch_x, pitch_y)` at the official SoccerNet GSR **5.0 m** tolerance.
  - Keypoint calibration check (`raw/<match>/<match>_keypoints.json`, 65 pitch landmarks):
    - `117092` (stitched dual-camera panorama): median `3.31 m`, p90 `8.02 m` (we therefore also report `8.0 m` sensitivity for `117092`).
    - `128057`: median `2.05 m`, p90 `4.86 m`; matched player median localization error on pitch is **`0.67 m`** (mean **`0.92 m`**), with minimum error at canonical v1.2 offset `gsr_frame = opencv_frame + 1` (`0.63 m`).
- All post-hoc evaluation code lives in [scripts/evaluate_soccertrack_v2.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/evaluate_soccertrack_v2.py).

---

## 2. Executive Comparison: Match `117092` (Train) vs Held-Out Match `128057` (Test)

All metrics below are computed on the exact unchanged model in authoritative 2D metric pitch space (`d <= 5.0 m`), unless explicitly noted.

| Metric | Match `117092` (Train, Night `3840×1906`) | Match `128057` (Held-Out Test, Day `4096×1080`) |
|---|---:|---:|
| **Video & Runtime** | | |
| Frames processed (`stride=2`) | `180` (`0..358`, `14.4 s`) | `180` (`0..358`, `14.4 s`) |
| Wall time / Avg FPS (CPU) | `482.4 s` (`0.37 FPS`) | `431.1 s` (`0.42 FPS`) |
| Avg tracked players / frame (of 22) | `3.11` | `7.41` |
| False camera cuts on fixed camera | **`0`** | **`0`** |
| **Player Detection (`<= 5.0 m` pitch space)** | | |
| Detection Precision (all / on-pitch) | `80.47%` / `82.84%` (`449 / 558`) | **`86.81%`** / `87.00%` (`1,158 / 1,334`) |
| Detection Recall (of 22 GT players/frame) | **`12.52%`** (`449 / 3,586`) | **`29.41%`** (`1,158 / 3,938`) |
| Detection F1 | `0.2167` | `0.4393` |
| Median / Mean matched pitch distance | `3.33 m` / `3.33 m` | **`0.67 m`** / **`0.92 m`** |
| Off-pitch sideline detections | `16` (`13 Team_A`, `3 Unknown`) | `62` (`60 Team_A`, `2 Unknown`)* |
| **Unsupervised Team Discovery (Post-Hoc Aligned)** | | |
| Selected `k` / Inter-team cosine dist / Silhouette | `k=2` / `0.6012` / `0.5101` | `k=2` / `0.6613` / `0.6810` |
| **Covered Outfield Accuracy** (stabilized) | **`83.16%`** (`242 / 291`) | **`98.50%`** (`1,054 / 1,070`) |
| **Covered Outfield Macro-F1** (stabilized) | **`0.8273`** | **`0.9849`** |
| Classification Coverage (stabilized) | `64.81%` (`291 / 449`) | `92.40%` (`1,070 / 1,158`) |
| Overall Outfield Accuracy incl. `Unknown` | `53.90%` | `91.02%` |
| Raw Pre-Stabilizer Covered Acc / Coverage | `83.74%` / `82.18%` | `98.09%` / `99.57%` |
| Adjusted Rand Index (stabilized / raw) | `0.1941` / `0.3138` | `0.8040` / `0.9172` |
| Track-Level Outfield Accuracy (covered) | `78.26%` (`18 / 23`) | **`96.77%`** (`30 / 31`) |
| GT Outfield Players Majority Team (OK / Wrong / Unk) | `10 / 1 / 5` (of 16 matched) | **`15 / 0 / 1`** (of 16 matched) |
| Within-Shot Mid-Track Team Switches | **`0`** (`100.0%` consistent) | **`0`** (`100.0%` consistent) |
| Goalkeepers detected | `0` (too small at far goals) | `0` at `5 m` (`2` obs at `8 m`, `2/2 Unknown`) |
| **Formal Player Identity Tracking (`<= 5.0 m`)** | | |
| **GS-HOTA** (`DetA` / `AssA` / `LocA`) | `0.0415` (`0.038` / `0.048` / `0.642`) | **`0.2731`** (`0.228` / `0.329` / `0.863`) |
| **MOTA** (all / on-pitch) | `0.0792` / `0.0837` | `0.2402` / `0.2410` |
| **IDF1** (`IDP` / `IDR`) | `0.1086` (`0.403` / `0.063`) | **`0.3065`** (`0.606` / `0.205`) |
| ID Switches / Fragmentations | `56` / `158` | `36` / `133` |
| MT (>=80%) / PT (20–80%) / ML (<20%) | `0 / 6 / 16` | `2 / 8 / 12` |
| Mean GT Dominant-ID Purity (when detected) | `54.56%` (`3.05` pred IDs/GT) | **`78.34%`** (`2.18` pred IDs/GT) |
| Distinct Tracks / Mean / Max Track Lifetime | `59` / `9.49` / `53` frames | `46` / `29.0` / `170` frames |
| **Ball Tracking & Possession** | | |
| Reported Ball Observed / Active Coverage | `0.00%` (`0 / 180`) | `3.33%` (`6 / 180`) — **ALL 6 FPs** |
| **Verified Sample Ball Precision** (`n=22` audit) | `N/A` (`0` predictions) | **`0.00%`** (`0 / 3` in sample; `0 / 6` full run) |
| **Verified Sample Ball Recall** (`n=22` audit) | **`0.00%`** (`0 / 10` visible) | **`0.00%`** (`0 / 8` visible) |
| Attributed Possession Coverage | `0.00%` (`100% UNKNOWN`) | `0.00%` (`96.67% UNKNOWN`, `3.33% LOOSE`) |
| Fabricated Possession Share | None (`null / null`) | None (`null / null`) |

*\*Note on `128057` off-pitch count: includes near-touchline player boxes whose projected foot coordinate falls slightly beyond `|y|=36.5 m` due to the 2.05 m boundary calibration error, while the pink-bib sideline assistant (`P3`) was visually confirmed cleanly rejected as `Unknown` across the clip.*

*(Sensitivity check on `117092` at `8.0 m` tolerance to account for its `3.31 m` stitched-panorama calibration error: Detection Precision `87.81%`, Recall `13.66%`, Covered Team Accuracy `83.91%`, Covered Macro-F1 `0.8383`, MOTA `0.0993`, IDF1 `0.1149`, GS-HOTA `0.0707`. On `128057` in raw image space at `IoU >= 0.30`: Covered Team Accuracy `99.01%`, Covered Macro-F1 `0.9889`, IDF1 `0.1674`, HOTA `0.0798`.)*

---

## 3. Step 2 Findings: Team Discovery & Player Identity Maintenance

### Does the unchanged model discover the actual two teams?
- **YES on daylight fixed-camera footage (`128057`, held-out test):** Without any labels or kit colors, `hsv_lab + kmeans` cleanly discovered **White (`Team_A`) vs Royal Blue (`Team_B`)** at **`98.50%` covered accuracy (`Macro-F1 = 0.9849`)**, **`96.77%` track-level accuracy**, **`15/16` matched outfield players assigned to the right team (`0` wrong)**, **`0` mid-track team flips**, and cleanly rejected the **pink-bib sideline official (`P3`) as `Unknown`**.
- **PARTIALLY on night floodlit stitched footage (`117092`):** It separated **White vs Dark Navy/Blue** at **`83.16%` covered accuracy (`Macro-F1 = 0.8273`)** and `10/16` outfield players right (`1` wrong, `5` Unknown), **but** two clear failure modes degraded it:
  1. **No outlier cluster selected (`selected_k = 2`):** Because `117092` only had `3.11` players/frame, dark-coated sideline coaches/substitutes (`P76`, bench region) entered the pool and were pulled into `Team_A` rather than rejected into a 3rd outlier cluster.
  2. **Harsh floodlight/night-glare illumination shift:** Bright-lit vs shadowed navy jerseys had wide appearance spread (`cluster_radii = 0.36` vs `0.24–0.27` on `128057`), pushing several dark-navy players (`#41`, etc.) into `Unknown` (`35.2%` rejection rate on matched players).

### Does it maintain player identities?
- **The primary bottleneck is player DETECTION RECALL (`12.5%` on `117092`, `29.4%` on `128057`), NOT tracker instability on large players:**
  - Because `yolo11n.pt` runs at `imgsz=1088` with `person_conf=0.40` on `3840×1906` / `4096×1080` full-pitch panoramas (a **3.5–3.8× linear shrink**), mid-field and far-side players shrink to **7–12 px tall** in YOLO's input tensor and drop below `0.40` confidence. As a result, `12–16` of the `22` players are **Mostly Lost (`ML`)** and goalkeepers at the far ends of the pitch are missed entirely.
  - On the near-side players that **are** consistently detected on `128057`, identity maintenance is encouraging: **2 players achieved Mostly Tracked (`>=80%`)**, max track lifetime reached **`170 / 180` processed frames**, mean dominant-ID purity reached **`78.34%`**, and `IDP` reached **`60.57%`** with **`0` false camera cuts**.
  - On `117092` (night + severe low recall), intermittent detections hovering around `person_conf=0.40` caused heavy fragmentation (`158` fragmentations, `56` ID switches, `47.5%` short tracks `< 5` obs, `mean_track_lifetime = 9.49` frames, `IDF1 = 0.1086`).

---

## 4. Step 3 Findings: Ball-Tracking Errors & Verified Ball-Box Sample

### Verified Ball-Box Sample (`n = 22` frames across both matches)
We created a **22-frame visually verified ball-box sample** inspected on 4× zoomed native-resolution patches (saved in [verified_ball_sample.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.json) and [verified_ball_sample.csv](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.csv) with explicit `source="visual_audit"` provenance, alongside human-reviewable 30-frame HTML5 `DetectionAnnotationStore` kits under [data/annotations/soccertrack_117092/detection_annotator.html](file:///c:/Users/jmorrison/projects/footballengineering/data/annotations/soccertrack_117092/detection_annotator.html) and [data/annotations/soccertrack_128057/detection_annotator.html](file:///c:/Users/jmorrison/projects/footballengineering/data/annotations/soccertrack_128057/detection_annotator.html)):

- **`117092` sample (`10` frames, all 10 match-ball visible):**
  - Native ball size: **`10–19 px` wide × `10–14 px` tall** (median **`11.5 × 11.0 px`**; `19×12 px` during motion-blurred high pass at `f=86`).
  - Unchanged model performance: **`TP = 0`, `FP = 0`, `FN = 10` -> Recall = `0.00%`, Precision = `N/A` (0 detections), F1 = `0.00`**.
- **`128057` sample (`12` frames: `8` match-ball visible + `4` verified out-of-play negative controls):**
  - Native ball size: **`9–14 px` wide × `9–13 px` tall** (median **`9.0 × 9.0 px`** on midfield/kickoff play, `14×13 px` at the near touchline `f=182`).
  - Unchanged model performance: **`TP = 0`, `FP = 3` (in sample; `6/6` across full 180-frame run), `FN = 8` -> Precision = `0.00%`, Recall = `0.00%`, F1 = `0.00`**.

### Root-Cause Ball Error Taxonomy
1. **Scale collapse at panoramic 4K -> `imgsz=1088` / 2-tile `960` (100% miss rate on real match ball):**
   - A `9×9 px` match ball on a `4096×1080` panorama shrinks to **`2.4 × 2.4 px`** at `imgsz=1088` and **`3.6 × 3.6 px`** in a `0.58`-width horizontal tile resized to `960`. Even on `117092` (`11.5×11.0 px` ball), it shrinks to `3.2 px` full-frame / `5.0 px` tiled. Pretrained COCO `yolo11n.pt` cannot detect `2–5 px` footballs.
2. **Off-pitch stationary spare ball false positive (`128057`, 100% of observed detections):**
   - All `6` observed ball detections on `128057` (`f=8, 10, 12, 270, 272, 274`) locked onto a **real spare football resting on the concrete curb outside the near-right touchline** at `(3492, 828)` (`17×23 px`, conf `0.18–0.50`), `~894 px` away from live play. Because green artificial turf surrounds the painted *"University of Tsukuba"* lettering outside the touchline, turf-surround filtering passed it, and there is no pitch-boundary polygon mask yet.
3. **Static-blacklist TTL re-acquisition loop (`static_zone_ttl_frames = 250`):**
   - `KalmanBallTracker` correctly identified the concrete-curb spare ball as static/isolated at frame `12` (`false_track_associations = 1`) and blacklisted `(3492, 828)`. However, after `250` frames elapsed (`f=270`), the TTL expired and the tracker immediately re-locked onto the exact same stationary curb ball for 3 more frames before re-blacklisting it (`false_track_associations = 2`). On a fixed camera, truly off-pitch static zones should stay masked permanently.
4. **Auxiliary ball track caveat confirmed visually:**
   - On high passes (`117092 f=110, 160, 180` and `128057 f=120, 148`), SoccerTrack v2's `ball/*.npz` moves linearly along the ground plane while the real ball is high in flight or already heading out of play—confirming that verified image ball boxes (not `ball/*.npz`) are strictly required to measure ball precision/recall.
5. **Conservative possession guardrail held:**
   - Despite `0%` true ball recall and `6` spare-ball FPs on `128057`, the possession engine refused to hallucinate possession (`0.0%` coverage, `team_a_pct = null, team_b_pct = null`, `100% UNKNOWN` on `117092`, `96.67% UNKNOWN / 3.33% LOOSE_BALL` on `128057`).

---

## 5. Priority Engineering Roadmap (Recorded Before Making Any Changes)

1. **Tiled / High-Resolution Panoramic Detection for Players & Ball (Highest Impact):**
   - Replace full-frame `imgsz=1088` single-pass detection on wide fixed panoramas with overlapping multi-tile pitch inference (e.g., 3–4× horizontal pitch crop tiles at native scale, ignoring top/bottom black panoramic letterbox bands) plus a lower person threshold / stronger detector (`yolo11s` or football-fine-tuned weights) to lift player recall from `12–29%` toward `>80%` and bring `9–14 px` balls to `12–20 px` in detector input space.
2. **Fixed-Camera Pitch-Polygon ROI Mask (Eliminates Sideline Coaches, Benches & Spare Balls):**
   - Use a single static venue pitch polygon (or `pitch-calibration` corners) to mask out off-pitch bench crowds (`117092` `Team_A` bench contamination) and off-pitch spare balls (`128057` concrete-curb ball at `(3492, 828)`), and disable TTL expiry for static false-ball zones on fixed cameras.
3. **Dedicated Small-Ball Detector / Motion-Differencing Candidate Proposal:**
   - Generic COCO `yolo11n.pt` misses `9–12 px` footballs even when visible on open turf. Combine multi-tile inference with frame-differencing / background-subtraction motion proposals (trivial and fast on a fixed camera!) or fine-tuned football-only YOLO weights, evaluated against the verified ball-box sample.
4. **Night-Match Robustness & GPU Execution:**
   - Improve night/floodlit team clustering (`117092` `83.2%` vs daylight `128057` `98.5%`) by excluding off-pitch boxes before clustering and normalizing luminance/glare in Lab space; also restore CUDA PyTorch execution on the local RTX 500 Ada GPU (`0.4 FPS` CPU -> expected `5–10+ FPS` GPU) so full-half evaluations become fast.

---

## 6. Recorded Artifacts

- Unchanged run outputs:
  - [match_117092_unchanged/analytics_report.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/match_117092_unchanged/analytics_report.json) & [team_cluster_gallery.png](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/match_117092_unchanged/team_cluster_gallery.png)
  - [match_128057_unchanged/analytics_report.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/match_128057_unchanged/analytics_report.json) & [team_cluster_gallery.png](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/match_128057_unchanged/team_cluster_gallery.png)
- Post-hoc SoccerTrack v2 evaluation reports:
  - [eval_117092_unchanged.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/eval_117092_unchanged.json)
  - [eval_128057_unchanged.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/eval_128057_unchanged.json)
  - [scripts/evaluate_soccertrack_v2.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/evaluate_soccertrack_v2.py)
- Verified ball-box sample & visual evidence:
  - [verified_ball_sample.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.json) & [verified_ball_sample.csv](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verified_ball_sample.csv)
  - [verify_117092_boxes_v2.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verify_117092_boxes_v2.jpg), [verify_128057_boxes_v2.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/ball_audit/verify_128057_boxes_v2.jpg), [128057_f0272_spare_ball_fp.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/soccertrack_v2_baseline/visuals/128057_f0272_spare_ball_fp.jpg)
  - Human review annotators: [soccertrack_117092/detection_annotator.html](file:///c:/Users/jmorrison/projects/footballengineering/data/annotations/soccertrack_117092/detection_annotator.html) and [soccertrack_128057/detection_annotator.html](file:///c:/Users/jmorrison/projects/footballengineering/data/annotations/soccertrack_128057/detection_annotator.html)

