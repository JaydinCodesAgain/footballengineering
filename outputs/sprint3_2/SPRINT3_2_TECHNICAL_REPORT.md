# Football Engineering — Sprint 3.2 Technical Report
## Panoramic Player Detection, Small-Ball Recovery & Fixed-Camera Accuracy

- **Branch:** `feat/sprint3.2-panoramic-detection-accuracy`
- **Date:** 2026-11-21
- **Compute Environment:** Windows CPU inference (`Python 3.12.10`, `torch 2.14.1+cu126` with `torch.cuda.is_available() == False`; RTX 500 Ada GPU present in `nvidia-smi` but unavailable to PyTorch runtime). All latency and memory figures below are measured on CPU.
- **Experimental Discipline (Zero Data Leakage):**
  - **Development matches (used for tiling, pitch-filter, motion-ball, and tracker ablations):** SoccerTrack v2 train split **`117092`** (night floodlit dual-camera stitched panorama, `3840×1906`) and **`118575`** (daylight panorama, `4096×1080`).
  - **Previously inspected baseline reference (NOT tuned on):** SoccerTrack v2 test split **`128057`** (`4096×1080`, inspected during the baseline report).
  - **Genuinely unexamined blind held-out match:** SoccerTrack v2 validation split **`117093`** (`4096×1080`, tightest calibration RMS `8.64 px`). Its video and venue calibration were downloaded only after freezing the Sprint 3.2 configuration; its reference labels (`gsr/117093/117093_1st.json`, `bas/117093/*`, `ball/117093_1st_ball.npz`) were **not downloaded onto disk until after both the unchanged baseline and frozen Sprint 3.2 finished running**.
  - No SoccerTrack v2 reference labels were supplied to detection, tracking, team discovery, or possession inference.

---

## 1. Executive Summary (Verified Before vs After Across All 4 Matches)

All end-to-end runs evaluate identical `180`-frame segments (`frames 0..358`, `stride=2`, `14.4 s` at `25 fps`) using the authoritative SoccerTrack v2 2D metric pitch-space evaluator ([scripts/evaluate_soccertrack_v2.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/evaluate_soccertrack_v2.py), `<= 5.0 m` SoccerNet GSR tolerance; `8.0 m` sensitivity also reported for stitched dual-fisheye `117092` due to its `3.31 m` median / `8.02 m` p90 calibration error).

**Frozen Sprint 3.2 Fixed-Camera Panoramic Configuration:**
`--detection-mode tiled --model yolo11s.pt --person-conf 0.28 --ball-conf 0.14 --tile-width 1280 --tile-height 1280 --tile-overlap 0.22 --tile-imgsz 1280 --tracker botsort.yaml --fixed-camera --pitch-calibration <venue_intrinsics> --auto-teams --enable-possession`
*(Broadcast CLI defaults remain 100% backward compatible: `inference_mode="full_frame"`, `fixed_camera=False`, `reset_on_camera_cut=True`.)*

### Table 1A — Player Detection, Formal Tracking (GS-HOTA / MOTA / IDF1), & Team Discovery

| Match (Split / Lighting) | Version | Det P / R / F1 (`5m`) | GS-HOTA (`DetA` / `AssA`) | MOTA | IDF1 | IDSW / Frag | MT / PT / ML (of 22) | Team Cov Acc (Macro-F1) | Team Cov / Overall Acc | GK Rejected (`Unknown`) | CPU Wall (FPS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`117092`** *(Dev Night `3840×1906`)* | Baseline | `80.5%` / `12.5%` / `0.217` | `0.0415` (`0.038`/`0.048`) | `0.0792` | `0.1086` | `56` / `158` | `0 / 6 / 16` | `83.2%` (`0.827`) | `64.8%` / `53.9%` | `0 / 0` *(missed)* | `482.4 s` (`0.37`) |
| **`117092`** *(Dev Night, `5m`)* | **Sprint 3.2** | `67.3%` / **`69.9%`** / **`0.686`** | **`0.2051`** (`0.175`/`0.252`) | **`0.3228`** | **`0.4174`** | `145` / `180` | **`7 / 15 / 0`** | `77.7%` (`0.774`) | **`94.3%`** / **`73.3%`** | **`306 / 306` (`100%`)** | `970.0 s` (`0.19`) |
| *`117092` (`8.0 m` sensitivity)* | *Base → S3.2* | *`13.7%` → **`89.8%`** R (`86.4%` P)* | *`0.0707` → **`0.3253`** (`0.348`/`0.324`)* | *`0.0993` → **`0.6884`*** | *`0.1149` → **`0.5181`*** | *`62/161` → `267/133`* | *`0/6/16` → **`20/2/0`*** | — | — | — | — |
| **`118575`** *(Dev Day `4096×1080`)* | Baseline | `84.1%` / `22.3%` / `0.353` | `0.1826` (`0.184`/`0.181`) | `0.1636` | `0.1693` | `68` / `155` | `1 / 10 / 11` | `97.6%` (`0.972`) | `79.4%` / `77.5%` | `0 / 0` *(missed)* | `749.5 s` (`0.24`) |
| **`118575`** *(Dev Day)* | **Sprint 3.2** | `84.7%` / **`97.3%`** / **`0.905`** | **`0.5667`** (`0.558`/`0.579`) | **`0.7864`** | **`0.7514`** | **`39` / `32`** | **`21 / 1 / 0`** | **`97.5%`** (`0.975`) | **`91.2%`** / **`88.9%`** | **`321 / 321` (`100%`)** | **`428.9 s` (`0.42`)** |
| **`128057`** *(Ref Day `4096×1080`, untuned)* | Baseline | `86.8%` / `29.4%` / `0.439` | `0.2731` (`0.228`/`0.329`) | `0.2402` | `0.3065` | `36` / `133` | `2 / 8 / 12` | `98.5%` (`0.985`) | `92.4%` / `91.0%` | `0 / 0` *(missed)* | `431.1 s` (`0.42`) |
| **`128057`** *(Ref Day, untuned)* | **Sprint 3.2** | `86.1%` / **`94.9%`** / **`0.903`** | **`0.6476`** (`0.571`/`0.737`) | **`0.7922`** | **`0.8672`** | **`14` / `37`** | **`20 / 2 / 0`** | **`100.0%`** (`1.000`) | **`97.5%`** / **`97.5%`** | **`283 / 283` (`100%`)** | `496.0 s` (`0.36`) |
| **`117093`** *(Blind Held-Out Val `4096×1080`)* | Baseline | `81.5%` / `26.0%` / `0.394` | `0.2301` (`0.193`/`0.274`) | `0.1894` | `0.2665` | `45` / `116` | `1 / 8 / 13` | `96.3%` (`0.963`) | `90.3%` / `86.9%` | `54 / 54` (`100%`) | `127.4 s` (`1.41`) |
| **`117093`** *(Blind Held-Out Val)* | **Sprint 3.2** | **`84.8%`** / **`92.6%`** / **`0.885`** | **`0.6077`** (`0.563`/`0.659`) | **`0.7535`** | **`0.7840`** | **`25` / `26`** | **`20 / 1 / 1`** | **`95.2%`** (`0.952`) | `84.7%` / `80.6%` | **`186 / 186` (`100%`)** | `479.8 s` (`0.38`) |

### Table 1B — Ball Tracking, Verified Ball Precision/Recall, & Conservative Possession Diagnostics

| Match | Version | Observed / Active Ball Cov | Detection Sources (Observed) | False Track Assoc | Verified Ball Overlap P / R (Err) | Possession `UNKNOWN` / `LOOSE` | BAS Acting-Team Agreement *(Diagnostic Only)* |
|---|---|---:|---|---:|---|---:|---|
| **`117092`** *(Dev Night)* | Baseline | `0.0%` / `0.0%` | `{}` | `0` | `N/A` / `0.0%` (`0/10`) | `100.0%` / `0.0%` | `0 / 0` (`0/6` attributed) |
| **`117092`** *(Dev Night)* | **Sprint 3.2** | **`54.4%` / `67.2%`** | `52` tiled + `41` 2×ROI + `5` motion | **`0`** (`1` perm zone) | **Obs `83.3%` / `41.7%` (`1.1 px`)**; Active `75.0%` / `50.0%` (`n=12`) | `40.0%` / `50.6%` | **`4 / 4` (`100%`)** (`2/6` Unknown) |
| **`118575`** *(Dev Day)* | Baseline | `15.6%` / `18.3%` | `17` full + `11` ROI *(mostly spare balls)* | `4` | Obs `0.0%` / `0.0%`; Active `50.0%` / `8.3%` (`n=12`) | `82.2%` / `12.8%` | `1 / 2` (`50%`; `7/9` Unknown) |
| **`118575`** *(Dev Day)* | **Sprint 3.2** | **`71.1%` / `80.0%`** | `121` tiled + `4` motion + `3` 2×ROI | **`0`** | **Obs `87.5%` / `58.3%` (`2.3 px`)**; **Active `88.9%` / `66.7%`** (`n=12`) | `28.3%` / `29.4%` | **`6 / 6` (`100%`)** (`3/9` Unknown) |
| **`128057`** *(Ref Day)* | Baseline | `3.3%` / `3.3%` | `6` full *(100% `(3492,828)` curb ball FP)* | `2` | `0.0%` / `0.0%` (`0/8`, `3` FPs in `n=12`) | `96.7%` / `3.3%` | `0 / 0` (`0/3` attributed) |
| **`128057`** *(Ref Day)* | **Sprint 3.2** | **`21.1%` / `33.9%`** | `30` tiled + `8` 2×ROI | **`0`** (`0` curb FPs!) | **Obs `80.0%` / `50.0%` (`1.7 px`)** (`4/8`, `1` neg FP in `n=12`) | `75.0%` / `13.3%` | **`1 / 1` (`100%`)** (`2/3` Unknown) |
| **`117093`** *(Blind Held-Out)* | Baseline | `9.4%` / `9.4%` | `16` full + `1` ROI *(off-pitch spare ball)* | `5` | `0/8` live (`0%` visual check; locked on spare ball) | `90.6%` / `9.4%` | `0 / 0` (`0/7` attributed) |
| **`117093`** *(Blind Held-Out)* | **Sprint 3.2** | `65.0%` / `70.0%`\* | `110` tiled + `6` 2×ROI + `1` motion | `0` | **Visual audit (`n=12`): `25.0%` P / `25.0%` R** — see honest cone FP audit (§6)! | `32.8%` / `30.0%` | `4 / 5` (`80%`)\* *(inflated after f=206 by cone)* |

*\*Honest held-out caveat on `117093`: Visual audit ([verify_117093_ball_audit.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/verify_117093_ball_audit.jpg)) confirmed clean live-ball tracking during open play (`f=36`, `f=52`) and complete rejection of baseline's off-pitch spare ball above "University", **but** after `f=206` the detector locked onto a stationary blue training cone sitting 1 m outside the left touchline (inside `ball_margin_m=2.5`) whenever left-wing players stood nearby. We report this unmodified without retuning on `117093`; see Section 6.*

---

## 2. Priority 1 — Native-Resolution Overlapping Multi-Tile Detection

Implemented in [src/detection/tiling.py](file:///c:/Users/jmorrison/projects/footballengineering/src/detection/tiling.py) and wired into [src/detection/detector.py](file:///c:/Users/jmorrison/projects/footballengineering/src/detection/detector.py) and [src/tracking/tracker.py](file:///c:/Users/jmorrison/projects/footballengineering/src/tracking/tracker.py):
- **Zero-copy tile extraction (`extract_tile_views`):** Slices native NumPy views (`np.shares_memory(view, frame) == True`), avoiding full-resolution panorama copies.
- **Automatic content / pitch ROI tiling (`estimate_active_content_bbox` / `PitchRegionFilter.get_tiling_roi_bbox`):** Skips black panoramic letterbox bars and sky/bleachers. Switching from `1280×1080` to square `1280×1280` tiles reduced `117092`'s pitch ROI (`3689×1185`) from 2 rows (`8` tiles, 90% vertical overlap) to **1 row × 4 columns (`4` tiles)** — halving inference latency with zero vertical resolution loss.
- **Seam-aware class-specific merging (`merge_tiled_detections`):** Maps local coordinates back to raw panoramic pixels, partitions strictly by class (`player` vs `ball` never suppress each other), suppresses internal-seam-clipped fragments when covered by an unclipped neighbor box (`IoM >= 0.55` or `IoU >= 0.35`), and applies class-specific NMS + cross-tile containment deduplication (`player_containment_iom = 0.72` across *different* tiles only, preserving true overlapping players inside the same tile).
- **Normal vs panoramic dispatch (`inference_mode`):** Supports `"full_frame"` (backward-compatible broadcast default), `"tiled"`, and `"auto"` (`frame_width >= 2400 px`).

### Table 2 — Full-Frame vs Multi-Tile Empirical Benchmark (`n = 12` frames per dev match, before pitch-region filtering)
Data source: [outputs/sprint3_2/tiling_benchmark.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/tiling_benchmark.json) & [scripts/benchmark_sprint3_2_tiling.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/benchmark_sprint3_2_tiling.py).

| Config | Tiles (`118575` / `117092`) | `118575` (Day) P5on / R5 / F1 / AP5 | `118575` Dup `<18px` | `117092` (Night) R5 / R8 / AP8 | `117092` Ball R (`n=10`) | CPU ms/f (`118575` / `117092`) | Peak Py Mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_full_yolo11n_1088_p040` | `1 / 1` | `0.890` / `0.307` / `0.456` / `0.277` | `0` | `0.144` / `0.163` / `0.156` | `0%` | `316` / `416` | `1.8` |
| `full_yolo11n_1280_p025` | `1 / 1` | `0.878` / `0.708` / `0.784` / `0.633` | `1` | `0.360` / `0.455` / `0.369` | `0%` | `460` / `542` | `2.3` |
| `full_yolo11s_1280_p025` | `1 / 1` | `0.903` / `0.845` / `0.873` / `0.775` | `0` | `0.489` / `0.648` / `0.556` | `0%` | `1,233` / `1,520` | `2.3` |
| `tiled_yolo11n_1600x1080_img960` | `3 / 6` | `0.860` / `0.909` / `0.884` / `0.826` | `3` | `0.659` / `0.890` / `0.761` | `0%` | `2,104` / `5,110` | `11.2` |
| `tiled_yolo11n_1280x1080_img1280` | `4 / 8` | `0.862` / `0.966` / `0.911` / `0.879` | `4` | `0.678` / `0.936` / `0.788` | `10%` | `3,857` / `9,511` | `32.0` |
| `tiled_yolo11s_1280x1080_img1280` | `4 / 8` | `0.827` / `0.977` / `0.896` / `0.887` | `10` | `0.708` / `0.951` / `0.828` | `20%` | `5,492` / `26,397` | `32.0` |
| `tiled_yolo11s_1024x960_img1024` | `10 / 10` | `0.830` / `0.981` / `0.899` / `0.893` | `7` | `0.697` / `0.939` / `0.841` | `40%` | `10,326` / `16,202` | `22.7` |
| **`tiled_s_1280x1280_p030 + PitchROI`** | **`4 / 4`** | **`0.849` (all) / `0.977` / `0.908` / `0.889`** | **`<5`** | **`0.697` / `0.936` / `0.852`** | **`50%`** | **`3,979` / `6,607`** | **`31.8`** |

**Key takeaway:** Full-frame `yolo11s@1280` still missed `35%` of night players and **`100%` of footballs (`0/10`)**. Native-resolution 4-tile `1280×1280@1280` with geometric pitch-ROI cropping lifted player AP from `0.277 → 0.889` (`118575`) and `0.156 → 0.852` (`117092@8m`), while lifting raw single-frame `117092` ball recall from `0% → 50%` at `4×` lower latency than `8`-tile night inference.

---

## 3. Priority 2 — Geometric Fixed-Camera Pitch-Region Filtering & Persistent Distractor Masking

Implemented in [src/geometry/pitch_region.py](file:///c:/Users/jmorrison/projects/footballengineering/src/geometry/pitch_region.py) and [src/possession/ball_tracker.py](file:///c:/Users/jmorrison/projects/footballengineering/src/possession/ball_tracker.py):
- **Purely geometric derivation (zero hard-coded pixels):** Loads planar `PitchCalibration` JSON, fisheye `camera_intrinsics.npz` + `keypoints.json` + `homography.npy`, or perimeter `keypoints.json` directly. Anchors the playable polygon to curved fisheye touchlines and expands outward by configurable metric margins (`player_margin_m = 2.0 m`, `ball_margin_m = 2.5 m`) via local calibration displacement.
- **Coordinate-space integrity:** Explicitly distinguishes `raw_fisheye` (via `cv2.fisheye.projectPoints` / `undistortPoints`) from `undistorted` (`homography.npy`) and `planar` spaces. Visual verification on [117092_pitch_region_f32.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/pitch_region_check/117092_pitch_region_f32.jpg) and [118575_pitch_region_f32.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/pitch_region_check/118575_pitch_region_f32.jpg) confirms pixel-accurate curved touchlines excluding sideline benches, coaches, spectators, and curb spare balls while preserving throw-in/corner margins.
- **Empirical impact on dev detection (`n=12` frames each, [pitch_filter_tiling_benchmark.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/pitch_filter_tiling_benchmark.json)):**
  - `117092` (night, `yolo11s`): Eliminated **`205` off-pitch bench/spectator player FPs (`17.1/frame`)**, jumping all-detection precision `0.367 → 0.622` (`5m`) and `0.838` (`8m`) while keeping recall intact (`0.939@8m`). In the full 180-frame run, off-pitch sideline detections dropped to **`0`**.
  - `118575` (day, `yolo11s`): Eliminated **`48` off-pitch player FPs** (`P_all: 0.729 → 0.843`) and **`33 / 51` (`64.7%`) off-pitch spare-ball FPs**.
- **Isolated `BOUNDARY_MARGIN` spare-ball rejection & persistent fixed-camera static distractor mask:**
  - Any ball detection outside `PitchZone.PLAYABLE` (inside `BOUNDARY_MARGIN`) requires either an active Kalman trajectory or a nearby throw-in/corner player (`<= 1.8 * median_player_height`), instantly eliminating isolated sideline spare balls such as `(439, 681)` on `118575` (raising `yolo11s` ball precision `0.326 → 0.652`).
  - When `--fixed-camera` is active, `KalmanBallTracker` marks recurring isolated static spots outside `PLAYABLE` as **permanent (`is_permanent=True`, exempt from the 250-frame broadcast TTL)** while still allowing on-pitch `PLAYABLE` kickoff/set-piece spots to release when a player steps onto the ball. On `128057`, the `(3492, 828)` concrete-curb spare-ball loop dropped from **`6 → 0` FPs** (`0` false track associations).
  - Kept strictly optional (`--fixed-camera`, `--pitch-calibration`) so broadcast footage preserves camera-cut detection and non-fixed TTL expiry.

---

## 4. Priority 3 — Small-Football Detection, Size-by-Depth Study & 50-Frame Dev Ablation

### Expanded Diverse Visually Verified Dev Ball Evaluation Set (`n = 50` frames)
To avoid making unsupported claims from the baseline 22-frame sample, we constructed a **50-frame visually verified dev ball benchmark** strictly on `117092` (night, `26` frames) and `118575` (day, `24` frames), verified on 4× zoomed native patches ([verified_ball_eval_set_dev.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/ball_eval/verified_ball_eval_set_dev.json), [verified_ball_eval_set_dev.csv](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/ball_eval/verified_ball_eval_set_dev.csv)):
- **Categories:** `visible_near` (`13`), `visible_mid` (`11`), `visible_far` (`2`), `occluded_at_foot` (`6`), `aerial_motion_blur` (`4`), `stationary_set_piece` (`4`), and verified `absent_out_of_play` negative controls (`10` `status==0` dead-ball frames with sideline spare balls visible).

### True Native Football Pixel Dimensions by Pitch Depth
| Pitch Depth Zone | Sample Count | Width px (min / med / max) | Height px (min / med / max) | Mean Area (`px²`) | Effective Size at Baseline `imgsz=1088` | Effective Size at 4-Tile `1280×1280` |
|---|---:|---:|---:|---:|---:|---:|
| **Near pitch** | `13` | `10 / 12.0 / 18` | `8 / 11.0 / 13` | `136.6` | `~3.2 px` | **`12.0 × 11.0 px` (`1:1`)** |
| **Mid pitch** | `23` | `8 / 10.0 / 19` | `7 / 10.0 / 14` | `112.2` | `~2.7 px` | **`10.0 × 10.0 px` (`1:1`)** |
| **Far pitch / aerial** | `4` | `6 / 7.5 / 8` | `6 / 7.5 / 8` | `53.2` | `~2.0 px` | **`7.5 × 7.5 px` (`1:1`)** |

### Table 4 — 6-Strategy Small-Football Ablation (`n = 50` verified dev frames: `40` visible, `10` negative controls)
Data source: [outputs/sprint3_2/ball_eval/ball_ablation_results.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/ball_eval/ball_ablation_results.json) & [scripts/benchmark_sprint3_2_ball.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/benchmark_sprint3_2_ball.py).

| Strategy | Precision | Recall | F1 | AP | TP / 40 | FP (`Neg FP / 10`) | Day `118575` P / R | Night `117092` P / R | Med Err | CPU ms/f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. Baseline full-frame `yolo11n@1088` | `N/A` | `0.0%` | `0.000` | `0.000` | `0` | `0` (`0/10`) | `N/A` / `0.0%` | `N/A` / `0.0%` | — | `150` |
| 2. 4-tile `yolo11n@1280` + Pitch Filter | `54.5%` | `15.0%` | `0.235` | `0.112` | `6` | `5` (`3/10`) | `50.0%` / `27.8%` | `100%` / `4.6%` | `1.8 px` | `1,330` |
| 3. 4-tile `yolo11s@1280` + Pitch Filter | `65.2%` | `37.5%` | `0.476` | `0.276` | `15` | `8` (`4/10`) | `60.0%` / `50.0%` | `75.0%` / `27.3%` | `1.8 px` | `4,517` |
| 4. #3 + Unverified 3-frame motion *(rejected)* | `24.3%` | `42.5%` | `0.309` | `0.188` | `17` | `53` (`15/10`) | `60.0%` / `50.0%` | `14.6%` / `36.4%` | `2.0 px` | `4,763` |
| 5. #3 + 3-frame motion + 2× local YOLO | `65.2%` | `37.5%` | `0.476` | `0.276` | `15` | `8` (`4/10`) | `60.0%` / `50.0%` | `75.0%` / `27.3%` | `1.8 px` | `6,460` |
| **6. #3 + Kalman + 2× ROI + Kalman-gated motion (Winner)** | **`65.4%`** | **`42.5%`** | **`0.515`** | **`0.316`** | **`17`** | **`9` (`4/10`)** | **`64.7%` / `61.1%`** | **`66.7%` / `27.3%`** | **`1.8 px`** | **`4,804`** |

**Scientific Findings & Fine-Tuning Assessment:**
1. **Why unverified motion-differencing was rejected:** Even with signed positive foreground differencing (`curr > prev` & `curr > bg`, removing trailing ghosts and dark shadows) + 3-frame collinear kinematics, unverified motion still injected `45` night FPs along `117092`'s dual-camera stitch seam and wet-grass floodlight glare (`night P = 14.6%`), despite adding `0` FPs on steady daylight `118575`. Gating motion strictly by active Kalman trajectory corroboration (`require_kalman_corroboration=True`) plus `2×`-upscaled local Kalman ROI (`roi_size_px=416`) lifted daylight recall to **`61.1%`** (including `33.3%` `occluded_at_foot` and `25.0%` `aerial_motion_blur`) and boosted overall F1 (`0.476 → 0.515`) and AP (`0.276 → 0.316`) **without weakening false-positive rejection**.
2. **Why ball-specific fine-tuning was NOT forced in Sprint 3.2:** Generic COCO `yolo11s` still misses `6–8 px` far-touchline balls (`visible_far = 0%`) and stationary kickoff/set-piece balls (`0%`). However, only `40` verified visible dev ball annotations exist across 2 stadiums (and SoccerTrack v2's `ball/*.npz` is piecewise constant-speed reconstructed rather than pixel-accurate frame labels). Fine-tuning on 40 boxes on CPU-only hardware would severely overfit 2 venues without valid match-disjoint train/val/test separation. Reaching `>75%` far-pitch/aerial recall requires a dedicated ball-only fine-tuning stage once `>= 2,000` independently verified panoramic ball annotations across `>= 6` venues (with strict match-disjoint splits) and GPU training are available.

---

## 5. Player Tracking Re-Evaluation: ByteTrack vs BoT-SORT & DetA/AssA Decomposition

To isolate identity association from detector variance, [scripts/evaluate_sprint3_2_possession_and_trackers.py](file:///c:/Users/jmorrison/projects/footballengineering/scripts/evaluate_sprint3_2_possession_and_trackers.py) ran both `bytetrack.yaml` and `botsort.yaml` (`gmc_method=None` on fixed cameras) over the **exact same 180-frame cached 4-tile `yolo11s` + `PitchRegionFilter` detections** ([tracker_comparison_117092.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/tracker_comparison_117092.json), [tracker_comparison_118575.json](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/tracker_comparison_118575.json)):

| Match (Tolerance) | Configuration | GS-HOTA | `DetA` | `AssA` | `LocA` | MOTA | IDF1 | IDSW | Frag | MT / PT / ML | Purity | Tracker ms/f |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`118575`** (`5m`) | Baseline (`full_frame` BoT-SORT) | `0.1826` | `0.1841` | `0.1813` | `0.8763` | `0.1636` | `0.1693` | `68` | `155` | `1 / 10 / 11` | `0.535` | — |
| **`118575`** (`5m`) | S3.2 + **ByteTrack** | `0.5604` | **`0.5587`** | `0.5656` | `0.8309` | **`0.7881`** | `0.7293` | **`35`** | **`31`** | `21 / 1 / 0` | `0.807` | `7.35` |
| **`118575`** (`5m`) | S3.2 + **BoT-SORT (`GMC=off`)** | **`0.5667`** | `0.5577` | **`0.5789`** | `0.8306` | `0.7864` | **`0.7514`** | `39` | `32` | **`21 / 1 / 0`** | **`0.831`** | `10.08` |
| **`117092`** (`5m`) | Baseline (`full_frame` BoT-SORT) | `0.0415` | `0.0380` | `0.0484` | `0.6424` | `0.0792` | `0.1086` | `56` | `158` | `0 / 6 / 16` | `0.546` | — |
| **`117092`** (`5m`) | S3.2 + **ByteTrack** | `0.2033` | `0.1748` | `0.2486` | `0.6750` | **`0.3243`** | `0.4140` | **`137`** | **`174`** | `7 / 15 / 0` | `0.577` | `8.35` |
| **`117092`** (`5m`) | S3.2 + **BoT-SORT (`GMC=off`)** | **`0.2051`** | **`0.1750`** | **`0.2524`** | **`0.6752`** | `0.3228` | **`0.4174`** | `145` | `180` | `7 / 15 / 0` | `0.565` | `7.72` |
| **`117092`** (`8m`) | S3.2 + **ByteTrack** / **BoT-SORT** | `0.3222` / **`0.3253`** | `0.3474` / **`0.3483`** | `0.3184` / **`0.3241`** | — | `0.6841` / **`0.6884`** | `0.5161` / **`0.5181`** | `272` / **`267`** | `134` / **`133`** | `19/3/0` / **`20/2/0`** | — | — |

**Gain Decomposition (`DetA` vs `AssA`):**
1. **Detector recall dominates total improvement (~95% of the gain):** Moving from full-frame `yolo11n@1088` to 4-tile `yolo11s@1280` + `PitchRegionFilter` increased `DetA` by **`3.03×–4.61×`**, which in turn prevented mid-track drop-outs, slashing fragmentations by **`77.6%–79.4%`** on daylight matches (`155 → 32` on `118575`, `133 → 37` on `128057`, `116 → 26` on `117093`) and lifting `AssA` by **`2.24×–5.26×`**.
2. **Association comparison (BoT-SORT vs ByteTrack on identical detections):** With GMC disabled on fixed cameras (reducing tracker overhead from seconds/frame of 4K sparse optical flow to `< 10.1 ms/frame`!), **BoT-SORT consistently outperforms ByteTrack in identity continuity (`AssA` `+1.33 pp`, `IDF1` `+2.21 pp`, dominant-ID purity `+2.4 pp` on `118575`; higher `GS-HOTA`, `AssA`, `IDF1`, and `MT@8m = 20` on `117092`)**, because its direct `[x, y, w, h]` Kalman state handles small panoramic boxes better during crossings.
3. **Team stability:** Across all four matches, `smoothed_team_label_changes == 0` (`100%` within-track stability), `1,096 / 1,096` (`100.0%`) matched goalkeeper observations across all 4 Sprint 3.2 runs were cleanly rejected as `Unknown`, and `0` false camera cuts occurred on fixed cameras while broadcast camera-cut detection remains intact when `--fixed-camera` is omitted.

---

## 6. Visual Evidence, Honest Failure-Case Analysis & Limitations

### Before/After Visual Evidence
- [before_after_117092_f060.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/before_after_117092_f060.jpg) — Dev night `117092 f=60`: `1 → 22` players, `0` off-pitch bench FPs, both GKs cleanly `Unknown`, live `10×10 px` ball observed at `2.6 px` error.
- [before_after_118575_f214.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/before_after_118575_f214.jpg) — Dev day `118575 f=214`: `9 → 25` detections (all 22 players + both GKs/officials), live mid-pitch ball observed.
- [before_after_128057_f272_spare_ball_fixed.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/before_after_128057_f272_spare_ball_fixed.jpg) — Ref day `128057 f=272`: `(3492, 828)` concrete-curb spare ball FP completely eliminated (`6 → 0`), `20/20` outfield players accurately classified (`100%` covered acc).
- [before_after_117093_f262.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/before_after_117093_f262.jpg) — Blind held-out `117093 f=262`: `9 → 24` detections, baseline's giant spare ball above "University" eliminated.

### Honest Remaining Failure Modes ([failure_cases_sprint3_2.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/failure_cases_sprint3_2.jpg) & [verify_117093_ball_audit.jpg](file:///c:/Users/jmorrison/projects/footballengineering/outputs/sprint3_2/visuals/verify_117093_ball_audit.jpg))
1. **Held-out `117093` sideline blue-cone lock (`f=206..280`, visual audit):** Because a blue training cone sat `~1 m` outside the near-left touchline (inside `ball_margin_m=2.5`) directly beside a throw-in and left-wing build-up (`dist <= 1.8 * ref_h`), it bypassed the isolated-boundary check and prevented static-spot registration (which required isolation `> 2.0 * ref_h`). *Next-sprint fix:* require **inter-frame motion** (not merely player proximity) or white/high-luminance appearance for candidates outside `PitchZone.PLAYABLE`, and build a static pixel-level background foreground check for `BOUNDARY_MARGIN` detections.
2. **Tiny `6–8 px` far-touchline and aerial balls (`118575 f=1582`, `117092 f=2286`):** Pretrained COCO `yolo11s` cannot reliably resolve `6–8 px` balls even at `1:1` native tile scale (`0%` recall on `visible_far`). Solving this requires a ball-specialized fine-tuned detector or multi-frame TrackNet-style heatmap model trained on match-disjoint panoramic ball boxes.
3. **Night floodlight specular glare & team confusion (`117092`):** Adding distant far-side players lifted overall team accuracy on `117092` (`53.9% → 73.3%`, coverage `64.8% → 94.3%`), but `5/20` dark-navy players standing under harsh specular floodlight glare (`Failure C`) were misassigned to the white team (`covered accuracy 77.7%`, vs `95.2%–100.0%` on all three daylight matches). Illumination-invariant torso descriptors / chroma-only weighting for night venues is a clear next improvement.
4. **Possession validation boundary:** Across all 4 matches, continuous human-annotated possession intervals are **not** provided in SoccerTrack v2; BAS events only provide discrete acting-team anchors (`10/10` agreement on dev `117092` + `118575`, `1/1` on `128057`, `4/5` on `117093` with the cone caveat). Accordingly, continuous possession percentages are **not** claimed as validated ground truth, and `Unknown` (`28%–75%`) is preserved wherever evidence is insufficient.
5. **Segment duration & CPU runtime:** Each end-to-end match evaluation covers `180` processed frames (`14.4 s`, `stride=2`), taking `429–496 s` (`0.36–0.42 FPS`) on daylight `4096×1080` panoramas and `970 s` (`0.19 FPS`) on `3840×1906` night footage on CPU-only PyTorch. Longer full-half benchmarks require restoring CUDA execution on the RTX 500 Ada GPU.

