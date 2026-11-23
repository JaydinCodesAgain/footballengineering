# Sprint 3.3 — Empirical Baseline Confirmation

**Date:** 2026-04-12  
**Base Commit (`main`):** `6b5cd90` (`Merge pull request #4 from JaydinCodesAgain/feat/sprint3.2-panoramic-detection-accuracy`)  
**Working Branch:** `feat/sprint3.3-reliable-match-understanding`  
**Verified Sources:**
- [SPRINT3_2_TECHNICAL_REPORT.md](../../sprint3_2/SPRINT3_2_TECHNICAL_REPORT.md)
- [SOCCERTRACK_V2_BASELINE_REPORT.md](../../soccertrack_v2_baseline/SOCCERTRACK_V2_BASELINE_REPORT.md)
- [sprint3_2_summary.json](../../sprint3_2/sprint3_2_summary.json)
- [ball_benchmark_results.json](../../sprint3_2/ball_eval/ball_benchmark_results.json)
- [tracker_and_possession_eval.json](../../sprint3_2/tracker_and_possession_eval.json)

---

## 1. Confirmed Sprint 3.2 Baseline Metrics (180 Sampled Frames, `stride=2`, `14.4 s`)

| Match ID | Role in Sprint 3.3 | Conditions | Resolution | Player Recall (`5m`) | Player Precision (`5m`) | Player F1 (`5m`) | GS-HOTA (`5m`) | IDF1 (`5m`) | MOTA (`5m`) | IDSW / Frag | Team Covered Accuracy | Team Coverage | GK Rejection |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `117092` | **Development (Night)** | Floodlit night | `3840×1906` | `69.92%` (`89.81%@8m`) | `71.09%` (`91.31%@8m`) | `70.50%` (`90.55%@8m`) | `0.205` (`0.325@8m`) | `27.80%` | `40.42%` | `145` / `180` | `77.70%` | `94.28%` | `100.0%` |
| `118575` | **Development (Day)** | Daylight | `4096×1080` | `97.32%` | `93.36%` | `95.30%` | `0.567` | `72.42%` | `90.24%` | `39` / `32` | `97.50%` | `91.21%` | `100.0%` |
| `128057` | **Previously Inspected Reference** | Daylight | `4096×1080` | `94.86%` | `95.00%` | `94.93%` | `0.648` | `64.61%` | `88.78%` | `79` / `82` | `100.00%` | `97.53%` | `100.0%` |
| `117093` | **Previously Inspected Reference** | Daylight | `4096×1080` | `92.59%` | `96.20%` | `94.36%` | `0.608` | `63.61%` | `87.77%` | `65` / `68` | `95.18%` | `93.26%` | `100.0%` |

> **Strict Data Hygiene Notice:** Matches `128057` and `117093` were evaluated and visually audited in prior sprints. They are tracked strictly as **previously inspected regression/reference datasets** in Sprint 3.3 and are **not** presented as untouched blind test sets. A separate untouched match (`132877`, validation split) is reserved exclusively for final post-freeze blind generalization evaluation (Step 9).

---

## 2. Confirmed Ball Detection & Tracking Baseline

### 2.1 Development Ball Benchmark (`n = 50` manually verified frames: `40` visible, `10` absent/out-of-view)
Winning Sprint 3.2 configuration (`S3_yolo11s_4tile_kalman_pitchfilter`):
- **Precision (`25 px` tolerance):** `65.38%` (`17 TP / 9 FP`)
- **Recall (`25 px` tolerance):** `42.50%` (`17 TP / 23 FN`)
- **F1 Score:** `0.5152` (`51.52%`)
- **AP (`@25 px`):** `0.3156`
- **Absent-Frame True Negatives:** `8 / 10` (`2` false-positive predictions on absent-ball frames)
- **Mean / Median Localization Error:** `7.27 px` / `6.14 px`
- **Per-match breakdown:**
  - `118575` (Day dev, `25` frames, `18` visible): `64.71%` Precision, `61.11%` Recall, `62.86%` F1 (`8.32 px` mean error)
  - `117092` (Night dev, `25` frames, `22` visible): `66.67%` Precision, `27.27%` Recall, `38.71%` F1 (`5.35 px` mean error)

### 2.2 Previously Inspected Validation Match `117093` (`n = 12` visual audit frames)
- **Precision (`25 px`):** `25.00%` (`2 TP / 6 FP`)
- **Recall (`25 px`):** `25.00%` (`2 TP / 6 FN`)
- **Absent-Frame True Negatives:** `4 / 4` (`100%`)
- **Primary Failure Cause:** Stationary **saturated blue training cone** located ~1 m outside the left touchline (`PitchZone.BOUNDARY_MARGIN`, roughly `(316, 488)`) where a player took a throw-in (`f=206..280`). Because a player stood within `1.8 * ref_h`, both the `BOUNDARY_MARGIN` proximity check and the isolated static-spot detector allowed the cone through, causing the Kalman tracker to lock onto the stationary blue cone.

---

## 3. Confirmed Known Bottlenecks Targeted in Sprint 3.3

1. **Ball Intelligence (`Sub-Agent A`):**
   - Stationary touchline cones (especially blue/orange/red markers in `BOUNDARY_MARGIN`) bypass spatial filtering whenever a player stands nearby (`<= 1.8 * ref_h`).
   - Static-spot accumulation (`_update_static_spot_detector`) previously required `is_isolated` (`min_p_d > 2.0 * ref_h`), allowing fixed sideline objects near players to evade blacklisting.
   - Far-side `6–9 px` balls (`117092` night recall `27.3%`) and fast passes after kicks suffer from missed detections and sluggish Kalman re-acquisition after sharp velocity changes.
2. **Night-Time Team Classification (`Sub-Agent B`):**
   - On `117092` (floodlit night match: White vs Dark Navy), specular floodlight glare and white jersey numbers on dark-navy jerseys have low HSV saturation (`S < 45`) and high value (`V > 180`), inflating `achromatic_ratio` and shifting `achrom_v_hist` brightward in `hsv_lab`, misclassifying 5 distant dark-navy player tracks as White (`77.70%` covered accuracy vs `95.2%–100%` in daylight).
3. **Long-Sequence Player Tracking (`Sub-Agent C`):**
   - Player tracks fragment over multi-hundred-frame sequences (`145` IDSW / `180` Frag on `117092`; `39` IDSW / `32` Frag on `118575`) when players cross paths or drop below detector confidence on the far side for `5–35` frames.
4. **Possession Verification & Geometric Out-of-Play (`Sub-Agent D`):**
   - `evaluate_frame_ownership_candidates` in `src/possession/ownership.py` still checks fractional image bounding boxes (`is_ball_out_of_play_bounds`) rather than `PitchRegionFilter` geometry for `OUT_OF_PLAY`, and possession has only been evaluated against unverified silver proxy timelines (`BAS + GSR`) rather than manually verified continuous intervals (`verified=True, source="manual"`).
5. **Long-Duration Validation & Compute Profiles:**
   - Evaluations to date covered `180` sampled frames (`14.4 s`). Multi-minute stability (`30 s`, `2 min`, `5 min`, `10 min`) and explicit hardware compute profiles (`local_low_vram`, `colab_t4`, `cpu_fallback`) must be engineered and benchmarked.

