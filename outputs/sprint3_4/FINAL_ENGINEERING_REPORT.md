# Football Engineering — Sprint 3.4 Engineering Report

## Model accuracy, optimisation, and generalisation

**Branch:** `feat/sprint3.4-model-accuracy`

**Base ancestry:** built on the unmerged Sprint 3.3 branch (`6f3f544`); `main` remains at `6b5cd90`.
**Status:** evidence-led implementation and data audit complete; **not Phase 1 ready**. No claim in this report establishes the requested `>=90%` target.

## 1. Executive summary

The primary failure is upstream, not a possession-state-machine threshold: on Sprint 3.3's dense 30-second daylight evaluation, a ball was directly observed in only `85 / 375` processed frames (`22.67%`) and observed-or-interpolated in `110 / 375` (`29.33%`). The possession engine therefore emitted `UNKNOWN` on `295 / 375` frames (`78.67%`) rather than fabricating control. Its `30.67%` state accuracy, `16.27%` confident attribution coverage, and `39.69 pp` possession-share error cannot be solved safely by forcing player ownership.

The 74-frame ball benchmark tells the same story. Sprint 3.3 reaches `27 / 55` visible-ball true positives (`49.09%` recall) and `27 / 34` selected detections (`79.41%` precision) at a 25-pixel centre tolerance. The unresolved misses include stationary kickoffs, balls at feet, 6–8-pixel far-side balls, aerial motion blur, and several near rolling passes. Existing hard-negative filters prevent many cones/spare balls, but do not create the missing detector signal.

Sprint 3.4 therefore adds a deliberately conservative recovery experiment rather than a brittle confidence-threshold relaxation:

- `FixedCameraMotionBallDetector` can now **optionally** retain only the lower foot zone of player boxes. The old full-player mask remains the default.
- The retained foot zone is still subject to existing three-frame trajectory or Kalman corroboration, compactness, contrast, pitch-region, and false-positive filters.
- A regression test proves the historical full mask remains unchanged and that an opt-in, temporally confirmed moving ball at a player's feet can be recovered.
- Dataset/protocol tooling now audits SoccerTrack assets and labels reconstructed pitch-coordinate ball tracks explicitly as a proxy, not independent image-space football ground truth.

The new recovery flag is not enabled in any shipped default configuration. Its real-data ablation must pass before it is promoted.

## 2. Data split and annotation audit

The prior Sprint 3.3 benchmark frames from `117092`, `118575`, `117093`, `128057`, and the former blind match `132877` have all been inspected. They are development/reference material only for this sprint.

Sprint 3.4 downloaded only the compact public SoccerTrack `132831` MOT clip and small metadata needed to inspect a new route without downloading multi-gigabyte full-match videos. The reproducible audit is in [`soccertrack_asset_audit.json`](soccertrack_asset_audit.json).

| Asset | Result | Allowed use |
|---|---|---|
| `132831` MOT clip | 5,976 frames, 25 FPS, 3,840×1,504; 22 annotated players/frame | Independent MOT evaluation if held out from tuning |
| `132831_1st_ball.npz` | 67,775 dense pitch-coordinate rows; `67,774` have status `4` | Trajectory/projection diagnostic only |
| Ball projection | 5,976 / 5,976 clip-overlap coordinates project inside the image | Coordinate-alignment sanity check only |
| Independently verified `132831` ball boxes | Not available | Cannot train/evaluate ball precision or recall |

The clip begins at source frame `4,575`, obtained from the release's `183,000 ms` first-half padding record. The coordinate conversion is explicit: source ball coordinates are pitch-centred and are translated by `(+52.5, +34.0)` before the released pitch-to-image homography is applied.

**Important limitation:** a dense reconstructed ball trajectory is not a visually reviewed ball bounding-box annotation. It must not be silently converted into claimed detector precision/recall, and it must not be used as fine-tuning supervision without image-level review. A genuine final holdout must be a different, untouched match after model selection is frozen.

## 3. Baseline versus Sprint 3.4 evidence

| Capability | Most recent measured baseline | Sprint 3.4 measured result | Sprint target | Status |
|---|---:|---:|---:|---|
| Player detection (`132877`, 5 m) | Recall `90.65%`, precision `85.33%` | Not rerun — full source clip unavailable locally | `>=90%` each | Precision target not met in baseline |
| Team classification (`132877`) | Covered accuracy `89.41%` | Not rerun | Macro-F1 `>=90%`, coverage `>=90%` | No validated achievement |
| Player tracking (`132877`) | IDF1 `74.63%` | Not rerun | IDF1 `>=90%` | Not achieved |
| Football detection (`n=74`) | P `79.41%`, R `49.09%`, F1 `0.607` | No independent post-change ball benchmark yet | P/R `>=90%` | Not achieved |
| Possession (`118575`, 30 s, dense stride 2) | State acc `30.67%`; coverage `16.27%`; share error `39.69 pp` | Not rerun; upstream ball signal remains limiting | State `>=90%`; coverage `>=80%` | Not achieved |

The existing 95% confidence intervals for the 74-frame ball set are necessarily broad because only 55 frames contain a visible ball. Using Wilson intervals, the baseline recall is approximately `49.1%` (`27 / 55`; 95% CI about `36–62%`) and selected-detection precision is `79.4%` (`27 / 34`; 95% CI about `63–90%`). These uncertainty bounds make a 90% claim indefensible even before considering the non-independent data split.

## 4. Ball detection and foot-zone recovery

### Root cause

The current `FixedCameraMotionBallDetector` masks every detected player box plus padding before extracting compact bright motion. That protects against false positives from legs, boots, and shadows, but it also removes the lower-body pixels where controlled balls appear. This is particularly damaging for possession: a ball can be visible at the feet and still be unavailable to recovery.

### Change

`MotionBallConfig` now exposes:

- `preserve_player_foot_region=False` (default)
- `player_foot_region_height_fraction=0.18`

When explicitly enabled, masking stops above the player's lower foot zone. No detector confidence rule has been loosened. A candidate must still survive the original positive-motion, local-contrast, green-turf, pitch-region, compactness, and temporal/Kalman confirmation logic.

### Regression evidence

`test_opt_in_foot_zone_motion_recovery_preserves_temporally_confirmed_ball` uses a causal four-frame sequence because the motion verifier needs an initial reference image plus three moving blobs. It establishes two properties:

1. The default full-box mask produces no recovery candidate.
2. The opt-in foot-zone mode produces one compact candidate only after three-frame motion confirmation.

This is an algorithmic regression test, **not** an accuracy claim. It cannot substitute for manually reviewed match data.

### Real-data experiment status

[`benchmark_sprint3_4_foot_recovery.py`](../../scripts/benchmark_sprint3_4_foot_recovery.py) compares full masking and foot-zone masking densely on `132831`, with released MOT player boxes held constant. It labels the ball score a `reconstructed_coordinate_proxy` and records unmatched proposals separately. It must not be presented as ball precision/recall.

The bounded 4K proxy run completed for four dense frames. Both full-mask and foot-zone variants achieved `0 / 4` proxy matches and emitted `3` unmatched proposals (`0.75` proposals/frame). This is a negative/inconclusive result, not evidence for activation; the feature remains disabled by default. The raw proxy record is [`foot_zone_motion_proxy.json`](foot_zone_motion_proxy.json).

## 5. Possession accuracy and coverage

The dense 30-second result has a clear causal chain:

```text
weak ball detection (22.67% observed) → short / broken trajectory (29.33% tracked or interpolated)
→ no reliable player-ball association → conservative UNKNOWN (78.67%)
→ 16.27% attributed-control coverage → unstable possession share
```

The team labels were not the immediate dominant issue in this clip: among the 43 frames where both ground truth and prediction had confident team attribution, the earlier report recorded `43 / 43` correct. The coverage failure is therefore more important than changing ownership thresholds. Sprint 3.4 preserves all six possession states and continues to avoid attributing long ball-loss intervals.

## 6. Tracking and team classification

No tracking or team-classification parameter was changed in this implementation pass. That is intentional: no current post-change benchmark demonstrates an association or appearance gain, and changing those systems would obscure the ball/possession causal diagnosis.

The `132831` MOT clip is suitable for a later dense, truly held-out tracking run against its 22-player MOT labels. It must remain excluded from tracker configuration selection until that final run. Existing Sprint 3.3 values (`IDF1 74.63%` on inspected `132877`, and the 30-second `118575` IDF1 `78.35%`) remain below the sprint target.

For team discovery, the pre-existing conflict is retained rather than hidden: visual night-cluster purity improved on `117092`, but pitch-reference covered accuracy declined from `77.33%` to `73.94%`. That discrepancy requires calibrated, independently labelled team references—not another feature change based only on a gallery.

## 7. Dense continuous benchmarks and runtime

The only defensible dense continuous numbers available at report time remain the Sprint 3.3 cached `118575` stride-2 result: 375 sampled frames spanning 30 seconds, 4-tile YOLO11s at 1280. It is reported as a baseline, not repeated as a Sprint 3.4 result.

| Measurement | Value | Interpretation |
|---|---:|---|
| Mean detection latency (prior CPU run) | `4,747 ms/frame` | Measured on the prior CPU run, not a GPU projection |
| Effective sampled throughput (prior CPU run) | `0.18 FPS` | Insufficient for multi-minute dense evaluation locally |
| Local CUDA availability at Sprint 3.4 audit | `false` | No local GPU throughput claim is possible |
| New 4K foot-zone proxy ablation | `0 / 4` proxy matches; `3` unmatched proposals in each variant | Negative/inconclusive; not eligible as independent ball accuracy |

No Colab T4 measurement has been performed in this sprint. The existing `colab_t4` profile remains the recommended route for 2-, 5-, and 10-minute dense `stride=2` benchmarks once the full match videos and an independently labelled protocol are available.

## 8. Reproducibility

The new asset audit and proxy experiment are deterministic and require no changes to the production CLI.

```powershell
hf download atomscott/soccertrack-v2 --type dataset `
  --include "ball/132831_1st_ball.npz" `
  --include "mot/clips/132831.mp4" `
  --include "mot/132831.txt" `
  --include "raw/132831/132831_homography.npy" `
  --include "raw/132831/132831_padding_info.csv" `
  --local-dir data/soccertrack_v2_external/132831

python scripts/audit_sprint3_4_soccertrack.py `
  --dataset-root data/soccertrack_v2_external/132831 --match-id 132831

python scripts/benchmark_sprint3_4_foot_recovery.py `
  --dataset-root data/soccertrack_v2_external/132831 --match-id 132831 `
  --frames 150 --stride 1
```

Downloaded SoccerTrack assets are ignored by Git. Large video files and model weights must remain out of the repository.

## 9. Remaining limitations and required next steps

1. **Training data blocker:** at least 3,000 independently verified image-space ball boxes across at least six match-disjoint venues are needed before dedicated small-ball detector fine-tuning. Split by match before sampling/augmentation; retain a never-viewed final match.
2. **Evaluation blocker:** download full development videos or run on Colab T4 to repeat the dense 30-second test after any detector change. Evaluate 2-, 5-, and 10-minute sequences at `stride=2`; do not use sparse stride 15–60 for tracking/possession claims.
3. **Ball benchmark:** label boxes and explicit absent/occluded states on a new final match. Report visible-frame recall, false-positive rate on absent frames, confidence intervals, and category breakdowns for near/far/aerial/at-feet/stationary/night.
4. **Possession:** only retune player association/state transitions after ball recall and trajectory continuity increase. Maintain `UNKNOWN`, `LOOSE_BALL`, `CONTESTED`, and `OUT_OF_PLAY`; report attribution coverage separately from state accuracy and possession-share error.
5. **Tracking/team:** run ByteTrack, BoT-SORT, and the fixed-camera stitcher against identical detections on the untouched MOT clip, then freeze the winner before final unseen-match evaluation.

## 10. Phase 1 readiness assessment

**Not ready.** The system has useful player detection, fixed-camera filtering, conservative uncertainty handling, and a reproducible evaluation foundation. However, verified football recall, continuous possession coverage, tracking identity consistency, and held-out team macro-F1 all remain materially below the requested thresholds. Releasing a 90% accuracy claim now would be misleading.
