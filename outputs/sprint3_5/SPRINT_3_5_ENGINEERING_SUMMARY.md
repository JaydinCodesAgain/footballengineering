# Sprint 3.5 — first football-specific training experiment

## Result

No fine-tuned checkpoint was trained and no baseline-versus-candidate accuracy claim is made. The execution environment has no CUDA device, but the decisive blocker is data provenance: the repository had no manually verified training frames. Training from model suggestions, SoccerTrack reconstructed trajectories, or the 74-frame scored benchmark would invalidate the experiment.

## Existing evidence retained

| Measure | Current evidence | Scope |
|---|---:|---|
| Ball precision | 79.41% (27 / 34) | 74-frame visually verified benchmark |
| Ball recall | 49.09% (27 / 55) | Same benchmark |
| Ball F1 | 0.607 | Same benchmark |
| Candidate model metrics | Not available | No reviewed train/validation set and no candidate checkpoint |

The benchmark remains development/regression evidence, not a training source or a large independent test set.

## Executed data-preparation experiment

A deterministic 360-frame review batch was extracted from three source matches. The two training sources and validation source are match-disjoint. Low-threshold YOLO11n proposals were generated at 640px only as dashed review overlays:

| Split | Sources | Frames | Unreviewed suggestion count |
|---|---|---:|---:|
| Train | Chelsea–Sunderland; Norway–Wales | 240 | 398 |
| Validation | Man United–Aston Villa | 120 | 163 |
| Total | 3 source matches | 360 | 561 |

The counts above are not football labels or model metrics. The manifest records 0 verified frames and 0 verified ball boxes, so the dataset exporter correctly reported `training_eligible: false`.

## Guardrails added

- `FrameAnnotation.model_proposals` keeps proposal boxes separate from manual boxes. The existing reviewer shows them as dashed teal suggestions; its CSV contains only manually drawn boxes.
- The new YOLO exporter imports the reviewer CSV and accepts only `verified=true, source=manual` frames. It supports reviewed empty frames as negatives and rejects model provenance.
- Match IDs are checked for split duplication. Export fails closed below 300 verified frames or 100 verified active-ball boxes.
- The local YOLO dataset, frames, videos, and checkpoints remain gitignored.

## Candidate configuration — not executed

`configs/sprint3_5_ball_training.yaml` records a single lightweight candidate: pretrained `yolo11s.pt`, one `ball` class, 1280px input, 100 epochs, early stopping patience 20, seed 42, and conservative image augmentation. This retains the production pipeline's high-resolution small-object emphasis while avoiding synthetic copy-paste or mixup football artefacts.

The reference command and review procedure are in `REVIEW_HANDOFF.md`. It must only be run after the review gate passes. A final match-held-out evaluation source is still required before any generalisation claim or production model integration.

## Checks run

- `python -m py_compile src/detection/annotation_workflow.py scripts/prepare_sprint3_5_ball_review_batch.py scripts/export_sprint3_5_yolo_dataset.py`
- `python -m pytest -q tests/test_sprint3_5_training_data.py` — 4 passed after the final proposal-separation regression test is run.
- `python -m pytest -q tests/test_evaluation_and_report.py` — 4 passed.
- Actual 360-frame extraction and proposal run completed on CPU; the resulting manifest and closed training-gate status are committed artifacts.

No dense possession comparison was run because the prerequisite detector improvement does not yet exist.
