# Sprint 3.5 football-box review hand-off

This batch contains **360 unreviewed frames**. It is not training data and must not be used for model fitting or evaluation until all review steps below are complete.

| Source match | Split reserved now | Frames | Suggested boxes |
|---|---:|---:|---:|
| Chelsea vs Sunderland | train | 120 | 196 |
| Norway vs Wales | train | 120 | 202 |
| Man United vs Aston Villa highlights | validation | 120 | 163 |

The dashed teal boxes in the local HTML files are low-confidence YOLO11n suggestions. They are deliberately noisy and do not appear in the ground-truth CSV export. A suggestion may be incorrect, may identify a non-match object, or may miss the ball entirely.

## Reviewer procedure

1. Open each local `detection_annotator.html` file under `data/annotations/sprint3_5_ball_review/<match_id>/` in a browser.
2. For every frame, inspect the image at native resolution. Draw a tight **BALL** box only around the active match ball when its visible extent can be located. Redraw accepted suggestions; do not regard a dashed suggestion as accepted merely because it appears.
3. Do **not** label cones, markings, football-shaped graphics, crowd objects, or spare/out-of-play balls as the active match ball. For a fully hidden, absent, or ambiguous ball, leave the box list empty and tick **Mark Frame Verified**. Those reviewed negatives are important precision controls.
4. For partially visible balls, box only the visible ball extent if it can be identified confidently. If it cannot, treat the frame as a reviewed negative rather than guessing.
5. Use the export button. Replace the corresponding `detection_boxes.csv` in that same match folder with the downloaded CSV. Keep the files in their existing match folders; do not mix source matches.

Then run:

```powershell
python scripts/export_sprint3_5_yolo_dataset.py
```

The command writes `outputs/sprint3_5/training_gate_status.json`. It will remain blocked unless there are at least 300 manually verified frames, at least 100 manual ball boxes, and both train and validation source matches remain distinct. Once it reports `training_eligible: true`, export the local, gitignored YOLO dataset:

```powershell
python scripts/export_sprint3_5_yolo_dataset.py --export
```

Only then may the configured candidate experiment be run:

```powershell
yolo detect train model=yolo11s.pt data=data/sprint3_5_ball_yolo/dataset.yaml imgsz=1280 epochs=100 patience=20 batch=2 device=cpu seed=42 deterministic=True
```

The complete proposed hyperparameters and augmentation policy are in `configs/sprint3_5_ball_training.yaml`. A separate genuinely unseen, manually reviewed source must still be reserved for final model evaluation; it must not influence training or model selection.
