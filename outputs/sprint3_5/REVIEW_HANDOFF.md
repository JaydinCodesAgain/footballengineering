# Sprint 3.5 football-box review hand-off

Start here: open `data/annotations/sprint3_5_ball_review/chelsea_sunderland/detection_annotator.html` in a browser. Do **five frames only** first, export its CSV, and run the gate command below. That proves the real local workflow before you invest time in the remaining batch.

## What the reviewer means

- **Dashed teal** boxes are low-confidence YOLO suggestions only. They are not labels and are never exported as ground truth.
- **Solid yellow BALL** boxes are manually drawn labels. Redraw a suggestion only when it is the active match ball.
- **No box + Mark Frame Verified** means you inspected the frame and the active match ball is not confidently visible. Use it for a cone, boot, graphic, spare/out-of-play ball, or genuine ambiguity.
- Use **Zoom +** for small footballs. Drawing remains in native image pixels at every zoom level.

## Save and resume safely

1. The reviewer saves each edit to this browser's local storage automatically. Use **Save Browser Progress** before closing the tab.
2. Click **Download detection_boxes.csv** at the end of each short session. Replace the `detection_boxes.csv` in the same match folder. The CSV is the portable record; browser storage is only a convenience.
3. On reopening in the same browser, use **Restore Browser Progress**. To continue on another browser/device, use **Import CSV** and select the complete CSV you previously downloaded.
4. Do not rerun batch preparation while you have an unexported browser-only session. Normal preparation is resumable and refuses to overwrite a changed reviewed set; destructive regeneration requires an explicit backup and confirmation flag.

## Review order

| Order | Match / role | HTML file | Frames |
|---:|---|---|---:|
| 1 | Chelsea–Sunderland / train | `chelsea_sunderland/detection_annotator.html` | 120 |
| 2 | Norway–Wales / train | `norway_wales/detection_annotator.html` | 120 |
| 3 | Man United–Aston Villa / validation | `man_utd_4_4_aston_villa_highlights/detection_annotator.html` | 120 |
| 4 | SoccerTrack 132831 panoramic / train-only context | `soccertrack_132831_panorama_clip/detection_annotator.html` | 40 |

The SoccerTrack clip was previously inspected in Sprint 3.4. It improves panoramic small-object coverage but is **not** an untouched final test match.

## Check progress and export

```powershell
python scripts/export_sprint3_5_yolo_dataset.py
Get-Content outputs/sprint3_5/training_gate_status.json
```

The gate is eligible only after all configured review frames are manually marked, at least 100 active-ball boxes exist overall, including at least 80 in train and 20 in validation, and every image/box passes validation. `verified=true` records a reviewer attestation; it cannot itself prove that a human inspected the image, so only mark it after inspection.

When the status says `training_eligible: true`, create the local, gitignored YOLO dataset:

```powershell
python scripts/export_sprint3_5_yolo_dataset.py --export
```

Then, and only then, run the configured candidate:

```powershell
python scripts/train_sprint3_5_yolo.py --execute
```

The automated test `test_yolo_export_requires_manual_match_disjoint_review` already exercises this exact export path using a two-image synthetic fixture. That fixture is test-only and never contributes to the real review counts, dataset, model, or metrics.
