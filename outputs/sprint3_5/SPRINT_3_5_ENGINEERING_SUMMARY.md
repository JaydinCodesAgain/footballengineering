# Sprint 3.5 — PR #7 training-readiness fixes

## Current outcome

No football-specific model was trained. Current verified review count is **0 / 400** and verified active-ball boxes are **0**; the training gate is correctly closed. Therefore the existing 74-frame benchmark remains unchanged: precision **79.41%**, recall **49.09%**, F1 **0.607**. No new accuracy metric is claimed.

## PR fixes

- Export now validates every reviewed box in image space before normalization: `0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`. It rejects negative, reversed, zero-area, and out-of-image boxes with match and frame context; it never clips labels.
- The gate now requires 300 reviewed frames, 100 active-ball boxes overall, and positive examples in **both** roles: at least 80 train and 20 validation. Unsupported splits, duplicate manifest sources, bad provenance, missing/corrupt images, invalid metadata, and invalid boxes block export.
- Normal preparation imports a reviewer CSV before writing anything, preserving CSV-only work. It refuses a changed frame selection. Destructive regeneration is available only with an explicit backup directory and exact confirmation phrase.
- The existing HTML reviewer now has zoom-aware native-pixel drawing, browser-progress save/restore, complete-CSV import, confirmed clearing, verified-negative support, and visible separation of dashed suggestions from manual boxes.
- The environment has the Ultralytics Python package and `yolo11s.pt`, but no `yolo` console executable. The repository launcher uses the Python API and is dry-run by default; `--execute` independently rechecks the reviewed-data gate.
- Added 40 unreviewed 3840×1504 SoccerTrack 132831 panoramic frames as train-only review context. The original 360 highlight frames remain unchanged. Total review workload is now 400 frames.

## Synthetic end-to-end proof

`test_yolo_export_requires_manual_match_disjoint_review` constructs one reviewed train image and one reviewed validation image, passes the manual-provenance and per-split-positive gates, exports a ball-only YOLO dataset, and checks the generated labels. It is a test fixture only and does not contribute to training readiness.

## Representativeness

`batch_representativeness_audit.json` records measurable dimensions, luminance summaries, and perceptual near-duplicate candidates. It does not guess whether a frame is a replay, close-up, transition, or contains a small football. Three sources remain 1280×720 highlights; the new SoccerTrack source is panoramic but previously inspected. A genuinely untouched, manually reviewed final test match is still required.

## Required next action

Follow `REVIEW_HANDOFF.md`: complete five Chelsea–Sunderland frames, export the CSV, and run the gate command. Continue only after that real review/export loop succeeds. Training, baseline comparison, possession re-evaluation, and production integration remain blocked until valid reviewed data exists.
