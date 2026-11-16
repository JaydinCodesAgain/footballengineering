# Football AI Analytics Report — Chelsea vs Sunderland (`chelsea_sunderland`)
**Generated:** 2026-11-16T17:27:32Z

## 1. Detection Summary
- **Total Frames Processed:** 45
- **Average Players / Frame:** 11.80
- **Frames with Ball Detected (Direct Observation):** 33 (73.3% coverage — distinct from ground-truth recall)
- **Bounding-Box IoU Evaluation:** Unavailable (awaiting manually verified ground-truth bounding boxes).

## 2. Team Classification Summary
- **Feature Method:** `spatial_embedding`
- **Classifier:** `knn` (threshold=0.2059)
- **Training Dataset Size:** 29 (`{'Team_A': 14, 'Team_B': 13, 'Referee': 2}`)
- **Held-Out Validation Accuracy:** 91.49%
- **Selective Accuracy (Non-Unknown):** 100.00%
- **Macro F1 (All Classes):** 92.92%
- **Macro F1 (Positive Classes):** 97.22%
- **Classification Coverage:** 74.47% (Unknown Rate: 25.53%)

### Per-Class Metrics
| Internal Label | Display Name | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| `Team_A` | Chelsea | 100.00% | 100.00% | 100.00% | 12 |
| `Team_B` | Sunderland | 100.00% | 84.62% | 91.67% | 26 |
| `Referee` | REF | 100.00% | 100.00% | 100.00% | 1 |
| `Unknown` | Unknown | 66.67% | 100.00% | 80.00% | 8 |

## 3. Tracking & Temporal Stability Summary
- **Tracker Config:** `bytetrack.yaml`
- **Distinct Tracking IDs:** 25
- **Tracks < 5 Observations:** 5
- **Mean Track Lifetime (obs):** 21.24
- **Total Player Observations:** 531
- **Raw Team-Label Changes:** 2
- **Smoothed Team-Label Changes:** 0
- **Stable-Label Coverage:** 82.11%
- **Identity Resets Triggered:** 0

## 4. Team Possession Summary (Segment Estimate)
- **Scope:** `partial_video_or_highlight_segment` (3.60s analysed)
- **Attributed Possession Share (Chelsea vs Sunderland):** **75.0%** vs **25.0%**
- **Possession Attribution Coverage:** 8.89%
- **Ball Observed Coverage:** 66.67% (+ 8 Kalman-interpolated frames)
