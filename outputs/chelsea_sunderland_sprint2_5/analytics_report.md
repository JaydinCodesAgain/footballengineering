# Football AI Analytics Report — Chelsea vs Sunderland (`chelsea_sunderland`)
**Generated:** 2026-11-16T19:18:36Z

## 1. Detection Summary
- **Total Frames Processed:** 375
- **Average Players / Frame:** 5.93
- **Frames with Ball Detected (Direct Observation):** 200 (53.3% coverage — distinct from ground-truth recall)
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
- **Distinct Tracking IDs:** 256
- **Tracks < 5 Observations:** 102
- **Mean Track Lifetime (obs):** 8.68
- **Total Player Observations:** 2222
- **Raw Team-Label Changes:** 55
- **Smoothed Team-Label Changes:** 7
- **Stable-Label Coverage:** 53.15%
- **Identity Resets Triggered:** 65

## 4. Team Possession Summary (Segment Estimate)
- **Scope:** `partial_video_or_highlight_segment` (60.00s analysed)
- **Attributed Possession Share (Chelsea vs Sunderland):** **17.39%** vs **82.61%**
- **Possession Attribution Coverage:** 12.27%
- **Ball Observed Coverage:** 53.33% (+ 43 Kalman-interpolated frames)
