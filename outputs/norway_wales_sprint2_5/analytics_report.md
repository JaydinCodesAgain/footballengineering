# Football AI Analytics Report — Norway vs Wales (`norway_wales`)
**Generated:** 2026-11-16T19:19:11Z

## 1. Detection Summary
- **Total Frames Processed:** 375
- **Average Players / Frame:** 5.53
- **Frames with Ball Detected (Direct Observation):** 172 (45.9% coverage — distinct from ground-truth recall)
- **Bounding-Box IoU Evaluation:** Unavailable (awaiting manually verified ground-truth bounding boxes).

## 2. Team Classification Summary
- **Feature Method:** `foreground_masked`
- **Classifier:** `knn` (threshold=0.2943)
- **Training Dataset Size:** 56 (`{'Referee': 12, 'Team_B': 21, 'Team_A': 23}`)
- **Held-Out Validation Accuracy:** 93.33%
- **Selective Accuracy (Non-Unknown):** 100.00%
- **Macro F1 (All Classes):** 91.86%
- **Macro F1 (Positive Classes):** 93.90%
- **Classification Coverage:** 73.33% (Unknown Rate: 26.67%)

### Per-Class Metrics
| Internal Label | Display Name | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| `Team_A` | Norway | 100.00% | 92.31% | 96.00% | 13 |
| `Team_B` | Wales | 100.00% | 100.00% | 100.00% | 7 |
| `Referee` | REF | 100.00% | 75.00% | 85.71% | 4 |
| `Unknown` | Unknown | 75.00% | 100.00% | 85.71% | 6 |

## 3. Tracking & Temporal Stability Summary
- **Tracker Config:** `bytetrack.yaml`
- **Distinct Tracking IDs:** 215
- **Tracks < 5 Observations:** 84
- **Mean Track Lifetime (obs):** 9.65
- **Total Player Observations:** 2074
- **Raw Team-Label Changes:** 87
- **Smoothed Team-Label Changes:** 8
- **Stable-Label Coverage:** 70.93%
- **Identity Resets Triggered:** 44

## 4. Team Possession Summary (Segment Estimate)
- **Scope:** `partial_video_or_highlight_segment` (60.00s analysed)
- **Attributed Possession Share (Norway vs Wales):** **93.75%** vs **6.25%**
- **Possession Attribution Coverage:** 4.27%
- **Ball Observed Coverage:** 45.87% (+ 60 Kalman-interpolated frames)
