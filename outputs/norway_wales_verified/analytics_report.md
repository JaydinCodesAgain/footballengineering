# Football AI Analytics Report — Norway vs Wales (`norway_wales`)
**Generated:** 2026-11-16T12:58:49Z

## 1. Detection Summary
- **Total Frames Processed:** 60
- **Average Players / Frame:** 3.72
- **Frames with Ball Detected:** 33 (55.0%)
- **Bounding-Box IoU Evaluation:** Unavailable (awaiting manually verified ground-truth bounding boxes).

## 2. Team Classification Summary
- **Feature Method:** `foreground_masked`
- **Classifier:** `knn` (threshold=0.2943)
- **Training Dataset Size:** 56 (`{'Referee': 12, 'Team_B': 21, 'Team_A': 23}`)
- **Held-Out Evaluation Accuracy:** 93.33%
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
- **Distinct Tracking IDs:** 22
- **Tracks < 5 Observations:** 5
- **Mean Track Lifetime (obs):** 10.14
- **Total Player Observations:** 223
- **Raw Team-Label Changes:** 0
- **Smoothed Team-Label Changes:** 0
- **Stable-Label Coverage:** 69.51%
- **Identity Resets Triggered:** 0
