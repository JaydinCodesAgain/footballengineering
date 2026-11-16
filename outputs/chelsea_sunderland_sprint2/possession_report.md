# Team Possession Analytics Report — Chelsea vs Sunderland (`chelsea_sunderland`)
**Generated:** 2026-11-16T17:26:11Z

> **Segment Scope Notice (`partial_video_or_highlight_segment`):** Statistics reflect ONLY the analysed video segment and must not be reported as full 90-minute match possession. Analysed **3.60s** of **61.30s** total video duration (45 frames at FPS=25.0, stride=2).

## 1. Attributed Possession Share
*Computed strictly over intervals where one team has verified ball control (`T_team / (T_Team_A + T_Team_B)`).*

| Team | Controlled Time (s) | Attributed Possession Share |
|---|---:|---:|
| **Chelsea** (`Team_A`) | 0.24s | **75.00%** |
| **Sunderland** (`Team_B`) | 0.08s | **25.00%** |
| **Total Confidently Attributed** | 0.32s | **Coverage: 8.89%** |

## 2. Total Analysed Time Distribution (All 6 States)
*Includes contested duels, loose/in-flight balls, unknown intervals, and out-of-play time without silently assigning them to either team.*

| Possession State | Duration (s) | Frame Count | Share of Total Analysed Time (%) |
|---|---:|---:|---:|
| `TEAM_A_CONTROL` | 0.24s | 3 | 6.67% |
| `TEAM_B_CONTROL` | 0.08s | 1 | 2.22% |
| `CONTESTED` | 0.40s | 5 | 11.11% |
| `LOOSE_BALL` | 2.00s | 25 | 55.56% |
| `UNKNOWN` | 0.88s | 11 | 24.44% |
| `OUT_OF_PLAY` | 0.00s | 0 | 0.00% |

## 3. Ball Tracking Provenance Summary
- **Observed Ball Frames (Direct Detector Hit):** 30 (66.67%)
- **Interpolated Ball Frames (Kalman Coasting):** 8
- **Lost Ball Frames:** 7
- **Combined Tracked/Coasting Coverage:** 84.44%
- **Camera-Cut Resets Triggered:** 0
- **Outlier Ball Jumps Rejected:** 8
- **Possession State Transitions:** 16

## 4. Ground-Truth Possession Validation
- **Status:** Pending manual interval verification (Manual ground-truth possession intervals pending.)
