# Team Possession Analytics Report — Norway vs Wales (`norway_wales`)
**Generated:** 2026-11-16T17:26:12Z

> **Segment Scope Notice (`partial_video_or_highlight_segment`):** Statistics reflect ONLY the analysed video segment and must not be reported as full 90-minute match possession. Analysed **4.80s** of **118.00s** total video duration (58 frames at FPS=25.0, stride=2).

## 1. Attributed Possession Share
*Computed strictly over intervals where one team has verified ball control (`T_team / (T_Team_A + T_Team_B)`).*

| Team | Controlled Time (s) | Attributed Possession Share |
|---|---:|---:|
| **Norway** (`Team_A`) | 0.00s | **0.00%** |
| **Wales** (`Team_B`) | 1.60s | **100.00%** |
| **Total Confidently Attributed** | 1.60s | **Coverage: 33.33%** |

## 2. Total Analysed Time Distribution (All 6 States)
*Includes contested duels, loose/in-flight balls, unknown intervals, and out-of-play time without silently assigning them to either team.*

| Possession State | Duration (s) | Frame Count | Share of Total Analysed Time (%) |
|---|---:|---:|---:|
| `TEAM_A_CONTROL` | 0.00s | 0 | 0.00% |
| `TEAM_B_CONTROL` | 1.60s | 20 | 33.33% |
| `CONTESTED` | 0.00s | 0 | 0.00% |
| `LOOSE_BALL` | 0.88s | 11 | 18.33% |
| `UNKNOWN` | 2.32s | 27 | 48.33% |
| `OUT_OF_PLAY` | 0.00s | 0 | 0.00% |

## 3. Ball Tracking Provenance Summary
- **Observed Ball Frames (Direct Detector Hit):** 36 (62.07%)
- **Interpolated Ball Frames (Kalman Coasting):** 3
- **Lost Ball Frames:** 19
- **Combined Tracked/Coasting Coverage:** 67.24%
- **Camera-Cut Resets Triggered:** 0
- **Outlier Ball Jumps Rejected:** 0
- **Possession State Transitions:** 5

## 4. Ground-Truth Possession Validation
- **Status:** Pending manual interval verification (Manual ground-truth possession intervals pending.)
