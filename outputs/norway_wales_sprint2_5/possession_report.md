# Team Possession Analytics Report — Norway vs Wales (`norway_wales`)
**Generated:** 2026-11-16T19:17:39Z

> **Segment Scope Notice (`partial_video_or_highlight_segment`):** Statistics reflect ONLY the analysed video segment and must not be reported as full 90-minute match possession. Analysed **60.00s** of **118.12s** total video duration (375 frames at FPS=25.0, stride=4).

## 1. Attributed Possession Share
*Computed strictly over intervals where one team has verified ball control (`T_team / (T_Team_A + T_Team_B)`).*

| Team | Controlled Time (s) | Attributed Possession Share |
|---|---:|---:|
| **Norway** (`Team_A`) | 2.40s | **93.75%** |
| **Wales** (`Team_B`) | 0.16s | **6.25%** |
| **Total Confidently Attributed** | 2.56s | **Coverage: 4.27%** |

## 2. Total Analysed Time Distribution (All 6 States)
*Includes contested duels, loose/in-flight balls, unknown intervals, and out-of-play time without silently assigning them to either team.*

| Possession State | Duration (s) | Frame Count | Share of Total Analysed Time (%) |
|---|---:|---:|---:|
| `TEAM_A_CONTROL` | 2.40s | 15 | 4.00% |
| `TEAM_B_CONTROL` | 0.16s | 1 | 0.27% |
| `CONTESTED` | 0.00s | 0 | 0.00% |
| `LOOSE_BALL` | 17.76s | 111 | 29.60% |
| `UNKNOWN` | 37.76s | 236 | 62.93% |
| `OUT_OF_PLAY` | 1.92s | 12 | 3.20% |

## 3. Ball Tracking Provenance Summary
- **Observed Ball Frames (Direct Detector Hit):** 172 (45.87%)
- **Interpolated Ball Frames (Kalman Coasting):** 60
- **Lost Ball Frames:** 143
- **Combined Tracked/Coasting Coverage:** 61.87%
- **Camera-Cut Resets Triggered:** 11
- **Outlier Ball Jumps Rejected:** 71
- **Static False-Track Associations Broken:** 1 (blacklisted static zones: 1)
- **Ball Reacquisition Events:** 21
- **Continuous Track Length (Mean / Max Frames):** 7.48 / 39
- **Possession State Transitions:** 67

### Continuous Camera Segments (Cut-Separated)
- **Detected Camera Cuts:** 11
- **Longest Continuous Shot:** 12.00s

| Shot # | Frame Range | Timestamp Range | Duration (s) | Team A (s) | Team B (s) | Attributed Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | `424–424` | `16.96s–16.96s` | 0.16s | 0.00s | 0.00s | 0.0% |
| 2 | `428–636` | `17.12s–25.44s` | 8.48s | 0.00s | 0.00s | 0.0% |
| 3 | `640–708` | `25.60s–28.32s` | 2.88s | 0.00s | 0.00s | 0.0% |
| 4 | `712–756` | `28.48s–30.24s` | 1.92s | 0.00s | 0.00s | 0.0% |
| 5 | `760–872` | `30.40s–34.88s` | 4.64s | 0.00s | 0.00s | 0.0% |
| 6 | `876–876` | `35.04s–35.04s` | 0.16s | 0.00s | 0.00s | 0.0% |
| 7 | `880–1176` | `35.20s–47.04s` | 12.00s | 1.76s | 0.00s | 14.7% |
| 8 | `1180–1240` | `47.20s–49.60s` | 2.56s | 0.00s | 0.00s | 0.0% |
| 9 | `1244–1376` | `49.76s–55.04s` | 5.44s | 0.00s | 0.16s | 2.9% |
| 10 | `1380–1640` | `55.20s–65.60s` | 10.56s | 0.00s | 0.00s | 0.0% |
| 11 | `1644–1856` | `65.76s–74.24s` | 8.64s | 0.64s | 0.00s | 7.4% |
| 12 | `1860–1920` | `74.40s–76.80s` | 2.56s | 0.00s | 0.00s | 0.0% |

## 4. Ground-Truth Possession Validation
- **Status:** Pending manual interval verification (`validation_blocker=missing_verified_manual_possession_annotations`)
- **Details:** No manually verified ground-truth possession intervals are present yet. Open `data/annotations/<match_id>/possession_annotator.html` or edit `possession_annotations.csv` with `verified=true` and `source=manual` to compute ground-truth possession accuracy.
