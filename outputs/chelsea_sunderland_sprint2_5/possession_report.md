# Team Possession Analytics Report — Chelsea vs Sunderland (`chelsea_sunderland`)
**Generated:** 2026-11-16T19:16:50Z

> **Segment Scope Notice (`partial_video_or_highlight_segment`):** Statistics reflect ONLY the analysed video segment and must not be reported as full 90-minute match possession. Analysed **60.00s** of **117.20s** total video duration (375 frames at FPS=25.0, stride=4).

## 1. Attributed Possession Share
*Computed strictly over intervals where one team has verified ball control (`T_team / (T_Team_A + T_Team_B)`).*

| Team | Controlled Time (s) | Attributed Possession Share |
|---|---:|---:|
| **Chelsea** (`Team_A`) | 1.28s | **17.39%** |
| **Sunderland** (`Team_B`) | 6.08s | **82.61%** |
| **Total Confidently Attributed** | 7.36s | **Coverage: 12.27%** |

## 2. Total Analysed Time Distribution (All 6 States)
*Includes contested duels, loose/in-flight balls, unknown intervals, and out-of-play time without silently assigning them to either team.*

| Possession State | Duration (s) | Frame Count | Share of Total Analysed Time (%) |
|---|---:|---:|---:|
| `TEAM_A_CONTROL` | 1.28s | 8 | 2.13% |
| `TEAM_B_CONTROL` | 6.08s | 38 | 10.13% |
| `CONTESTED` | 0.32s | 2 | 0.53% |
| `LOOSE_BALL` | 17.76s | 111 | 29.60% |
| `UNKNOWN` | 33.44s | 209 | 55.73% |
| `OUT_OF_PLAY` | 1.12s | 7 | 1.87% |

## 3. Ball Tracking Provenance Summary
- **Observed Ball Frames (Direct Detector Hit):** 200 (53.33%)
- **Interpolated Ball Frames (Kalman Coasting):** 43
- **Lost Ball Frames:** 132
- **Combined Tracked/Coasting Coverage:** 64.80%
- **Camera-Cut Resets Triggered:** 14
- **Outlier Ball Jumps Rejected:** 142
- **Static False-Track Associations Broken:** 0 (blacklisted static zones: 0)
- **Ball Reacquisition Events:** 9
- **Continuous Track Length (Mean / Max Frames):** 13.5 / 35
- **Possession State Transitions:** 83

### Continuous Camera Segments (Cut-Separated)
- **Detected Camera Cuts:** 14
- **Longest Continuous Shot:** 14.40s

| Shot # | Frame Range | Timestamp Range | Duration (s) | Team A (s) | Team B (s) | Attributed Coverage |
|---:|---|---|---:|---:|---:|---:|
| 1 | `180–536` | `7.20s–21.44s` | 14.40s | 0.64s | 4.64s | 36.7% |
| 2 | `540–608` | `21.60s–24.32s` | 2.88s | 0.00s | 0.00s | 0.0% |
| 3 | `612–616` | `24.48s–24.64s` | 0.32s | 0.00s | 0.00s | 0.0% |
| 4 | `620–768` | `24.80s–30.72s` | 6.08s | 0.00s | 0.80s | 13.2% |
| 5 | `772–772` | `30.88s–30.88s` | 0.16s | 0.00s | 0.00s | 0.0% |
| 6 | `776–1016` | `31.04s–40.64s` | 9.76s | 0.00s | 0.00s | 0.0% |
| 7 | `1020–1280` | `40.80s–51.20s` | 10.56s | 0.32s | 0.00s | 3.0% |
| 8 | `1284–1312` | `51.36s–52.48s` | 1.28s | 0.00s | 0.00s | 0.0% |
| 9 | `1316–1376` | `52.64s–55.04s` | 2.56s | 0.00s | 0.00s | 0.0% |
| 10 | `1380–1436` | `55.20s–57.44s` | 2.40s | 0.00s | 0.64s | 26.7% |
| 11 | `1440–1480` | `57.60s–59.20s` | 1.76s | 0.00s | 0.00s | 0.0% |
| 12 | `1484–1592` | `59.36s–63.68s` | 4.48s | 0.32s | 0.00s | 7.1% |
| 13 | `1596–1628` | `63.84s–65.12s` | 1.44s | 0.00s | 0.00s | 0.0% |
| 14 | `1632–1632` | `65.28s–65.28s` | 0.16s | 0.00s | 0.00s | 0.0% |
| 15 | `1636–1676` | `65.44s–67.04s` | 1.76s | 0.00s | 0.00s | 0.0% |

## 4. Ground-Truth Possession Validation
- **Status:** Pending manual interval verification (`validation_blocker=missing_verified_manual_possession_annotations`)
- **Details:** No manually verified ground-truth possession intervals are present yet. Open `data/annotations/<match_id>/possession_annotator.html` or edit `possession_annotations.csv` with `verified=true` and `source=manual` to compute ground-truth possession accuracy.
