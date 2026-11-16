"""Time-based possession aggregation separating attributed team share from total analysed time distribution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.config import MatchConfig
from src.schemas import BallObservationMode, InternalLabel, PossessionFrameRecord, PossessionState


def compute_possession_summary(
    records: Sequence[PossessionFrameRecord],
    match_config: MatchConfig,
    total_video_duration_seconds: Optional[float] = None,
    fps: float = 25.0,
    frame_stride: int = 2,
) -> Dict[str, Any]:
    """
    Aggregate time-based possession statistics from per-frame `PossessionFrameRecord` telemetry.

    Explicitly separates:
    1. `attributed_possession_share`:
       Team A / Team B percentage of the combined confidently attributed team control time
       (`T_A / (T_A + T_B)`).
    2. `total_analysed_time_distribution`:
       Proportion of total analysed segment duration spent in each of the 6 states
       (`TEAM_A_CONTROL`, `TEAM_B_CONTROL`, `CONTESTED`, `LOOSE_BALL`, `UNKNOWN`, `OUT_OF_PLAY`).
    """
    state_durations: Dict[str, float] = {st: 0.0 for st in PossessionState.all_states()}
    state_frame_counts: Dict[str, int] = {st: 0 for st in PossessionState.all_states()}

    observed_ball_seconds = 0.0
    interpolated_ball_seconds = 0.0
    lost_ball_seconds = 0.0
    transitions = 0
    camera_cuts_count = 0
    prev_state: Optional[str] = None

    # Partition records into continuous camera segments separated by camera cuts
    raw_segments: list[list[PossessionFrameRecord]] = []
    current_seg: list[PossessionFrameRecord] = []

    for rec in records:
        dt = max(0.0, float(rec.dt_seconds))
        st = rec.possession_state if rec.possession_state in state_durations else PossessionState.UNKNOWN.value
        state_durations[st] += dt
        state_frame_counts[st] += 1

        if rec.ball_observation_mode == BallObservationMode.OBSERVED.value:
            observed_ball_seconds += dt
        elif rec.ball_observation_mode == BallObservationMode.INTERPOLATED.value:
            interpolated_ball_seconds += dt
        else:
            lost_ball_seconds += dt

        if rec.camera_cut and current_seg:
            camera_cuts_count += 1
            raw_segments.append(current_seg)
            current_seg = [rec]
            prev_state = st
        else:
            if prev_state is not None and st != prev_state:
                transitions += 1
            prev_state = st
            current_seg.append(rec)

    if current_seg:
        raw_segments.append(current_seg)

    camera_segments = []
    for idx, seg in enumerate(raw_segments, start=1):
        seg_dur = sum(max(0.0, float(r.dt_seconds)) for r in seg)
        seg_a = sum(
            max(0.0, float(r.dt_seconds))
            for r in seg
            if r.possession_state == PossessionState.TEAM_A_CONTROL.value
        )
        seg_b = sum(
            max(0.0, float(r.dt_seconds))
            for r in seg
            if r.possession_state == PossessionState.TEAM_B_CONTROL.value
        )
        seg_attr = seg_a + seg_b
        camera_segments.append(
            {
                "segment_index": idx,
                "start_frame": int(seg[0].frame_number),
                "end_frame": int(seg[-1].frame_number),
                "start_timestamp": round(float(seg[0].timestamp), 2),
                "end_timestamp": round(float(seg[-1].timestamp), 2),
                "frames_count": len(seg),
                "duration_seconds": round(seg_dur, 2),
                "team_a_controlled_seconds": round(seg_a, 2),
                "team_b_controlled_seconds": round(seg_b, 2),
                "attributed_coverage_pct": round((seg_attr / seg_dur * 100.0) if seg_dur > 0 else 0.0, 2),
            }
        )
    longest_continuous_sec = max((s["duration_seconds"] for s in camera_segments), default=0.0)

    total_analysed_seconds = sum(state_durations.values())
    team_a_sec = state_durations[PossessionState.TEAM_A_CONTROL.value]
    team_b_sec = state_durations[PossessionState.TEAM_B_CONTROL.value]
    contested_sec = state_durations[PossessionState.CONTESTED.value]
    loose_sec = state_durations[PossessionState.LOOSE_BALL.value]
    unknown_sec = state_durations[PossessionState.UNKNOWN.value]
    out_of_play_sec = state_durations[PossessionState.OUT_OF_PLAY.value]

    attributed_sec = team_a_sec + team_b_sec
    possession_coverage = attributed_sec / total_analysed_seconds if total_analysed_seconds > 0 else 0.0

    if attributed_sec > 0:
        team_a_share_pct: Optional[float] = round((team_a_sec / attributed_sec) * 100.0, 2)
        team_b_share_pct: Optional[float] = round((team_b_sec / attributed_sec) * 100.0, 2)
    else:
        team_a_share_pct = None
        team_b_share_pct = None

    total_distribution_pct: Dict[str, float] = {}
    for st, sec in state_durations.items():
        pct = (sec / total_analysed_seconds * 100.0) if total_analysed_seconds > 0 else 0.0
        total_distribution_pct[st] = round(pct, 2)

    video_dur = (
        float(total_video_duration_seconds)
        if total_video_duration_seconds and total_video_duration_seconds > 0
        else total_analysed_seconds
    )
    segment_fraction = min(1.0, total_analysed_seconds / max(1e-6, video_dur)) if video_dur > 0 else 0.0
    is_full_match = bool(video_dur >= 50.0 * 60.0 and segment_fraction >= 0.90)

    team_a_name = match_config.to_display_label(InternalLabel.TEAM_A.value)
    team_b_name = match_config.to_display_label(InternalLabel.TEAM_B.value)

    return {
        "match_id": match_config.match_id,
        "match_name": match_config.match_name,
        "segment_scope": "full_match" if is_full_match else "partial_video_or_highlight_segment",
        "scope_disclaimer": (
            "Statistics reflect ONLY the analysed video segment and must not be reported as "
            "full 90-minute match possession."
            if not is_full_match
            else "Full match duration analysed."
        ),
        "frames_processed": len(records),
        "video_fps": round(float(fps), 2),
        "frame_stride": int(frame_stride),
        "effective_dt_seconds": round(float(frame_stride) / max(1.0, float(fps)), 4),
        "analysed_duration_seconds": round(total_analysed_seconds, 2),
        "total_video_duration_seconds": round(video_dur, 2),
        "analysed_video_fraction": round(segment_fraction, 4),
        "durations_seconds": {
            "team_a_controlled": round(team_a_sec, 2),
            "team_b_controlled": round(team_b_sec, 2),
            "attributed_total": round(attributed_sec, 2),
            "contested": round(contested_sec, 2),
            "loose_ball": round(loose_sec, 2),
            "unknown": round(unknown_sec, 2),
            "out_of_play": round(out_of_play_sec, 2),
        },
        "frame_counts_by_state": state_frame_counts,
        "ball_visibility_seconds": {
            "observed": round(observed_ball_seconds, 2),
            "interpolated": round(interpolated_ball_seconds, 2),
            "lost": round(lost_ball_seconds, 2),
        },
        "possession_coverage": round(possession_coverage, 4),
        "possession_coverage_pct": round(possession_coverage * 100.0, 2),
        "attributed_possession_share": {
            "team_a_name": team_a_name,
            "team_b_name": team_b_name,
            "team_a_pct": team_a_share_pct,
            "team_b_pct": team_b_share_pct,
            "definition": "T_team / (T_Team_A + T_Team_B) over confidently attributed control intervals only.",
        },
        "total_analysed_time_distribution_pct": total_distribution_pct,
        "state_transitions_count": transitions,
        "camera_cuts_count": camera_cuts_count,
        "longest_continuous_segment_seconds": round(longest_continuous_sec, 2),
        "camera_segments": camera_segments,
    }

