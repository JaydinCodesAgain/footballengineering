"""Player-to-ball proximity calculation, scale normalization, and per-frame ownership scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import PossessionConfig
from src.schemas import (
    BallObservationMode,
    BallTrajectoryRecord,
    FrameObservationRecord,
    InternalLabel,
    PlayerProximityRecord,
    PossessionState,
)


def compute_player_foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """
    Approximate a player's ground contact point `(foot_x, foot_y)` and bounding-box height `h`.
    Because broadcast cameras view players from an elevated angle, feet sit near `y2 - 0.05 * h`.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    foot_x = float(x1) + 0.5 * w
    foot_y = float(y2) - 0.05 * h
    return foot_x, foot_y, h


def compute_normalized_ball_distance(
    player_bbox: Tuple[int, int, int, int],
    ball_center: Tuple[float, float],
    min_scale_px: float = 30.0,
) -> Tuple[float, float, bool]:
    """
    Calculate Euclidean pixel distance and perspective-normalized distance between a player's
    foot region and the ball centre.

    Returns:
        `(pixel_distance, normalized_distance, is_aerial_ball)`
    """
    foot_x, foot_y, h = compute_player_foot_position(player_bbox)
    bx, by = ball_center

    # Also consider lower-shin control point (y2 - 0.15 * h) when ball is slightly off the turf
    shin_y = float(player_bbox[3]) - 0.15 * h
    d_foot = math.hypot(bx - foot_x, by - foot_y)
    d_shin = math.hypot(bx - foot_x, by - shin_y)
    px_dist = min(d_foot, d_shin)

    scale = max(float(min_scale_px), h)
    norm_dist = px_dist / scale

    # Ball is considered aerial relative to this player if its center is above chest level
    is_aerial = by < (float(player_bbox[1]) + 0.35 * h)
    return float(px_dist), float(norm_dist), bool(is_aerial)


def is_ball_out_of_play_bounds(
    ball_center: Tuple[float, float],
    frame_shape: Tuple[int, int],
    config: PossessionConfig,
) -> bool:
    """Check whether the ball centre falls outside the configured broadcast pitch region."""
    fh, fw = frame_shape[:2]
    if fh <= 0 or fw <= 0:
        return False
    bx, by = ball_center
    x_frac = bx / float(fw)
    y_frac = by / float(fh)
    return (
        x_frac < config.pitch_x_min_frac
        or x_frac > config.pitch_x_max_frac
        or y_frac < config.pitch_y_min_frac
        or y_frac > config.pitch_y_max_frac
    )


def evaluate_frame_ownership_candidates(
    frame_observations: Sequence[FrameObservationRecord],
    ball_record: BallTrajectoryRecord,
    config: Optional[PossessionConfig] = None,
    frame_shape: Optional[Tuple[int, int]] = None,
    prev_controlling_player_id: Optional[int] = None,
    prev_controlling_team_id: Optional[str] = None,
) -> Tuple[List[PlayerProximityRecord], Dict[str, Any]]:
    """
    Evaluate all players in a frame relative to the ball and determine the raw frame interaction state.

    Rules enforced:
    - Referees (`InternalLabel.REFEREE`) are strictly ineligible for ball ownership.
    - Unclassified / Unknown players (including goalkeepers unless explicitly mapped) are ineligible
      for team attribution; if an Unknown player is in close control of the ball, the raw state is
      `UNKNOWN` rather than inventing team control.
    - If both Team A and Team B have players within `contested_distance_threshold` with normalized
      distances differing by at most `contested_margin_threshold`, the state is `CONTESTED`.
    """
    cfg = config or PossessionConfig()
    proximities: List[PlayerProximityRecord] = []

    ball_visible = ball_record.observation_mode in (
        BallObservationMode.OBSERVED.value,
        BallObservationMode.INTERPOLATED.value,
    ) and (ball_record.center_x is not None and ball_record.center_y is not None)

    if not ball_visible:
        for obs in frame_observations:
            fx, fy, ph = compute_player_foot_position(obs.player_bbox)
            elig = obs.team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
            proximities.append(
                PlayerProximityRecord(
                    match_id=obs.match_id,
                    frame_number=obs.frame_number,
                    timestamp=obs.timestamp,
                    player_id=obs.player_id,
                    team_id=obs.team_id,
                    team_display_name=obs.team_display_name,
                    foot_x=round(fx, 2),
                    foot_y=round(fy, 2),
                    player_height_px=round(ph, 2),
                    pixel_distance_to_ball=None,
                    normalized_distance_to_ball=None,
                    is_aerial_ball=False,
                    eligible_for_ownership=elig,
                    control_score=0.0,
                )
            )
        return proximities, {
            "raw_state": PossessionState.UNKNOWN.value,
            "controlling_player_id": None,
            "controlling_team_id": None,
            "confidence": 0.0,
            "closest_player_id": None,
            "closest_team_id": None,
            "closest_norm_dist": None,
            "second_closest_team_id": None,
            "second_closest_norm_dist": None,
        }

    assert ball_record.center_x is not None and ball_record.center_y is not None
    ball_center = (float(ball_record.center_x), float(ball_record.center_y))

    # Reject non-tactical player close-up / celebration shots where a single player box dominates the screen
    if frame_shape is not None and frame_observations:
        fh, fw = frame_shape[:2]
        frame_area = float(max(1, fh * fw))
        p_heights = [float(o.player_bbox[3] - o.player_bbox[1]) for o in frame_observations]
        max_p_area = max(
            float((o.player_bbox[2] - o.player_bbox[0]) * (o.player_bbox[3] - o.player_bbox[1]))
            for o in frame_observations
        )
        med_p_h = float(sum(sorted(p_heights)[len(p_heights) // 2 : len(p_heights) // 2 + 1]))
        if (fh > 0 and med_p_h > 0.44 * fh) or (max_p_area / frame_area > 0.18):
            for obs in frame_observations:
                fx, fy, ph = compute_player_foot_position(obs.player_bbox)
                px_d, norm_d, aerial = compute_normalized_ball_distance(obs.player_bbox, ball_center)
                elig = obs.team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
                proximities.append(
                    PlayerProximityRecord(
                        match_id=obs.match_id,
                        frame_number=obs.frame_number,
                        timestamp=obs.timestamp,
                        player_id=obs.player_id,
                        team_id=obs.team_id,
                        team_display_name=obs.team_display_name,
                        foot_x=round(fx, 2),
                        foot_y=round(fy, 2),
                        player_height_px=round(ph, 2),
                        pixel_distance_to_ball=round(px_d, 2),
                        normalized_distance_to_ball=round(norm_d, 4),
                        is_aerial_ball=aerial,
                        eligible_for_ownership=elig,
                        control_score=0.0,
                    )
                )
            return proximities, {
                "raw_state": PossessionState.UNKNOWN.value,
                "controlling_player_id": None,
                "controlling_team_id": None,
                "confidence": 0.0,
                "reason": "non_tactical_closeup_frame",
                "closest_player_id": None,
                "closest_team_id": None,
                "closest_norm_dist": None,
                "second_closest_team_id": None,
                "second_closest_norm_dist": None,
                "note": "non_tactical_closeup_camera_shot",
            }

    if frame_shape is not None and is_ball_out_of_play_bounds(ball_center, frame_shape, cfg):
        for obs in frame_observations:
            fx, fy, ph = compute_player_foot_position(obs.player_bbox)
            px_d, norm_d, aerial = compute_normalized_ball_distance(obs.player_bbox, ball_center)
            elig = obs.team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
            proximities.append(
                PlayerProximityRecord(
                    match_id=obs.match_id,
                    frame_number=obs.frame_number,
                    timestamp=obs.timestamp,
                    player_id=obs.player_id,
                    team_id=obs.team_id,
                    team_display_name=obs.team_display_name,
                    foot_x=round(fx, 2),
                    foot_y=round(fy, 2),
                    player_height_px=round(ph, 2),
                    pixel_distance_to_ball=round(px_d, 2),
                    normalized_distance_to_ball=round(norm_d, 4),
                    is_aerial_ball=aerial,
                    eligible_for_ownership=elig,
                    control_score=0.0,
                )
            )
        return proximities, {
            "raw_state": PossessionState.OUT_OF_PLAY.value,
            "controlling_player_id": None,
            "controlling_team_id": None,
            "confidence": round(float(ball_record.confidence), 4),
            "closest_player_id": None,
            "closest_team_id": None,
            "closest_norm_dist": None,
            "second_closest_team_id": None,
            "second_closest_norm_dist": None,
        }

    ball_conf_factor = max(0.35, min(1.0, float(ball_record.confidence) / 0.35))
    if ball_record.is_interpolated:
        ball_conf_factor *= 0.80

    # Scale speed threshold by observed player height (perspective/zoom invariance)
    p_heights_all = [
        float(o.player_bbox[3] - o.player_bbox[1])
        for o in frame_observations
        if (o.player_bbox[3] - o.player_bbox[1]) > 15
    ]
    if p_heights_all:
        s_sorted = sorted(p_heights_all)
        med_ph_scale = float(s_sorted[len(s_sorted) // 2])
        scale_mult = max(0.65, min(1.75, med_ph_scale / 80.0))
    else:
        scale_mult = 1.0
    effective_speed_thresh = float(cfg.loose_ball_speed_threshold_px_s) * scale_mult

    speed = float(ball_record.speed_px_per_sec)
    speed_factor = (
        1.0
        if speed <= effective_speed_thresh * 0.65
        else max(0.35, 1.0 - (speed / max(1.0, effective_speed_thresh * 1.5)))
    )

    for obs in frame_observations:
        fx, fy, ph = compute_player_foot_position(obs.player_bbox)
        px_d, norm_d, aerial = compute_normalized_ball_distance(obs.player_bbox, ball_center)

        eff_team_id = obs.team_id
        conf_mult = 1.0
        # Only fall back to raw team prediction for brand-new tracks still warming up
        # (`track_team_confidence == 0.0`), never overriding a stabilizer disagreement/rejection.
        if (
            eff_team_id == InternalLabel.UNKNOWN.value
            and float(obs.track_team_confidence or 0.0) <= 0.0
            and obs.raw_team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
            and float(obs.classification_confidence or 0.0) >= 0.72
        ):
            eff_team_id = obs.raw_team_id
            conf_mult = 0.78

        if eff_team_id == InternalLabel.REFEREE.value and cfg.exclude_referees:
            elig = False
        elif eff_team_id in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value):
            elig = True
        else:
            elig = False

        if not elig or norm_d > cfg.control_distance_threshold * 1.35:
            c_score = 0.0
        else:
            rel_d = norm_d / max(1e-4, cfg.control_distance_threshold)
            dist_term = max(0.22, 1.0 - 0.65 * (rel_d ** 2)) if rel_d <= 1.0 else max(0.0, 0.35 * (1.35 - rel_d))
            team_conf = max(
                0.35,
                float(obs.track_team_confidence or obs.classification_confidence or 0.55),
            ) * conf_mult
            aerial_mult = 0.55 if aerial else 1.0
            c_score = dist_term * speed_factor * aerial_mult * min(1.0, team_conf) * ball_conf_factor

            # Continuity bonus for established controller
            if norm_d <= cfg.control_distance_threshold:
                if prev_controlling_player_id is not None and obs.player_id == prev_controlling_player_id:
                    c_score = min(1.0, c_score + 0.14)
                elif prev_controlling_team_id is not None and eff_team_id == prev_controlling_team_id:
                    c_score = min(1.0, c_score + 0.06)

        proximities.append(
            PlayerProximityRecord(
                match_id=obs.match_id,
                frame_number=obs.frame_number,
                timestamp=obs.timestamp,
                player_id=obs.player_id,
                team_id=eff_team_id,
                team_display_name=obs.team_display_name,
                foot_x=round(fx, 2),
                foot_y=round(fy, 2),
                player_height_px=round(ph, 2),
                pixel_distance_to_ball=round(px_d, 2),
                normalized_distance_to_ball=round(norm_d, 4),
                is_aerial_ball=aerial,
                eligible_for_ownership=elig,
                control_score=round(float(c_score), 4),
            )
        )

    # Sort all non-referee players by normalized distance
    non_ref = [
        p
        for p in proximities
        if p.normalized_distance_to_ball is not None
        and not (p.team_id == InternalLabel.REFEREE.value and cfg.exclude_referees)
    ]
    non_ref.sort(
        key=lambda p: float(p.normalized_distance_to_ball)
        if p.normalized_distance_to_ball is not None
        else 999.0
    )

    closest_overall = non_ref[0] if non_ref else None
    second_overall = non_ref[1] if len(non_ref) > 1 else None

    # Best candidate per positive team
    team_a_cands = [p for p in non_ref if p.team_id == InternalLabel.TEAM_A.value]
    team_b_cands = [p for p in non_ref if p.team_id == InternalLabel.TEAM_B.value]
    best_a = team_a_cands[0] if team_a_cands else None
    best_b = team_b_cands[0] if team_b_cands else None

    dist_a = float(best_a.normalized_distance_to_ball) if best_a and best_a.normalized_distance_to_ball is not None else 999.0
    dist_b = float(best_b.normalized_distance_to_ball) if best_b and best_b.normalized_distance_to_ball is not None else 999.0

    # 1. Check if an Unknown player (e.g. Goalkeeper or unclassified player) clearly controls the ball
    if (
        closest_overall is not None
        and closest_overall.team_id == InternalLabel.UNKNOWN.value
        and closest_overall.normalized_distance_to_ball is not None
        and float(closest_overall.normalized_distance_to_ball) <= cfg.control_distance_threshold
        and not cfg.attribute_unknown_goalkeepers
    ):
        if float(closest_overall.normalized_distance_to_ball) + 0.10 < min(dist_a, dist_b):
            return proximities, {
                "raw_state": PossessionState.UNKNOWN.value,
                "controlling_player_id": closest_overall.player_id,
                "controlling_team_id": None,
                "confidence": 0.0,
                "closest_player_id": closest_overall.player_id,
                "closest_team_id": closest_overall.team_id,
                "closest_norm_dist": closest_overall.normalized_distance_to_ball,
                "second_closest_team_id": second_overall.team_id if second_overall else None,
                "second_closest_norm_dist": second_overall.normalized_distance_to_ball if second_overall else None,
            }

    # 2. Check for CONTESTED duel between Team A and Team B
    if (
        dist_a <= cfg.contested_distance_threshold
        and dist_b <= cfg.contested_distance_threshold
        and abs(dist_a - dist_b) <= cfg.contested_margin_threshold
    ):
        conf_duel = round(max(best_a.control_score if best_a else 0.0, best_b.control_score if best_b else 0.0), 4)
        return proximities, {
            "raw_state": PossessionState.CONTESTED.value,
            "controlling_player_id": None,
            "controlling_team_id": None,
            "confidence": conf_duel,
            "closest_player_id": closest_overall.player_id if closest_overall else None,
            "closest_team_id": closest_overall.team_id if closest_overall else None,
            "closest_norm_dist": closest_overall.normalized_distance_to_ball if closest_overall else None,
            "second_closest_team_id": second_overall.team_id if second_overall else None,
            "second_closest_norm_dist": second_overall.normalized_distance_to_ball if second_overall else None,
        }

    # 3. Check for single-team control vs LOOSE_BALL
    eligible_cands = [p for p in non_ref if p.eligible_for_ownership]
    eligible_cands.sort(key=lambda p: p.control_score, reverse=True)
    top_cand = eligible_cands[0] if eligible_cands else None

    if top_cand is not None and top_cand.normalized_distance_to_ball is not None:
        d_top = float(top_cand.normalized_distance_to_ball)
        if d_top <= cfg.control_distance_threshold:
            # High-speed ball in flight not directly at feet -> LOOSE_BALL
            if speed > effective_speed_thresh and d_top > 0.32:
                raw_st = PossessionState.LOOSE_BALL.value
                ctrl_pid = None
                ctrl_tid = None
                conf_val = round(float(ball_record.confidence), 4)
            elif top_cand.control_score < cfg.min_ownership_confidence:
                raw_st = PossessionState.UNKNOWN.value
                ctrl_pid = top_cand.player_id
                ctrl_tid = None
                conf_val = round(float(top_cand.control_score), 4)
            else:
                raw_st = (
                    PossessionState.TEAM_A_CONTROL.value
                    if top_cand.team_id == InternalLabel.TEAM_A.value
                    else PossessionState.TEAM_B_CONTROL.value
                )
                ctrl_pid = top_cand.player_id
                ctrl_tid = top_cand.team_id
                conf_val = round(float(top_cand.control_score), 4)

            return proximities, {
                "raw_state": raw_st,
                "controlling_player_id": ctrl_pid,
                "controlling_team_id": ctrl_tid,
                "confidence": conf_val,
                "closest_player_id": closest_overall.player_id if closest_overall else None,
                "closest_team_id": closest_overall.team_id if closest_overall else None,
                "closest_norm_dist": closest_overall.normalized_distance_to_ball if closest_overall else None,
                "second_closest_team_id": second_overall.team_id if second_overall else None,
                "second_closest_norm_dist": second_overall.normalized_distance_to_ball if second_overall else None,
            }

    # Otherwise the ball is visible on the pitch but not controlled by any player -> LOOSE_BALL
    return proximities, {
        "raw_state": PossessionState.LOOSE_BALL.value,
        "controlling_player_id": None,
        "controlling_team_id": None,
        "confidence": round(float(ball_record.confidence), 4),
        "closest_player_id": closest_overall.player_id if closest_overall else None,
        "closest_team_id": closest_overall.team_id if closest_overall else None,
        "closest_norm_dist": closest_overall.normalized_distance_to_ball if closest_overall else None,
        "second_closest_team_id": second_overall.team_id if second_overall else None,
        "second_closest_norm_dist": second_overall.normalized_distance_to_ball if second_overall else None,
    }
