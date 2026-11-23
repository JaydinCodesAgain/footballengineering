"""Temporal possession state machine with hysteresis, pass reception handling, and camera-cut resets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.config import PossessionConfig
from src.schemas import InternalLabel, PossessionState


class PossessionStateMachine:
    """
    Finite state machine managing transitions between:
    `TEAM_A_CONTROL`, `TEAM_B_CONTROL`, `CONTESTED`, `LOOSE_BALL`, `UNKNOWN`, and `OUT_OF_PLAY`.

    Design principles:
    - Hysteresis (`min_control_frames`): prevents 1-frame proximity noise from flipping possession
      between opposing teams.
    - Same-team pass continuity (`max_pass_retention_frames`, `max_pass_flight_hold_frames`):
      when Team X has confirmed control and kicks the ball into flight (`LOOSE_BALL`), if the
      nearest receiving player within `2.2 * control_distance_threshold` belongs to Team X and no
      opponent is within `1.25 * control_distance_threshold`, bridges `TEAM_X_CONTROL` smoothly
      across short pass receptions while transitioning cleanly to `LOOSE_BALL` once flight exceeds
      `max_pass_flight_hold_frames`.
    - Conservative uncertain-ball decay (`max_ball_loss_hold_frames`): when the ball is `LOST` or
      unreliable (`ball_confidence < 0.25`), decays confidence rapidly and transitions to `UNKNOWN`
      so team possession is never fabricated during prolonged ball loss.
    - Explicit uncertainty: `LOOSE_BALL`, `CONTESTED`, `UNKNOWN`, and `OUT_OF_PLAY` are emitted
      explicitly rather than silently attributing ambiguous intervals to a team.
    - Camera cut protection: resets state and history on broadcast cuts.
    """

    def __init__(
        self,
        config: Optional[PossessionConfig] = None,
        max_ball_loss_hold_frames: Optional[int] = None,
        max_pass_flight_hold_frames: Optional[int] = None,
    ) -> None:
        self.config = config or PossessionConfig()
        self.max_ball_loss_hold_frames = (
            int(max_ball_loss_hold_frames)
            if max_ball_loss_hold_frames is not None
            else int(getattr(self.config, "max_ball_loss_hold_frames", 2))
        )
        self.max_pass_flight_hold_frames = (
            int(max_pass_flight_hold_frames)
            if max_pass_flight_hold_frames is not None
            else int(
                getattr(
                    self.config,
                    "max_pass_flight_hold_frames",
                    max(4, int(self.config.max_pass_retention_frames)),
                )
            )
        )
        self.reset()

    def reset(self) -> None:
        """Reset state machine to initial UNKNOWN state."""
        self.current_state: str = PossessionState.UNKNOWN.value
        self.controlling_team_id: Optional[str] = None
        self.controlling_player_id: Optional[int] = None
        self.current_confidence: float = 0.0
        self.candidate_state: Optional[str] = None
        self.candidate_streak: int = 0
        self.candidate_player_id: Optional[int] = None
        self.last_confirmed_team_state: Optional[str] = None
        self.last_confirmed_player_id: Optional[int] = None
        self.frames_since_team_control: int = 999
        self.uncertain_ball_streak: int = 0
        self.state_transitions: int = 0

    def step(
        self,
        raw_interaction: Dict[str, Any],
        camera_cut: bool = False,
    ) -> Tuple[str, Optional[str], Optional[int], float]:
        """
        Advance the possession state machine by one frame.

        Returns:
            `(possession_state, controlling_team_id, controlling_player_id, confidence)`
        """
        prev_state = self.current_state

        if camera_cut:
            self.reset()
            return (
                self.current_state,
                self.controlling_team_id,
                self.controlling_player_id,
                self.current_confidence,
            )

        raw_state = str(raw_interaction.get("raw_state", PossessionState.UNKNOWN.value))
        raw_pid = raw_interaction.get("controlling_player_id")
        raw_conf = float(raw_interaction.get("confidence", 0.0))
        ball_mode = raw_interaction.get("ball_observation_mode")
        ball_conf_raw = raw_interaction.get("ball_confidence")
        ball_conf = float(ball_conf_raw) if ball_conf_raw is not None else raw_conf

        team_states = (
            PossessionState.TEAM_A_CONTROL.value,
            PossessionState.TEAM_B_CONTROL.value,
        )

        # Conservative uncertain-ball decay: if ball is LOST or unreliable (< 0.25 while coasting),
        # do not sustain indefinite team possession.
        ball_unreliable = (
            ball_mode == "lost"
            or (ball_mode == "interpolated" and ball_conf < 0.25)
        )
        if ball_unreliable:
            self.uncertain_ball_streak += 1
        else:
            self.uncertain_ball_streak = 0

        if raw_state == PossessionState.OUT_OF_PLAY.value:
            self.current_state = PossessionState.OUT_OF_PLAY.value
            self.controlling_team_id = None
            self.controlling_player_id = None
            self.current_confidence = raw_conf
            self.candidate_state = None
            self.candidate_streak = 0
            self.last_confirmed_team_state = None
            self.last_confirmed_player_id = None
            self.frames_since_team_control = 999

        elif raw_state == PossessionState.CONTESTED.value:
            self.current_state = PossessionState.CONTESTED.value
            self.controlling_team_id = None
            self.controlling_player_id = None
            self.current_confidence = raw_conf
            self.candidate_state = None
            self.candidate_streak = 0
            self.frames_since_team_control += 1

        elif raw_state in team_states:
            if ball_unreliable and self.uncertain_ball_streak > max(1, self.max_ball_loss_hold_frames):
                # Do not fabricate team control during prolonged ball uncertainty/coasting
                self.current_state = PossessionState.UNKNOWN.value
                self.controlling_team_id = None
                self.controlling_player_id = None
                self.current_confidence = 0.0
                self.candidate_state = None
                self.candidate_streak = 0
                self.frames_since_team_control += 1
            else:
                target_team = (
                    InternalLabel.TEAM_A.value
                    if raw_state == PossessionState.TEAM_A_CONTROL.value
                    else InternalLabel.TEAM_B.value
                )

                # Case 1: Already in this team's control -> continue immediately
                if self.current_state == raw_state:
                    self.controlling_team_id = target_team
                    self.controlling_player_id = raw_pid
                    self.current_confidence = raw_conf
                    self.last_confirmed_team_state = raw_state
                    self.last_confirmed_player_id = raw_pid
                    self.frames_since_team_control = 0
                    self.candidate_state = raw_state
                    self.candidate_streak = self.config.min_control_frames

                # Case 2: Same team receiving a pass after LOOSE_BALL / short flight
                elif (
                    self.last_confirmed_team_state == raw_state
                    and self.frames_since_team_control <= max(
                        int(self.config.max_pass_retention_frames),
                        int(self.max_pass_flight_hold_frames),
                    )
                    and self.current_state in (PossessionState.LOOSE_BALL.value, PossessionState.UNKNOWN.value)
                ):
                    self.current_state = raw_state
                    self.controlling_team_id = target_team
                    self.controlling_player_id = raw_pid
                    self.current_confidence = raw_conf
                    self.last_confirmed_team_state = raw_state
                    self.last_confirmed_player_id = raw_pid
                    self.frames_since_team_control = 0
                    self.candidate_state = raw_state
                    self.candidate_streak = self.config.min_control_frames

                # Case 3: New team control candidate -> apply hysteresis (min_control_frames)
                else:
                    if self.candidate_state == raw_state:
                        self.candidate_streak += 1
                    else:
                        self.candidate_state = raw_state
                        self.candidate_streak = 1
                        self.candidate_player_id = raw_pid

                    if self.candidate_streak >= max(1, int(self.config.min_control_frames)):
                        self.current_state = raw_state
                        self.controlling_team_id = target_team
                        self.controlling_player_id = raw_pid
                        self.current_confidence = raw_conf
                        self.last_confirmed_team_state = raw_state
                        self.last_confirmed_player_id = raw_pid
                        self.frames_since_team_control = 0
                    else:
                        # Hysteresis hold: if previously controlled by opponent, mark CONTESTED during transition
                        if self.current_state in team_states and self.current_state != raw_state:
                            self.current_state = PossessionState.CONTESTED.value
                            self.controlling_team_id = None
                            self.controlling_player_id = None
                            self.current_confidence = round(raw_conf * 0.7, 4)
                        else:
                            self.current_state = PossessionState.LOOSE_BALL.value
                            self.controlling_team_id = None
                            self.controlling_player_id = None
                            self.current_confidence = round(raw_conf * 0.7, 4)
                        self.frames_since_team_control += 1

        elif raw_state == PossessionState.LOOSE_BALL.value:
            # Pass & flight continuity between same-team teammates:
            # When Team X has confirmed control and kicks/touches the ball, if a DIFFERENT
            # receiving player of Team X is already within 1.55 * control_radius and no opponent
            # is within 1.25 * control_radius, bridge TEAM_X_CONTROL smoothly across the reception
            # up to max_pass_flight_hold_frames while leaving longer open-turf flights as LOOSE_BALL.
            ctrl_r = float(self.config.control_distance_threshold)
            dist_a = raw_interaction.get("team_a_norm_dist")
            dist_b = raw_interaction.get("team_b_norm_dist")
            closest_tid = raw_interaction.get("closest_team_id")
            closest_d = raw_interaction.get("closest_norm_dist")
            bridged = False

            if (
                not ball_unreliable
                and self.last_confirmed_team_state in team_states
                and self.frames_since_team_control < self.max_pass_flight_hold_frames
                and closest_d is not None
            ):
                if self.last_confirmed_team_state == PossessionState.TEAM_A_CONTROL.value:
                    same_d = float(dist_a) if dist_a is not None else (
                        float(closest_d) if closest_tid == InternalLabel.TEAM_A.value else 999.0
                    )
                    opp_d = float(dist_b) if dist_b is not None else (
                        float(closest_d) if closest_tid == InternalLabel.TEAM_B.value else 999.0
                    )
                    same_pid = raw_interaction.get("team_a_player_id")
                    same_team = InternalLabel.TEAM_A.value
                else:
                    same_d = float(dist_b) if dist_b is not None else (
                        float(closest_d) if closest_tid == InternalLabel.TEAM_B.value else 999.0
                    )
                    opp_d = float(dist_a) if dist_a is not None else (
                        float(closest_d) if closest_tid == InternalLabel.TEAM_A.value else 999.0
                    )
                    same_pid = raw_interaction.get("team_b_player_id")
                    same_team = InternalLabel.TEAM_B.value

                is_new_teammate = (
                    same_pid is not None
                    and self.last_confirmed_player_id is not None
                    and int(same_pid) != int(self.last_confirmed_player_id)
                )
                if is_new_teammate and same_d <= 1.55 * ctrl_r and opp_d > 1.25 * ctrl_r and same_d < opp_d:
                    bridged = True
                    self.frames_since_team_control += 1
                    decay = 0.88 ** self.frames_since_team_control
                    self.current_state = self.last_confirmed_team_state
                    self.controlling_team_id = same_team
                    self.controlling_player_id = same_pid
                    self.current_confidence = round(max(0.28, raw_conf * decay), 4)

            if not bridged:
                self.current_state = PossessionState.LOOSE_BALL.value
                self.controlling_team_id = None
                self.controlling_player_id = None
                self.current_confidence = raw_conf
                self.candidate_state = None
                self.candidate_streak = 0
                self.frames_since_team_control += 1

        else:
            # UNKNOWN
            self.current_state = PossessionState.UNKNOWN.value
            self.controlling_team_id = None
            self.controlling_player_id = raw_pid
            self.current_confidence = 0.0
            self.candidate_state = None
            self.candidate_streak = 0
            self.frames_since_team_control += 1
            if self.uncertain_ball_streak > max(1, self.max_ball_loss_hold_frames):
                self.last_confirmed_team_state = None
                self.last_confirmed_player_id = None

        if self.current_state != prev_state:
            self.state_transitions += 1

        return (
            self.current_state,
            self.controlling_team_id,
            self.controlling_player_id,
            round(float(self.current_confidence), 4),
        )

