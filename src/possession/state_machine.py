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
    - Same-team pass continuity (`max_pass_retention_frames`): when Team A kicks the ball into flight
      (`LOOSE_BALL`) and another Team A player receives it within `max_pass_retention_frames`, Team A
      re-establishes control immediately without hysteresis delay.
    - Explicit uncertainty: `LOOSE_BALL`, `CONTESTED`, `UNKNOWN`, and `OUT_OF_PLAY` are emitted
      explicitly rather than silently attributing ambiguous intervals to a team.
    - Camera cut protection: resets state and history on broadcast cuts.
    """

    def __init__(self, config: Optional[PossessionConfig] = None) -> None:
        self.config = config or PossessionConfig()
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
        self.frames_since_team_control: int = 999
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

        team_states = (
            PossessionState.TEAM_A_CONTROL.value,
            PossessionState.TEAM_B_CONTROL.value,
        )

        if raw_state == PossessionState.OUT_OF_PLAY.value:
            self.current_state = PossessionState.OUT_OF_PLAY.value
            self.controlling_team_id = None
            self.controlling_player_id = None
            self.current_confidence = raw_conf
            self.candidate_state = None
            self.candidate_streak = 0
            self.last_confirmed_team_state = None
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
                self.frames_since_team_control = 0
                self.candidate_state = raw_state
                self.candidate_streak = self.config.min_control_frames

            # Case 2: Same team receiving a short pass after LOOSE_BALL flight
            elif (
                self.last_confirmed_team_state == raw_state
                and self.frames_since_team_control <= self.config.max_pass_retention_frames
                and self.current_state in (PossessionState.LOOSE_BALL.value, PossessionState.UNKNOWN.value)
            ):
                self.current_state = raw_state
                self.controlling_team_id = target_team
                self.controlling_player_id = raw_pid
                self.current_confidence = raw_conf
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

        if self.current_state != prev_state:
            self.state_transitions += 1

        return (
            self.current_state,
            self.controlling_team_id,
            self.controlling_player_id,
            round(float(self.current_confidence), 4),
        )

