"""Temporal team-label stabilization with confidence weighting, soft locking, and identity-reset protection."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple
import math

from src.config import TrackingConfig
from src.schemas import ClassificationPrediction, InternalLabel


@dataclass
class VoteEntry:
    """Single historical team observation for a track."""

    frame_number: int
    label: str
    weight: float


@dataclass
class TrackTeamState:
    """Per-track temporal classification and diagnostic state."""

    track_id: int
    first_frame: int
    last_frame: int
    last_bbox: Tuple[int, int, int, int]
    observations: int = 0
    known_observations: int = 0
    unknown_observations: int = 0
    raw_changes: int = 0
    stable_changes: int = 0
    identity_resets: int = 0
    last_raw_known_label: Optional[str] = None
    last_stable_known_label: Optional[str] = None
    current_stable_label: str = InternalLabel.UNKNOWN.value
    track_confidence: float = 0.0
    locked_label: Optional[str] = None
    contradiction_streak: int = 0
    contradiction_candidate: Optional[str] = None
    history: Deque[VoteEntry] = field(default_factory=deque)


class TemporalTeamStabilizer:
    """
    Stabilizes per-frame player team classifications across time.

    Key engineering improvements over unweighted 7-frame majority voting:
    1. Confidence-weighted voting over a sliding window (`history_length`).
    2. Minimum evidence requirements (`min_votes`, `min_agreement`) before emitting a known team.
    3. Explicit Unknown handling: transient `Unknown` frames (e.g. occlusion or motion blur) do not
       erase established team state.
    4. Soft track-level label locking once `lock_min_observations` and `lock_confidence_threshold`
       are met, preventing flicker during brief pose changes.
    5. Contradiction unlock: if `reclassify_contradiction_streak` consecutive confident observations
       support a different class, the lock is released and updated (avoiding permanent bad locks).
    6. Identity contamination guard: if a track ID reappears after `max_missing_frames_before_reset`
       or jumps spatially by an implausible distance relative to player height, its classification
       history is reset.
    """

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self.tracks: Dict[int, TrackTeamState] = {}

    def _should_reset_identity(
        self,
        state: TrackTeamState,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        frame_gap = frame_number - state.last_frame
        if frame_gap > self.config.max_missing_frames_before_reset:
            return True

        # Check spatial jump relative to player height per elapsed frame
        px1, py1, px2, py2 = state.last_bbox
        cx1, cy1, cx2, cy2 = bbox
        prev_cx, prev_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        curr_cx, curr_cy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
        dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

        ref_height = max(25.0, float((py2 - py1 + cy2 - cy1) / 2.0))
        allowed_jump = self.config.max_spatial_jump_ratio * ref_height * max(1, min(frame_gap, 5))
        return dist > allowed_jump

    def _reset_track_history(
        self,
        state: TrackTeamState,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        state.history.clear()
        state.locked_label = None
        state.contradiction_streak = 0
        state.contradiction_candidate = None
        state.current_stable_label = InternalLabel.UNKNOWN.value
        state.track_confidence = 0.0
        state.identity_resets += 1
        state.last_frame = frame_number
        state.last_bbox = bbox

    def update(
        self,
        track_id: Optional[int],
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        prediction: ClassificationPrediction,
    ) -> Tuple[str, float, bool]:
        """
        Update tracking state with a new frame observation and return:
        `(stable_internal_label, track_team_confidence, is_locked)`
        """
        raw_label = prediction.predicted_label
        raw_weight = (
            max(0.05, float(prediction.confidence_score))
            if self.config.confidence_weighted
            else 1.0
        )

        # Untracked detection fallback
        if track_id is None:
            return raw_label, float(prediction.confidence_score), False

        tid = int(track_id)
        if tid not in self.tracks:
            self.tracks[tid] = TrackTeamState(
                track_id=tid,
                first_frame=frame_number,
                last_frame=frame_number,
                last_bbox=bbox,
                history=deque(maxlen=self.config.history_length),
            )
        state = self.tracks[tid]

        if state.observations > 0 and self._should_reset_identity(state, frame_number, bbox):
            self._reset_track_history(state, frame_number, bbox)

        state.observations += 1
        state.last_frame = frame_number
        state.last_bbox = bbox

        # Track raw known label transitions
        if raw_label != InternalLabel.UNKNOWN.value:
            if state.last_raw_known_label is not None and state.last_raw_known_label != raw_label:
                state.raw_changes += 1
            state.last_raw_known_label = raw_label
            state.history.append(VoteEntry(frame_number=frame_number, label=raw_label, weight=raw_weight))

            # Check for strong contradiction against a locked label
            if state.locked_label is not None:
                if raw_label != state.locked_label and prediction.confidence_score >= 0.45:
                    if state.contradiction_candidate == raw_label:
                        state.contradiction_streak += 1
                    else:
                        state.contradiction_candidate = raw_label
                        state.contradiction_streak = 1

                    if state.contradiction_streak >= self.config.reclassify_contradiction_streak:
                        # Unlock and keep only recent contradictory evidence
                        state.locked_label = None
                        state.contradiction_streak = 0
                        state.contradiction_candidate = None
                else:
                    state.contradiction_streak = 0
                    state.contradiction_candidate = None

        # Compute weighted vote across history
        if state.locked_label is not None:
            stable_label = state.locked_label
            # Compute current confidence from history
            total_w = sum(v.weight for v in state.history) or 1.0
            locked_w = sum(v.weight for v in state.history if v.label == state.locked_label)
            state.track_confidence = round(max(0.65, locked_w / total_w), 4)
        else:
            if len(state.history) < self.config.min_votes:
                stable_label = InternalLabel.UNKNOWN.value
                state.track_confidence = 0.0
            else:
                class_weights: Dict[str, float] = {}
                class_counts: Counter[str] = Counter()
                total_w = 0.0
                for entry in state.history:
                    class_weights[entry.label] = class_weights.get(entry.label, 0.0) + entry.weight
                    class_counts[entry.label] += 1
                    total_w += entry.weight

                best_label, best_w = max(class_weights.items(), key=lambda kv: kv[1])
                agreement = best_w / max(1e-6, total_w)

                if class_counts[best_label] < self.config.min_votes or agreement < self.config.min_agreement:
                    stable_label = InternalLabel.UNKNOWN.value
                    state.track_confidence = round(float(agreement), 4)
                else:
                    stable_label = best_label
                    state.track_confidence = round(float(agreement), 4)

                    # Check if eligible for soft label lock
                    if (
                         len(state.history) >= self.config.lock_min_observations
                        and agreement >= self.config.lock_confidence_threshold
                    ):
                        state.locked_label = best_label

        if stable_label == InternalLabel.UNKNOWN.value:
            state.unknown_observations += 1
        else:
            state.known_observations += 1
            if state.last_stable_known_label is not None and state.last_stable_known_label != stable_label:
                state.stable_changes += 1
            state.last_stable_known_label = stable_label

        state.current_stable_label = stable_label
        return stable_label, state.track_confidence, (state.locked_label is not None)

    def summarize_diagnostics(self, processed_frames: int) -> Dict[str, Any]:
        """Compute comprehensive tracking and team-stability summary statistics."""
        total_obs = sum(s.observations for s in self.tracks.values())
        known_obs = sum(s.known_observations for s in self.tracks.values())
        unk_obs = sum(s.unknown_observations for s in self.tracks.values())
        raw_changes = sum(s.raw_changes for s in self.tracks.values())
        stable_changes = sum(s.stable_changes for s in self.tracks.values())
        identity_resets = sum(s.identity_resets for s in self.tracks.values())
        short_tracks = sum(1 for s in self.tracks.values() if s.observations < 5)
        locked_tracks = sum(1 for s in self.tracks.values() if s.locked_label is not None)

        lifetimes = [s.observations for s in self.tracks.values()]
        mean_lifetime = float(sum(lifetimes)) / len(lifetimes) if lifetimes else 0.0
        max_lifetime = max(lifetimes) if lifetimes else 0

        coverage = float(known_obs) / float(total_obs) if total_obs > 0 else 0.0
        unknown_rate = float(unk_obs) / float(total_obs) if total_obs > 0 else 0.0

        return {
            "processed_frames": int(processed_frames),
            "distinct_tracking_ids": len(self.tracks),
            "short_tracks_under_5_obs": int(short_tracks),
            "short_track_fraction": round(short_tracks / float(max(1, len(self.tracks))), 4),
            "mean_track_observations": round(mean_lifetime, 2),
            "max_track_observations": int(max_lifetime),
            "total_player_observations": int(total_obs),
            "known_team_observations": int(known_obs),
            "unknown_team_observations": int(unk_obs),
            "stable_label_coverage": round(coverage, 4),
            "unknown_rate": round(unknown_rate, 4),
            "raw_team_label_changes": int(raw_changes),
            "smoothed_team_label_changes": int(stable_changes),
            "label_change_reduction_ratio": round(
                1.0 - (stable_changes / float(raw_changes)), 4
            )
            if raw_changes > 0
            else 1.0,
            "locked_tracks_count": int(locked_tracks),
            "identity_resets_triggered": int(identity_resets),
            "note": (
                "Descriptive tracking stability diagnostics. Does not represent formal ground-truth "
                "identity accuracy (IDF1/HOTA) unless evaluated against manual identity annotations."
            ),
        }

