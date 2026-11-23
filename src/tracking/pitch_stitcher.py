"""Causal fixed-camera pitch-plane & appearance tracklet re-associator (`src/tracking/pitch_stitcher.py`).

Resolves short-to-medium player track fragmentations (`1..45` sampled frames) on wide-angle /
panoramic fixed-camera footage when Ultralytics `BYTETracker` or `BOTSORT` drops a track during
partial occlusion, tile-seam crossing, or far-side scale drop and spawns a new raw `track_id`
a few metres away.

Safety invariants:
1. Strictly causal (online frame-by-frame): only matches newly spawned raw track IDs on frame `f`
   against canonical identities that are **inactive** (absent from all currently active raw tracks
   in frame `f`) and were last observed within `max_lost_frames` sampled frames.
2. Never merges two simultaneously active raw tracks in the same frame into the same canonical ID.
3. Enforces simultaneous kinematic extrapolation gates (in metric pitch space when calibration is
   available and/or bbox-height-scaled pixel space), bounding-box height compatibility, and
   optional torso HSV+LAB cosine appearance gating (`<= max_appearance_cosine_dist`) so crossing
   opposing-team players are never merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.classification.cropper import extract_torso_crop
from src.classification.features import extract_hsv_lab_features
from src.config import CroppingConfig
from src.geometry.pitch_region import PitchRegionFilter
from src.schemas import DetectionRecord


@dataclass
class StitcherConfig:
    """Configuration for causal fixed-camera tracklet re-association."""

    enabled: bool = True
    max_lost_frames: int = 45  # in sampled frames (e.g. 45 sampled frames @ stride=2 -> 90 video frames / 3.6s)
    min_track_obs_for_velocity: int = 2
    velocity_ema_alpha: float = 0.45
    appearance_ema_alpha: float = 0.30
    # Spatial / kinematic gates in metric pitch space (metres)
    base_pitch_gate_m: float = 3.2
    pitch_expansion_per_lost_frame_m: float = 0.18
    max_pitch_gate_m: float = 7.2
    max_pitch_speed_m_per_sampled_frame: float = 0.85  # ~10.6 m/s at 12.5 Hz sampled rate
    # Spatial / kinematic gates in pixel space (scaled by player bbox height `bh`)
    base_pixel_gate_bh_mult: float = 1.65
    pixel_expansion_per_lost_frame_bh_mult: float = 0.11
    max_pixel_gate_bh_mult: float = 3.80
    max_pixel_speed_bh_per_sampled_frame: float = 0.55
    # Scale compatibility gate (`bh_new / bh_old`)
    min_height_ratio: float = 0.68
    max_height_ratio: float = 1.47
    # Appearance cosine distance gate (`1 - cos_sim(e_new, e_old)`) when embeddings exist for both
    max_appearance_cosine_dist: float = 0.26
    # Damping factor applied to extrapolated velocity per lost sampled frame
    velocity_decay_per_lost_frame: float = 0.94


@dataclass
class _CanonicalTrackState:
    """Internal state for a canonical player track."""

    canonical_id: int
    last_raw_id: int
    last_frame_number: int
    last_sampled_step: int
    last_bbox: Tuple[int, int, int, int]
    foot_px: Tuple[float, float]
    vel_px: Tuple[float, float] = (0.0, 0.0)  # per sampled frame step
    pitch_pos_m: Optional[Tuple[float, float]] = None
    vel_pitch_m: Tuple[float, float] = (0.0, 0.0)  # per sampled frame step
    smooth_bh: float = 60.0
    smooth_bw: float = 24.0
    appearance_embedding: Optional[np.ndarray] = None
    total_observations: int = 1
    stitch_count: int = 0


class FixedCameraTrackletAssociator:
    """Causal (online frame-by-frame) pitch-plane and appearance tracklet stitcher.

    Wraps raw `DetectionRecord` outputs from Ultralytics `BYTETracker` / `BOTSORT` and
    maintains persistent canonical `tracking_id` assignments across short-to-medium
    detection dropouts without ever merging simultaneously active tracks.
    """

    def __init__(
        self,
        config: Optional[StitcherConfig] = None,
        pitch_region_filter: Optional[PitchRegionFilter] = None,
        pitch_projector_fn: Optional[Callable[[float, float], Optional[Tuple[float, float]]]] = None,
        cropping_config: Optional[CroppingConfig] = None,
    ) -> None:
        self.config = config or StitcherConfig()
        self.pitch_region_filter = pitch_region_filter
        self._pitch_projector_fn = pitch_projector_fn
        self.cropping_config = cropping_config or CroppingConfig(
            min_player_height=20,
            min_player_width=8,
            min_crop_width=6,
            min_crop_height=8,
            min_blur_variance=3.0,
        )

        self.canonical_tracks: Dict[int, _CanonicalTrackState] = {}
        self.raw_to_canonical: Dict[int, int] = {}
        self.seen_raw_ids: Set[int] = set()
        self.sampled_step: int = 0
        self.total_stitches: int = 0
        self.stitch_events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset all tracklet state (e.g., on camera cut or new video)."""
        self.canonical_tracks.clear()
        self.raw_to_canonical.clear()
        self.seen_raw_ids.clear()
        self.sampled_step = 0
        self.total_stitches = 0
        self.stitch_events.clear()

    def _project_foot_to_pitch(self, fx: float, fy: float) -> Optional[Tuple[float, float]]:
        if self._pitch_projector_fn is not None:
            return self._pitch_projector_fn(fx, fy)
        if self.pitch_region_filter is not None:
            return self.pitch_region_filter.project_pixel_to_pitch_m(fx, fy)
        return None

    def _extract_appearance_embedding(
        self,
        frame_bgr: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        all_bboxes: Sequence[Tuple[int, int, int, int]],
    ) -> Optional[np.ndarray]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        crop, _, qm = extract_torso_crop(
            frame_bgr,
            bbox,
            config=self.cropping_config,
            other_player_bboxes=all_bboxes,
        )
        if crop is None or not qm.is_valid:
            return None
        try:
            vec = extract_hsv_lab_features(crop)
            norm = float(np.linalg.norm(vec))
            return (vec / norm).astype(np.float32) if norm > 1e-8 else None
        except ValueError:
            return None

    @staticmethod
    def _foot_and_dims(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        bw = max(2.0, float(x2 - x1))
        bh = max(4.0, float(y2 - y1))
        fx = float(x1 + x2) / 2.0
        fy = float(y2)
        return fx, fy, bw, bh

    def _update_canonical_state(
        self,
        state: _CanonicalTrackState,
        raw_id: int,
        frame_number: int,
        sampled_step: int,
        bbox: Tuple[int, int, int, int],
        pitch_pos: Optional[Tuple[float, float]],
        embedding: Optional[np.ndarray],
    ) -> None:
        fx, fy, bw, bh = self._foot_and_dims(bbox)
        dt_steps = max(1, sampled_step - state.last_sampled_step)

        if state.total_observations >= 1 and dt_steps <= 12:
            inst_vx = (fx - state.foot_px[0]) / float(dt_steps)
            inst_vy = (fy - state.foot_px[1]) / float(dt_steps)
            max_px_spd = self.config.max_pixel_speed_bh_per_sampled_frame * max(bh, state.smooth_bh)
            spd_px = math.hypot(inst_vx, inst_vy)
            if spd_px > max_px_spd and spd_px > 1e-6:
                scale = max_px_spd / spd_px
                inst_vx *= scale
                inst_vy *= scale
            alpha_v = self.config.velocity_ema_alpha if state.total_observations >= 2 else 0.70
            state.vel_px = (
                (1.0 - alpha_v) * state.vel_px[0] + alpha_v * inst_vx,
                (1.0 - alpha_v) * state.vel_px[1] + alpha_v * inst_vy,
            )
            if pitch_pos is not None and state.pitch_pos_m is not None:
                inst_vpx = (pitch_pos[0] - state.pitch_pos_m[0]) / float(dt_steps)
                inst_vpy = (pitch_pos[1] - state.pitch_pos_m[1]) / float(dt_steps)
                spd_m = math.hypot(inst_vpx, inst_vpy)
                max_m_spd = self.config.max_pitch_speed_m_per_sampled_frame
                if spd_m > max_m_spd and spd_m > 1e-6:
                    scale_m = max_m_spd / spd_m
                    inst_vpx *= scale_m
                    inst_vpy *= scale_m
                state.vel_pitch_m = (
                    (1.0 - alpha_v) * state.vel_pitch_m[0] + alpha_v * inst_vpx,
                    (1.0 - alpha_v) * state.vel_pitch_m[1] + alpha_v * inst_vpy,
                )

        state.last_raw_id = raw_id
        state.last_frame_number = frame_number
        state.last_sampled_step = sampled_step
        state.last_bbox = bbox
        state.foot_px = (fx, fy)
        if pitch_pos is not None:
            state.pitch_pos_m = pitch_pos
        state.smooth_bh = 0.70 * state.smooth_bh + 0.30 * bh
        state.smooth_bw = 0.70 * state.smooth_bw + 0.30 * bw
        state.total_observations += 1

        if embedding is not None:
            if state.appearance_embedding is None:
                state.appearance_embedding = embedding.copy()
            else:
                a_emb = self.config.appearance_ema_alpha
                blended = (1.0 - a_emb) * state.appearance_embedding + a_emb * embedding
                norm = float(np.linalg.norm(blended))
                if norm > 1e-8:
                    state.appearance_embedding = (blended / norm).astype(np.float32)

    def _compute_stitch_cost(
        self,
        lost_state: _CanonicalTrackState,
        new_bbox: Tuple[int, int, int, int],
        new_pitch_pos: Optional[Tuple[float, float]],
        new_embedding: Optional[np.ndarray],
        sampled_step: int,
    ) -> Optional[float]:
        """Return normalized association cost in `[0, 1)` if candidate satisfies all gates, else `None`."""
        gap_steps = sampled_step - lost_state.last_sampled_step
        if gap_steps < 1 or gap_steps > self.config.max_lost_frames:
            return None

        fx, fy, bw, bh = self._foot_and_dims(new_bbox)
        ref_bh = max(12.0, 0.5 * (bh + lost_state.smooth_bh))

        # 1. Bounding-box height compatibility gate
        h_ratio = bh / max(4.0, lost_state.smooth_bh)
        if h_ratio < self.config.min_height_ratio or h_ratio > self.config.max_height_ratio:
            return None

        # Effective velocity displacement factor with geometric decay over gap_steps
        decay = self.config.velocity_decay_per_lost_frame
        if abs(1.0 - decay) > 1e-4:
            eff_steps = (1.0 - (decay ** gap_steps)) / (1.0 - decay)
        else:
            eff_steps = float(gap_steps)
        use_vel = lost_state.total_observations >= self.config.min_track_obs_for_velocity

        # 2. Pixel-space kinematic gate (always checked to prevent cross-image jumps even if projection has distortion)
        pred_fx = lost_state.foot_px[0] + (lost_state.vel_px[0] * eff_steps if use_vel else 0.0)
        pred_fy = lost_state.foot_px[1] + (lost_state.vel_px[1] * eff_steps if use_vel else 0.0)
        # Also compute un-extrapolated distance in case player stopped during occlusion
        dist_px_extrap = math.hypot(fx - pred_fx, fy - pred_fy)
        dist_px_static = math.hypot(fx - lost_state.foot_px[0], fy - lost_state.foot_px[1])
        dist_px = min(dist_px_extrap, 0.85 * dist_px_extrap + 0.15 * dist_px_static, dist_px_static * 1.05)

        px_gate_mult = min(
            self.config.max_pixel_gate_bh_mult,
            self.config.base_pixel_gate_bh_mult + self.config.pixel_expansion_per_lost_frame_bh_mult * gap_steps,
        )
        max_px_dist = px_gate_mult * ref_bh
        if dist_px > max_px_dist:
            return None
        norm_px_cost = dist_px / max(1.0, max_px_dist)

        # 3. Metric pitch-space kinematic gate (authoritative when available)
        norm_pitch_cost = norm_px_cost
        if new_pitch_pos is not None and lost_state.pitch_pos_m is not None:
            pred_px_m = lost_state.pitch_pos_m[0] + (lost_state.vel_pitch_m[0] * eff_steps if use_vel else 0.0)
            pred_py_m = lost_state.pitch_pos_m[1] + (lost_state.vel_pitch_m[1] * eff_steps if use_vel else 0.0)
            dist_m_extrap = math.hypot(new_pitch_pos[0] - pred_px_m, new_pitch_pos[1] - pred_py_m)
            dist_m_static = math.hypot(
                new_pitch_pos[0] - lost_state.pitch_pos_m[0],
                new_pitch_pos[1] - lost_state.pitch_pos_m[1],
            )
            dist_m = min(dist_m_extrap, dist_m_static * 1.05)
            pitch_gate_m = min(
                self.config.max_pitch_gate_m,
                self.config.base_pitch_gate_m + self.config.pitch_expansion_per_lost_frame_m * gap_steps,
            )
            if dist_m > pitch_gate_m:
                return None
            norm_pitch_cost = dist_m / max(0.1, pitch_gate_m)

        # 4. Appearance cosine distance gate when embeddings exist for both
        app_cost = 0.0
        if new_embedding is not None and lost_state.appearance_embedding is not None:
            cos_sim = float(np.dot(new_embedding, lost_state.appearance_embedding))
            cos_dist = max(0.0, 1.0 - cos_sim)
            if cos_dist > self.config.max_appearance_cosine_dist:
                return None
            app_cost = cos_dist / max(1e-4, self.config.max_appearance_cosine_dist)

        scale_cost = abs(math.log(max(0.1, h_ratio))) / max(1e-4, abs(math.log(self.config.max_height_ratio)))
        gap_cost = float(gap_steps) / float(max(1, self.config.max_lost_frames))
        # Reward longer-established canonical tracks slightly to prioritize primary identity continuity
        maturity_bonus = min(0.08, 0.01 * min(8, lost_state.total_observations))

        total_cost = (
            0.42 * norm_pitch_cost
            + 0.25 * norm_px_cost
            + 0.18 * app_cost
            + 0.08 * scale_cost
            + 0.07 * gap_cost
            - maturity_bonus
        )
        return max(0.001, float(total_cost))

    def update(
        self,
        frame_number: int,
        player_records: Sequence[DetectionRecord],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[DetectionRecord]:
        """Remap player `tracking_id` values in-place / copy while enforcing strict 1-to-1 per-frame uniqueness."""
        self.sampled_step += 1
        if not self.config.enabled or not player_records:
            return list(player_records)

        all_bboxes = [r.bbox for r in player_records]

        # Separate continuing raw tracks vs newly spawned raw tracks on this frame
        continuing_indices: List[int] = []
        new_indices: List[int] = []
        active_canonical_ids: Set[int] = set()

        for idx, rec in enumerate(player_records):
            if rec.tracking_id is None:
                continue
            raw_id = int(rec.tracking_id)
            if raw_id in self.raw_to_canonical:
                can_id = self.raw_to_canonical[raw_id]
                # Safety invariant: never allow two raw tracks in the same frame to share a canonical ID
                if can_id not in active_canonical_ids:
                    continuing_indices.append(idx)
                    active_canonical_ids.add(can_id)
                else:
                    # Unexpected duplicate raw mapping: treat as new track
                    new_indices.append(idx)
            else:
                new_indices.append(idx)

        # 1. Update all continuing canonical tracks first so their state is current and they are excluded from lost pool
        for idx in continuing_indices:
            rec = player_records[idx]
            raw_id = int(rec.tracking_id)  # type: ignore[arg-type]
            can_id = self.raw_to_canonical[raw_id]
            fx, fy, _, _ = self._foot_and_dims(rec.bbox)
            pitch_pos = self._project_foot_to_pitch(fx, fy)
            emb = self._extract_appearance_embedding(frame_bgr, rec.bbox, all_bboxes)
            state = self.canonical_tracks.get(can_id)
            if state is not None:
                self._update_canonical_state(
                    state=state,
                    raw_id=raw_id,
                    frame_number=frame_number,
                    sampled_step=self.sampled_step,
                    bbox=rec.bbox,
                    pitch_pos=pitch_pos,
                    embedding=emb,
                )

        # 2. Match newly spawned raw tracks against currently INACTIVE recently lost canonical tracks
        if new_indices:
            inactive_candidates: List[_CanonicalTrackState] = [
                st
                for can_id, st in self.canonical_tracks.items()
                if can_id not in active_canonical_ids
                and 1 <= (self.sampled_step - st.last_sampled_step) <= self.config.max_lost_frames
            ]

            new_Infos: List[Tuple[int, int, Tuple[int, int, int, int], Optional[Tuple[float, float]], Optional[np.ndarray]]] = []
            for idx in new_indices:
                rec = player_records[idx]
                raw_id = int(rec.tracking_id)  # type: ignore[arg-type]
                fx, fy, _, _ = self._foot_and_dims(rec.bbox)
                pitch_pos = self._project_foot_to_pitch(fx, fy)
                emb = self._extract_appearance_embedding(frame_bgr, rec.bbox, all_bboxes)
                new_Infos.append((idx, raw_id, rec.bbox, pitch_pos, emb))

            matched_new_to_can: Dict[int, int] = {}
            if inactive_candidates:
                n_new = len(new_Infos)
                n_cand = len(inactive_candidates)
                cost_mat = np.full((n_new, n_cand), 1e6, dtype=np.float64)
                for i, (_, _, bbox, p_pos, emb) in enumerate(new_Infos):
                    for j, cand_st in enumerate(inactive_candidates):
                        c = self._compute_stitch_cost(
                            lost_state=cand_st,
                            new_bbox=bbox,
                            new_pitch_pos=p_pos,
                            new_embedding=emb,
                            sampled_step=self.sampled_step,
                        )
                        if c is not None:
                            cost_mat[i, j] = c

                r_ind, c_ind = linear_sum_assignment(cost_mat)
                for r, c in zip(r_ind, c_ind):
                    if cost_mat[r, c] < 1e5:
                        cand_st = inactive_candidates[c]
                        if cand_st.canonical_id not in active_canonical_ids:
                            matched_new_to_can[r] = cand_st.canonical_id
                            active_canonical_ids.add(cand_st.canonical_id)

            # Apply stitched canonical IDs or register genuinely new canonical tracks
            for i, (rec_idx, raw_id, bbox, p_pos, emb) in enumerate(new_Infos):
                self.seen_raw_ids.add(raw_id)
                if i in matched_new_to_can:
                    can_id = matched_new_to_can[i]
                    st = self.canonical_tracks[can_id]
                    gap_steps = self.sampled_step - st.last_sampled_step
                    self.raw_to_canonical[raw_id] = can_id
                    self._update_canonical_state(
                        state=st,
                        raw_id=raw_id,
                        frame_number=frame_number,
                        sampled_step=self.sampled_step,
                        bbox=bbox,
                        pitch_pos=p_pos,
                        embedding=emb,
                    )
                    st.stitch_count += 1
                    self.total_stitches += 1
                    self.stitch_events.append(
                        {
                            "frame_number": frame_number,
                            "sampled_step": self.sampled_step,
                            "new_raw_id": raw_id,
                            "canonical_id": can_id,
                            "gap_sampled_frames": gap_steps,
                        }
                    )
                else:
                    # Ensure newly assigned canonical ID does not collide with any existing canonical ID
                    can_id = raw_id
                    if can_id in self.canonical_tracks or can_id in active_canonical_ids:
                        can_id = max(list(self.canonical_tracks.keys()) + [raw_id]) + 1
                    self.raw_to_canonical[raw_id] = can_id
                    active_canonical_ids.add(can_id)
                    fx, fy, bw, bh = self._foot_and_dims(bbox)
                    self.canonical_tracks[can_id] = _CanonicalTrackState(
                        canonical_id=can_id,
                        last_raw_id=raw_id,
                        last_frame_number=frame_number,
                        last_sampled_step=self.sampled_step,
                        last_bbox=bbox,
                        foot_px=(fx, fy),
                        vel_px=(0.0, 0.0),
                        pitch_pos_m=p_pos,
                        vel_pitch_m=(0.0, 0.0),
                        smooth_bh=bh,
                        smooth_bw=bw,
                        appearance_embedding=emb.copy() if emb is not None else None,
                        total_observations=1,
                        stitch_count=0,
                    )

        # 3. Build output records and verify strict per-frame uniqueness invariant
        out_records: List[DetectionRecord] = []
        emitted_ids: Set[int] = set()
        for rec in player_records:
            if rec.tracking_id is None:
                out_records.append(rec)
                continue
            raw_id = int(rec.tracking_id)
            can_id = self.raw_to_canonical.get(raw_id, raw_id)
            if can_id in emitted_ids:
                # Defensive fallback: never emit duplicate tracking_id in the same frame
                can_id = max(emitted_ids.union(self.canonical_tracks.keys())) + 1
                self.raw_to_canonical[raw_id] = can_id
            emitted_ids.add(can_id)
            out_records.append(
                DetectionRecord(
                    frame_number=rec.frame_number,
                    timestamp=rec.timestamp,
                    class_id=rec.class_id,
                    class_name=rec.class_name,
                    bbox=rec.bbox,
                    confidence=rec.confidence,
                    tracking_id=can_id,
                )
            )
        return out_records

    def summarize(self) -> Dict[str, Any]:
        """Return diagnostic summary of tracklet stitching."""
        lifetimes = [st.total_observations for st in self.canonical_tracks.values()]
        return {
            "enabled": bool(self.config.enabled),
            "max_lost_frames": int(self.config.max_lost_frames),
            "raw_unique_ids_seen": len(self.seen_raw_ids),
            "canonical_unique_ids": len(self.canonical_tracks),
            "total_stitches": int(self.total_stitches),
            "mean_canonical_track_lifetime_frames": round(float(np.mean(lifetimes)), 2) if lifetimes else 0.0,
            "median_canonical_track_lifetime_frames": round(float(np.median(lifetimes)), 2) if lifetimes else 0.0,
        }
