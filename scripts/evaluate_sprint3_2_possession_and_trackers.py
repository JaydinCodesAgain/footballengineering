"""Sprint 3.2 Phase E & F Evaluation Utilities:
1. Head-to-head ByteTrack vs BoT-SORT tracking benchmark on IDENTICAL fixed-camera detections
   (measuring GS-HOTA, DetA, AssA, LocA, MOTA, IDF1/IDP/IDR, IDSW, Fragmentations, MT/PT/ML,
   and explicitly decomposing detector-recall vs identity-association gains).
2. Possession & ball-trajectory post-hoc verification against:
   - Verified ball bounding boxes overlapping the evaluated segment,
   - Independent BAS ball-action team anchors (`left`/`right` aligned post-hoc via discovered teams),
   - Explicit Unknown preservation & persistent static-distractor rejection counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scripts.evaluate_soccertrack_v2 import (
    evaluate_pitch_space_gsr,
    load_ball_trajectory,
    load_fisheye_pitch_projector,
    load_gsr_frames_streaming,
    load_pipeline_observations,
)
from src.config import DetectionConfig
from src.detection.detector import FootballDetector
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter
from src.schemas import DetectionRecord, FrameObservationRecord, InternalLabel
from src.tracking.tracker import _NumpyDetections, _build_ultralytics_tracker


def evaluate_possession_and_ball_vs_bas_and_verified(
    run_dir: Path,
    eval_json_path: Path,
    bas_json_path: Optional[Path],
    verified_ball_json_path: Optional[Path],
    match_id: str,
) -> Dict[str, Any]:
    """Audit ball trajectory correctness and BAS-event possession agreement for a finished run."""
    eval_data = json.loads(eval_json_path.read_text(encoding="utf-8"))
    ball_recs = load_ball_trajectory(run_dir / "ball_tracking_diagnostics.csv")
    ball_by_f = {int(r.frame_number): r for r in ball_recs}

    poss_by_f: Dict[int, Dict[str, str]] = {}
    poss_csv = run_dir / "possession_timeline.csv"
    if poss_csv.is_file():
        with poss_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                poss_by_f[int(row["frame_number"])] = row

    # Determine post-hoc team alignment (left -> Team_A or Team_B) from GSR evaluation
    pitch_ev = eval_data.get("pitch_space_gsr_evaluation_5m") or {}
    td_ev = (pitch_ev.get("team_discovery") or {}).get("outfield_frame_observations_eval") or {}
    use_swap = bool(td_ev.get("permutation_applied", False))
    # In evaluate_pitch_space_gsr: left -> Team_A, right -> Team_B; if permutation_applied=True then pred Team_A maps to right & Team_B maps to left
    left_pred_label = InternalLabel.TEAM_B.value if use_swap else InternalLabel.TEAM_A.value
    right_pred_label = InternalLabel.TEAM_A.value if use_swap else InternalLabel.TEAM_B.value

    # 1. Ball trajectory summary + source breakdown
    obs_recs = [r for r in ball_recs if r.is_observed]
    interp_recs = [r for r in ball_recs if r.is_interpolated]
    lost_recs = [r for r in ball_recs if not r.is_observed and not r.is_interpolated]
    src_counts = dict(Counter(r.detection_source for r in obs_recs))

    # 2. Check ball trajectory against any verified ball boxes inside the run's evaluated frame window
    verified_ball_check: Dict[str, Any] = {"available": False}
    if verified_ball_json_path and verified_ball_json_path.is_file():
        vb = json.loads(verified_ball_json_path.read_text(encoding="utf-8"))
        m_entry = (vb.get("matches") or {}).get(match_id)
        if m_entry:
            frames_list = m_entry.get("frames") or m_entry.get("per_frame") or []
            tp_obs = fp_obs = fn_obs = tp_active = fp_active = fn_active = 0
            neg_fp_obs = neg_fp_active = neg_n = vis_n = 0
            errs_obs: List[float] = []
            errs_active: List[float] = []
            details = []
            for item in frames_list:
                fno = int(item["frame_number"])
                if fno not in ball_by_f:
                    continue
                br = ball_by_f[fno]
                gt_box = item.get("bbox")
                if gt_box is None:
                    neg_n += 1
                    if br.is_observed:
                        fp_obs += 1
                        neg_fp_obs += 1
                    if br.is_observed or br.is_interpolated:
                        fp_active += 1
                        neg_fp_active += 1
                    details.append({"frame": fno, "gt_visible": False, "mode": br.observation_mode, "source": br.detection_source})
                    continue

                vis_n += 1
                gcx, gcy = (gt_box[0] + gt_box[2]) / 2.0, (gt_box[1] + gt_box[3]) / 2.0
                err = (
                    math.hypot(br.center_x - gcx, br.center_y - gcy)
                    if (br.center_x is not None and br.center_y is not None)
                    else None
                )
                hit_obs = br.is_observed and err is not None and err <= 30.0
                hit_act = (br.is_observed or br.is_interpolated) and err is not None and err <= 35.0
                if hit_obs:
                    tp_obs += 1
                    errs_obs.append(err)
                elif br.is_observed:
                    fp_obs += 1
                    fn_obs += 1
                else:
                    fn_obs += 1

                if hit_act:
                    tp_active += 1
                    errs_active.append(err)
                elif br.is_observed or br.is_interpolated:
                    fp_active += 1
                    fn_active += 1
                else:
                    fn_active += 1

                details.append({
                    "frame": fno,
                    "gt_visible": True,
                    "mode": br.observation_mode,
                    "source": br.detection_source,
                    "center_err_px": round(err, 1) if err is not None else None,
                    "hit_obs_30px": hit_obs,
                    "hit_active_35px": hit_act,
                })

            verified_ball_check = {
                "available": True,
                "overlapping_visible_frames": vis_n,
                "overlapping_absent_frames": neg_n,
                "observed_only": {
                    "tp": tp_obs, "fp": fp_obs, "fn": fn_obs, "neg_fp": neg_fp_obs,
                    "precision": round(tp_obs / float(tp_obs + fp_obs), 4) if (tp_obs + fp_obs) > 0 else None,
                    "recall": round(tp_obs / float(max(1, vis_n)), 4) if vis_n > 0 else None,
                    "median_err_px": round(float(np.median(errs_obs)), 1) if errs_obs else None,
                },
                "observed_plus_interpolated": {
                    "tp": tp_active, "fp": fp_active, "fn": fn_active, "neg_fp": neg_fp_active,
                    "precision": round(tp_active / float(tp_active + fp_active), 4) if (tp_active + fp_active) > 0 else None,
                    "recall": round(tp_active / float(max(1, vis_n)), 4) if vis_n > 0 else None,
                    "median_err_px": round(float(np.median(errs_active)), 1) if errs_active else None,
                },
                "details": details,
            }

    # 3. BAS acting-team alignment (point-in-time action anchor diagnostic, NOT continuous interval GT)
    bas_poss_check: Dict[str, Any] = {
        "available": False,
        "caveat": (
            "BAS actions provide point-in-time acting-team labels (left/right) at discrete ball actions, "
            "NOT continuous human-annotated possession intervals; continuous possession percentages remain "
            "unvalidated unless independent interval ground truth is supplied."
        ),
    }
    if bas_json_path and bas_json_path.is_file() and poss_by_f:
        bas_data = json.loads(bas_json_path.read_text(encoding="utf-8"))
        eval_frames = sorted(poss_by_f.keys())
        min_f, max_f = eval_frames[0], eval_frames[-1]
        n_actions = 0
        n_attributed = 0
        n_agree = 0
        n_disagree = 0
        n_unknown_or_loose = 0
        action_details = []
        for act in bas_data.get("actions", []):
            if not str(act.get("gameTime", "")).startswith("1 -"):
                continue
            ev_f = int(act.get("frame", -1))
            if ev_f < min_f + 1 or ev_f > max_f + 1:
                continue
            team_side = act.get("team")
            if team_side not in ("left", "right"):
                continue
            expected_lbl = left_pred_label if team_side == "left" else right_pred_label
            # Check a small +-4 frame window around the BAS action (accounting for stride=2 and min_control_frames hysteresis)
            window_frames = [pf for pf in eval_frames if abs((pf + 1) - ev_f) <= 6]
            if not window_frames:
                continue
            n_actions += 1
            nearest_pf = min(window_frames, key=lambda pf: abs((pf + 1) - ev_f))
            # Check controlling team at nearest frame or within +-6 frame action window
            win_teams = [
                poss_by_f[pf].get("controlling_team_id", "")
                for pf in window_frames
                if poss_by_f[pf].get("controlling_team_id", "") in (InternalLabel.TEAM_A.value, InternalLabel.TEAM_B.value)
            ]
            nearest_state = poss_by_f[nearest_pf].get("possession_state", "")
            nearest_team = poss_by_f[nearest_pf].get("controlling_team_id", "")
            if win_teams:
                dom_team = Counter(win_teams).most_common(1)[0][0]
                n_attributed += 1
                agreed = dom_team == expected_lbl
                if agreed:
                    n_agree += 1
                else:
                    n_disagree += 1
            else:
                dom_team = "Unknown/Unattributed"
                agreed = None
                n_unknown_or_loose += 1
            action_details.append({
                "bas_frame_1b": ev_f,
                "label": act.get("label"),
                "bas_team_side": team_side,
                "expected_internal_team": expected_lbl,
                "nearest_eval_frame": nearest_pf,
                "nearest_state": nearest_state,
                "nearest_team": nearest_team,
                "window_attributed_team": dom_team,
                "agreed": agreed,
            })

        bas_poss_check.update({
            "available": True,
            "team_alignment_map": {"left": left_pred_label, "right": right_pred_label, "permutation_swapped": use_swap},
            "bas_actions_in_window": n_actions,
            "attributed_actions": n_attributed,
            "unattributed_unknown_or_loose": n_unknown_or_loose,
            "attributed_agreement_count": n_agree,
            "attributed_disagreement_count": n_disagree,
            "attributed_agreement_rate": round(n_agree / float(n_attributed), 4) if n_attributed > 0 else None,
            "actions": action_details,
        })

    # State counts from possession timeline
    state_counts = dict(Counter(r.get("possession_state", "UNKNOWN") for r in poss_by_f.values()))
    n_poss_f = len(poss_by_f)
    state_pct = {k: round(v / float(max(1, n_poss_f)) * 100.0, 2) for k, v in state_counts.items()}

    return {
        "match_id": match_id,
        "run_dir": str(run_dir),
        "ball_trajectory": {
            "total_frames": len(ball_recs),
            "observed_frames": len(obs_recs),
            "interpolated_frames": len(interp_recs),
            "lost_frames": len(lost_recs),
            "observed_pct": round(len(obs_recs) / float(max(1, len(ball_recs))) * 100.0, 2),
            "active_pct": round((len(obs_recs) + len(interp_recs)) / float(max(1, len(ball_recs))) * 100.0, 2),
            "detection_sources": src_counts,
            "false_track_associations": (eval_data.get("possession_summary_excerpt", {}).get("ball_tracking") or {}).get("false_track_associations"),
            "permanent_static_zones_count": (eval_data.get("possession_summary_excerpt", {}).get("ball_tracking") or {}).get("permanent_static_zones_count"),
        },
        "verified_ball_overlap_eval": verified_ball_check,
        "possession_state_distribution_pct": state_pct,
        "bas_acting_team_diagnostic": bas_poss_check,
    }


def benchmark_bytetrack_vs_botsort_identical_detections(
    match_id: str,
    video_path: Path,
    gsr_json: Path,
    raw_dir: Path,
    det_cfg: DetectionConfig,
    max_frames: int = 180,
    stride: int = 2,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run ByteTrack and BoT-SORT on IDENTICAL native-resolution tiled detections + pitch filter.

    Guarantees identical per-frame detections (`DetA` controlled) so association gains (`AssA`,
    `IDF1`, `IDSW`, `Fragmentations`, `MT/PT/ML`) are measured cleanly without detector noise.
    """
    prf = PitchRegionFilter.from_calibration_path(raw_dir, coordinate_space="raw_fisheye", config=PitchRegionConfig())
    proj = load_fisheye_pitch_projector(raw_dir, match_id)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = prf.get_tiling_roi_bbox((fh, fw))

    cached_players: List[Tuple[int, float, List[DetectionRecord]]] = []
    if cache_path is not None and cache_path.is_file():
        cap.release()
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        det_wall_s = float(raw_cache["det_wall_time_s"])
        for entry in raw_cache["frames"]:
            recs = [
                DetectionRecord(
                    frame_number=int(entry["frame"]),
                    timestamp=float(entry["ts"]),
                    class_id=0,
                    class_name="player",
                    bbox=tuple(int(v) for v in p["bbox"]),  # type: ignore[arg-type]
                    confidence=float(p["conf"]),
                    tracking_id=None,
                )
                for p in entry["players"]
            ]
            cached_players.append((int(entry["frame"]), float(entry["ts"]), recs))
        print(f"Loaded {len(cached_players)} cached detection frames from {cache_path}", flush=True)
    else:
        detector = FootballDetector(config=det_cfg)
        t_det_start = time.perf_counter()
        f_idx = 0
        yielded = 0
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if f_idx % stride != 0:
                f_idx += 1
                continue
            ts = round(f_idx / fps, 3)
            raw_dets = detector.detect_frame_tiled(frame, f_idx, ts, roi_bbox=roi)
            players = [d for d in raw_dets if d.class_name == "player" and prf.is_player_in_region(d.bbox, include_margin=True)]
            cached_players.append((f_idx, ts, players))
            yielded += 1
            f_idx += 1
        cap.release()
        det_wall_s = time.perf_counter() - t_det_start
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "match_id": match_id,
                        "det_wall_time_s": round(det_wall_s, 2),
                        "frames": [
                            {
                                "frame": f,
                                "ts": t,
                                "players": [{"bbox": list(p.bbox), "conf": p.confidence} for p in ps],
                            }
                            for f, t, ps in cached_players
                        ],
                    }
                ),
                encoding="utf-8",
            )
            print(f"Cached {len(cached_players)} detection frames to {cache_path}", flush=True)

    eval_frames = [f for f, _, _ in cached_players]
    gsr, _ = load_gsr_frames_streaming(gsr_json, {f + 1 for f in eval_frames}, stop_after_frame=max(eval_frames) + 10)
    eff_fps = max(1, int(round(fps / float(stride))))

    results: Dict[str, Any] = {
        "match_id": match_id,
        "frames_evaluated": len(eval_frames),
        "frame_range": [eval_frames[0], eval_frames[-1]],
        "detector_config": {
            "model": det_cfg.model_name,
            "person_conf": det_cfg.person_conf,
            "tile_size": f"{det_cfg.tile_width}x{det_cfg.tile_height}@{det_cfg.tile_imgsz}",
            "pitch_roi": list(roi),
            "det_wall_time_s": round(det_wall_s, 2),
            "det_ms_per_frame": round(det_wall_s / max(1, len(eval_frames)) * 1000.0, 1),
            "avg_players_detected": round(float(np.mean([len(p) for _, _, p in cached_players])), 2),
        },
        "trackers": {},
    }

    for trk_name in ("bytetrack.yaml", "botsort.yaml"):
        mot = _build_ultralytics_tracker(trk_name, frame_rate=eff_fps, disable_gmc=True)
        obs_by_frame: Dict[int, List[FrameObservationRecord]] = {}
        t0 = time.perf_counter()
        for fno, ts, players in cached_players:
            arr = mot.update(_NumpyDetections.from_records(players), img=None)
            recs: List[FrameObservationRecord] = []
            if arr is not None and len(arr) > 0:
                for row in arr:
                    bbox = tuple(int(round(float(v))) for v in row[:4])
                    tid = int(round(float(row[4])))
                    conf = round(float(row[5]), 4)
                    recs.append(
                        FrameObservationRecord(
                            match_id=match_id,
                            frame_number=fno,
                            timestamp=ts,
                            player_id=tid,
                            team_id=InternalLabel.UNKNOWN.value,
                            team_display_name="Unknown",
                            raw_team_id=InternalLabel.UNKNOWN.value,
                            player_bbox=bbox,  # type: ignore[arg-type]
                            player_detection_confidence=conf,
                            classification_confidence=0.0,
                            classification_distance=1.0,
                            track_team_confidence=0.0,
                            is_team_locked=False,
                        )
                    )
            obs_by_frame[fno] = recs
        trk_ms = (time.perf_counter() - t0) / max(1, len(eval_frames)) * 1000.0

        ev5 = evaluate_pitch_space_gsr(eval_frames, obs_by_frame, gsr, frame_offset=1, projector=proj, match_radius_m=5.0)
        ev8 = evaluate_pitch_space_gsr(eval_frames, obs_by_frame, gsr, frame_offset=1, projector=proj, match_radius_m=8.0) if match_id == "117092" else None
        t5 = dict(ev5["tracking"])
        t5.pop("per_gt_player", None)
        t8 = dict(ev8["tracking"]) if ev8 else None
        if t8:
            t8.pop("per_gt_player", None)
        results["trackers"][trk_name] = {
            "tracker_ms_per_frame": round(trk_ms, 2),
            "pitch_5m": t5,
            "pitch_8m": t8,
        }
        print(
            f"  [{match_id}] {trk_name:<15} ({trk_ms:.2f}ms/f) | "
            f"5m: GS-HOTA={t5['gs_hota']:.4f} (DetA={t5['gs_deta']:.4f} AssA={t5['gs_assa']:.4f} LocA={t5['gs_loca']:.4f}) "
            f"MOTA={t5['mota_all']:.4f} IDF1={t5['idf1']:.4f} IDSW={t5['id_switches']} Frag={t5['fragmentations']} "
            f"MT={t5['mostly_tracked_mt']} PT={t5['partially_tracked_pt']} ML={t5['mostly_lost_ml']} Pur={t5['mean_gt_dominant_id_purity']:.3f}"
        )
        if t8:
            print(
                f"                  8m: GS-HOTA={t8['gs_hota']:.4f} (DetA={t8['gs_deta']:.4f} AssA={t8['gs_assa']:.4f}) "
                f"MOTA={t8['mota_all']:.4f} IDF1={t8['idf1']:.4f} IDSW={t8['id_switches']} Frag={t8['fragmentations']} MT={t8['mostly_tracked_mt']}"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3.2 Tracking & Possession Evaluation Utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_trk = sub.add_parser("compare-trackers", help="Head-to-head ByteTrack vs BoT-SORT on identical tiled detections")
    p_trk.add_argument("--match-id", required=True)
    p_trk.add_argument("--model", default="yolo11s.pt")
    p_trk.add_argument("--person-conf", type=float, default=0.28)
    p_trk.add_argument("--max-frames", type=int, default=180)
    p_trk.add_argument("--out-json", type=Path, required=True)

    p_pos = sub.add_parser("audit-possession", help="Audit ball trajectory & BAS acting-team alignment")
    p_pos.add_argument("--run-dir", type=Path, required=True)
    p_pos.add_argument("--eval-json", type=Path, required=True)
    p_pos.add_argument("--match-id", required=True)
    p_pos.add_argument("--bas-json", type=Path, default=None)
    p_pos.add_argument("--verified-ball-json", type=Path, default=None)
    p_pos.add_argument("--out-json", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "compare-trackers":
        mid = args.match_id
        det_cfg = DetectionConfig(
            model_name=args.model,
            person_conf=args.person_conf,
            ball_conf=0.14,
            inference_mode="tiled",
            tile_width=1280,
            tile_height=1280,
            tile_overlap=0.22,
            tile_imgsz=1280,
        )
        res = benchmark_bytetrack_vs_botsort_identical_detections(
            match_id=mid,
            video_path=Path(f"SoccerTrack-v2/data/videos/{mid}/{mid}_panorama_1st_half.mp4"),
            gsr_json=Path(f"SoccerTrack-v2/data/gsr/{mid}/{mid}_1st.json"),
            raw_dir=Path(f"SoccerTrack-v2/data/raw/{mid}"),
            det_cfg=det_cfg,
            max_frames=args.max_frames,
            cache_path=args.out_json.parent / f"tracker_det_cache_{mid}.json",
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved {args.out_json}")
    elif args.cmd == "audit-possession":
        res = evaluate_possession_and_ball_vs_bas_and_verified(
            run_dir=args.run_dir,
            eval_json_path=args.eval_json,
            bas_json_path=args.bas_json,
            verified_ball_json_path=args.verified_ball_json,
            match_id=args.match_id,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

