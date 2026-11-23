"""Sprint 3.3 Player Tracking & Long-Sequence Continuity Benchmark (`scripts/benchmark_sprint3_3_tracking.py`).

1. Head-to-head comparison on IDENTICAL cached tiled detections (`outputs/sprint3_2/tracker_det_cache_117092.json`
   and `outputs/sprint3_2/tracker_det_cache_118575.json`) across:
     - `ByteTrack_Baseline` (Sprint 3.2 default)
     - `BoTSORT_Baseline` (Sprint 3.2 default)
     - `ByteTrack_FixedCam_Stitched` (Sprint 3.3 fixed-camera parameters + causal pitch-plane/appearance stitcher)
     - `BoTSORT_FixedCam_Stitched` (Sprint 3.3 fixed-camera parameters + causal pitch-plane/appearance stitcher)
   Evaluates at both `5.0 m` and `8.0 m` pitch-space tolerances:
     - `GS-HOTA`, `DetA`, `AssA`, `LocA` (explicitly isolating Association Accuracy `AssA` gains from `DetA`)
     - `IDF1`, `MOTA`, `IDSW`, `Fragmentations`
     - `Total Predicted Tracks` (vs GT Tracks), `Mean Track Lifetime (frames)`, `MT`, `PT`, `ML`
   Writes results to `outputs/sprint3_3/player_tracking/tracker_comparison.json`.

2. Long-sequence continuous tracking evaluation on `118575` (`750` video frames = `30 seconds` continuous at
   `stride=2` -> `375` sampled frames), caching the extended detections from `f=360..748` so downstream
   end-to-end validation can reuse them, and writing results to
   `outputs/sprint3_3/player_tracking/long_sequence_tracking.json`.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scripts.evaluate_soccertrack_v2 import (
    evaluate_pitch_space_gsr,
    load_fisheye_pitch_projector,
    load_gsr_frames_streaming,
)
from src.config import DetectionConfig
from src.detection.detector import FootballDetector
from src.geometry.pitch_region import PitchRegionConfig, PitchRegionFilter
from src.schemas import DetectionRecord, FrameObservationRecord, InternalLabel
from src.tracking.pitch_stitcher import FixedCameraTrackletAssociator, StitcherConfig
from src.tracking.tracker import _NumpyDetections, _build_ultralytics_tracker


TRACKER_VARIANTS: Tuple[Tuple[str, str, bool, bool], ...] = (
    ("ByteTrack_Baseline", "bytetrack.yaml", False, False),
    ("BoTSORT_Baseline", "botsort.yaml", False, False),
    ("ByteTrack_FixedCam_Stitched", "bytetrack.yaml", True, True),
    ("BoTSORT_FixedCam_Stitched", "botsort.yaml", True, True),
)


def _load_or_extend_cached_detections(
    match_id: str,
    video_path: Path,
    prf: PitchRegionFilter,
    det_cfg: DetectionConfig,
    target_sampled_frames: int,
    stride: int,
    base_cache_path: Path,
    extended_cache_path: Optional[Path] = None,
    load_video_frames_for_appearance: bool = False,
) -> Tuple[List[Tuple[int, float, List[DetectionRecord], List[DetectionRecord]]], Dict[int, np.ndarray], float]:
    """Load cached player (+ optional ball) detections up to `target_sampled_frames` sampled frames.

    If `extended_cache_path` exists with >= `target_sampled_frames`, loads directly from it.
    Otherwise loads initial frames from `base_cache_path` (e.g. `f=0..358`) and runs the tiled
    detector only on the remaining frames (`f=360..`), saving the combined cache to `extended_cache_path`.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = prf.get_tiling_roi_bbox((fh, fw))

    cached_entries: List[Tuple[int, float, List[DetectionRecord], List[DetectionRecord]]] = []
    det_wall_s = 0.0

    source_cache = (
        extended_cache_path
        if (extended_cache_path is not None and extended_cache_path.is_file())
        else base_cache_path
    )
    if source_cache.is_file():
        raw_cache = json.loads(source_cache.read_text(encoding="utf-8"))
        det_wall_s = float(raw_cache.get("det_wall_time_s", 0.0))
        for entry in raw_cache.get("frames", [])[:target_sampled_frames]:
            fno = int(entry["frame"])
            ts = float(entry["ts"])
            players = [
                DetectionRecord(
                    frame_number=fno,
                    timestamp=ts,
                    class_id=0,
                    class_name="player",
                    bbox=tuple(int(v) for v in p["bbox"]),  # type: ignore[arg-type]
                    confidence=float(p["conf"]),
                    tracking_id=None,
                )
                for p in entry.get("players", [])
            ]
            balls = [
                DetectionRecord(
                    frame_number=fno,
                    timestamp=ts,
                    class_id=32,
                    class_name="ball",
                    bbox=tuple(int(v) for v in b["bbox"]),  # type: ignore[arg-type]
                    confidence=float(b["conf"]),
                    tracking_id=None,
                )
                for b in entry.get("balls", [])
            ]
            cached_entries.append((fno, ts, players, balls))
        print(f"[{match_id}] Loaded {len(cached_entries)}/{target_sampled_frames} frames from {source_cache}", flush=True)

    if len(cached_entries) < target_sampled_frames:
        start_video_f = (cached_entries[-1][0] + stride) if cached_entries else 0
        needed = target_sampled_frames - len(cached_entries)
        print(
            f"[{match_id}] Running tiled detector for {needed} additional sampled frames starting at video frame {start_video_f}...",
            flush=True,
        )
        detector = FootballDetector(config=det_cfg)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_f)
        f_idx = start_video_f
        added = 0
        t0 = time.perf_counter()
        while added < needed:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (f_idx - start_video_f) % stride != 0:
                f_idx += 1
                continue
            ts = round(f_idx / fps, 3)
            raw_dets = detector.detect_frame_tiled(frame, f_idx, ts, roi_bbox=roi)
            players = [
                d for d in raw_dets if d.class_name == "player" and prf.is_player_in_region(d.bbox, include_margin=True)
            ]
            balls = [
                d for d in raw_dets if d.class_name == "ball" and prf.is_ball_in_region(d.bbox, include_margin=True)
            ]
            cached_entries.append((f_idx, ts, players, balls))
            added += 1
            f_idx += 1
            if added % 25 == 0 or added == needed:
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{match_id}] Detected +{added}/{needed} frames (f={f_idx - 1}, {elapsed:.1f}s elapsed)",
                    flush=True,
                )
        extra_s = time.perf_counter() - t0
        det_wall_s += extra_s

        save_target = extended_cache_path or base_cache_path
        save_target.parent.mkdir(parents=True, exist_ok=True)
        save_target.write_text(
            json.dumps(
                {
                    "match_id": match_id,
                    "det_wall_time_s": round(det_wall_s, 2),
                    "frames": [
                        {
                            "frame": f,
                            "ts": t,
                            "players": [{"bbox": list(p.bbox), "conf": p.confidence} for p in ps],
                            "balls": [{"bbox": list(b.bbox), "conf": b.confidence} for b in bs],
                        }
                        for f, t, ps, bs in cached_entries
                    ],
                }
            ),
            encoding="utf-8",
        )
        print(f"[{match_id}] Saved {len(cached_entries)} cached frames to {save_target}", flush=True)

    frames_bgr_by_fno: Dict[int, np.ndarray] = {}
    if load_video_frames_for_appearance and cached_entries:
        target_fnos = {f for f, _, _, _ in cached_entries}
        max_fno = max(target_fnos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cur_f = 0
        while cur_f <= max_fno:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            if cur_f in target_fnos:
                frames_bgr_by_fno[cur_f] = frm
            cur_f += 1
        print(f"[{match_id}] Loaded {len(frames_bgr_by_fno)} video frames for appearance gating.", flush=True)

    cap.release()
    return cached_entries[:target_sampled_frames], frames_bgr_by_fno, det_wall_s


def _format_tracking_metrics(
    raw_tracking_dict: Dict[str, Any],
    all_pred_track_counts: Counter[int],
    tracker_ms_per_frame: float,
    stitch_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract and enrich pitch-space tracking metrics with total predicted tracks and track lifetimes."""
    lifetimes = list(all_pred_track_counts.values())
    total_pred_tracks = len(lifetimes)
    mean_lifetime = round(float(np.mean(lifetimes)), 2) if lifetimes else 0.0
    median_lifetime = round(float(np.median(lifetimes)), 2) if lifetimes else 0.0
    short_tracks_under_5 = sum(1 for c in lifetimes if c < 5)

    out = {
        "gs_hota": raw_tracking_dict["gs_hota"],
        "deta": raw_tracking_dict["gs_deta"],
        "assa": raw_tracking_dict["gs_assa"],
        "loca": raw_tracking_dict["gs_loca"],
        "idf1": raw_tracking_dict["idf1"],
        "idp": raw_tracking_dict["idp"],
        "idr": raw_tracking_dict["idr"],
        "mota": raw_tracking_dict["mota_all"],
        "mota_on_pitch": raw_tracking_dict["mota_on_pitch"],
        "id_switches": raw_tracking_dict["id_switches"],
        "fragmentations": raw_tracking_dict["fragmentations"],
        "gt_unique_identities": raw_tracking_dict["gt_unique_identities"],
        "total_predicted_tracks": total_pred_tracks,
        "pred_unique_identities_matched": raw_tracking_dict["pred_unique_identities_matched"],
        "mean_track_lifetime_frames": mean_lifetime,
        "median_track_lifetime_frames": median_lifetime,
        "short_tracks_under_5_frames": short_tracks_under_5,
        "mostly_tracked_mt": raw_tracking_dict["mostly_tracked_mt"],
        "partially_tracked_pt": raw_tracking_dict["partially_tracked_pt"],
        "mostly_lost_ml": raw_tracking_dict["mostly_lost_ml"],
        "mean_gt_dominant_id_purity": raw_tracking_dict["mean_gt_dominant_id_purity"],
        "mean_distinct_pred_ids_per_gt_player": raw_tracking_dict["mean_distinct_pred_ids_per_gt_player"],
        "detection_precision": raw_tracking_dict["detection_precision_all"],
        "detection_recall": raw_tracking_dict["detection_recall"],
        "detection_f1": raw_tracking_dict["detection_f1_all"],
        "mean_matched_distance_m": raw_tracking_dict["mean_matched_distance_m"],
        "median_matched_distance_m": raw_tracking_dict["median_matched_distance_m"],
        "tp": raw_tracking_dict["tp"],
        "fp": raw_tracking_dict["fp_all"],
        "fn": raw_tracking_dict["fn"],
        "tracker_ms_per_frame": round(tracker_ms_per_frame, 2),
    }
    if stitch_summary is not None:
        out["stitcher_diagnostics"] = stitch_summary
    return out


def evaluate_tracker_variants_on_sequence(
    match_id: str,
    video_path: Path,
    gsr_json: Path,
    raw_dir: Path,
    det_cfg: DetectionConfig,
    target_sampled_frames: int,
    stride: int,
    base_cache_path: Path,
    extended_cache_path: Optional[Path] = None,
    use_frame_appearance: bool = False,
) -> Dict[str, Any]:
    """Run all 4 tracker variants on identical detections and score at 5.0m and 8.0m pitch tolerances."""
    prf = PitchRegionFilter.from_calibration_path(
        raw_dir, coordinate_space="raw_fisheye", config=PitchRegionConfig()
    )
    proj = load_fisheye_pitch_projector(raw_dir, match_id)
    if proj is None:
        raise RuntimeError(f"Could not load fisheye pitch projector from {raw_dir}")

    cached_entries, frames_bgr_by_fno, det_wall_s = _load_or_extend_cached_detections(
        match_id=match_id,
        video_path=video_path,
        prf=prf,
        det_cfg=det_cfg,
        target_sampled_frames=target_sampled_frames,
        stride=stride,
        base_cache_path=base_cache_path,
        extended_cache_path=extended_cache_path,
        load_video_frames_for_appearance=use_frame_appearance,
    )

    eval_frames = [f for f, _, _, _ in cached_entries]
    gsr, _ = load_gsr_frames_streaming(
        gsr_json,
        {f + 1 for f in eval_frames},
        stop_after_frame=max(eval_frames) + 10,
    )
    fps = 25.0
    eff_fps = max(1, int(round(fps / float(stride))))

    def _p2p_tuple(u: float, v: float) -> Optional[Tuple[float, float]]:
        arr = proj["pixel_to_pitch"](np.array([[u, v]], dtype=np.float64))[0]
        xm, ym = float(arr[0]), float(arr[1])
        return (xm, ym) if (np.isfinite(xm) and np.isfinite(ym)) else None

    seq_result: Dict[str, Any] = {
        "match_id": match_id,
        "frames_evaluated": len(eval_frames),
        "video_frame_range_0based": [eval_frames[0], eval_frames[-1]],
        "duration_seconds": round((eval_frames[-1] - eval_frames[0] + stride) / fps, 2),
        "frame_stride": stride,
        "detector_config": {
            "model": det_cfg.model_name,
            "person_conf": det_cfg.person_conf,
            "tile_size": f"{det_cfg.tile_width}x{det_cfg.tile_height}@{det_cfg.tile_imgsz}",
            "det_wall_time_s": round(det_wall_s, 2),
            "avg_players_detected_per_frame": round(
                float(np.mean([len(ps) for _, _, ps, _ in cached_entries])), 2
            ),
        },
        "variants": {},
    }

    for variant_name, yaml_name, fixed_cam_mode, use_stitcher in TRACKER_VARIANTS:
        mot = _build_ultralytics_tracker(
            yaml_name,
            frame_rate=eff_fps,
            disable_gmc=True,
            fixed_camera_mode=fixed_cam_mode,
        )
        stitcher = (
            FixedCameraTrackletAssociator(
                config=StitcherConfig(enabled=True),
                pitch_region_filter=prf,
                pitch_projector_fn=_p2p_tuple,
            )
            if use_stitcher
            else None
        )

        obs_by_frame: Dict[int, List[FrameObservationRecord]] = {}
        all_pred_track_counts: Counter[int] = Counter()
        t0 = time.perf_counter()

        for fno, ts, players, _ in cached_entries:
            arr = mot.update(_NumpyDetections.from_records(players), img=None)
            det_recs: List[DetectionRecord] = []
            if arr is not None and len(arr) > 0:
                for row in arr:
                    bbox = tuple(int(round(float(v))) for v in row[:4])
                    tid = int(round(float(row[4])))
                    conf = round(float(row[5]), 4)
                    det_recs.append(
                        DetectionRecord(
                            frame_number=fno,
                            timestamp=ts,
                            class_id=0,
                            class_name="player",
                            bbox=bbox,  # type: ignore[arg-type]
                            confidence=conf,
                            tracking_id=tid,
                        )
                    )
            if stitcher is not None and det_recs:
                det_recs = stitcher.update(
                    frame_number=fno,
                    player_records=det_recs,
                    frame_bgr=frames_bgr_by_fno.get(fno),
                )

            frame_obs: List[FrameObservationRecord] = []
            seen_tids_in_frame = set()
            for r in det_recs:
                if r.tracking_id is None:
                    continue
                tid = int(r.tracking_id)
                assert tid not in seen_tids_in_frame, f"Duplicate track_id {tid} in frame {fno}!"
                seen_tids_in_frame.add(tid)
                all_pred_track_counts[tid] += 1
                frame_obs.append(
                    FrameObservationRecord(
                        match_id=match_id,
                        frame_number=fno,
                        timestamp=ts,
                        player_id=tid,
                        team_id=InternalLabel.UNKNOWN.value,
                        team_display_name="Unknown",
                        raw_team_id=InternalLabel.UNKNOWN.value,
                        player_bbox=r.bbox,
                        player_detection_confidence=r.confidence,
                        classification_confidence=0.0,
                        classification_distance=1.0,
                        track_team_confidence=0.0,
                        is_team_locked=False,
                    )
                )
            obs_by_frame[fno] = frame_obs

        trk_ms = (time.perf_counter() - t0) / max(1, len(eval_frames)) * 1000.0
        stitch_sum = stitcher.summarize() if stitcher is not None else None

        ev5 = evaluate_pitch_space_gsr(
            eval_frames, obs_by_frame, gsr, frame_offset=1, projector=proj, match_radius_m=5.0
        )
        ev8 = evaluate_pitch_space_gsr(
            eval_frames, obs_by_frame, gsr, frame_offset=1, projector=proj, match_radius_m=8.0
        )
        m5 = _format_tracking_metrics(ev5["tracking"], all_pred_track_counts, trk_ms, stitch_sum)
        m8 = _format_tracking_metrics(ev8["tracking"], all_pred_track_counts, trk_ms, stitch_sum)

        seq_result["variants"][variant_name] = {
            "tracker_yaml": yaml_name,
            "fixed_camera_mode": fixed_cam_mode,
            "stitch_lost_tracks": use_stitcher,
            "tracker_ms_per_frame": round(trk_ms, 2),
            "pitch_5m": m5,
            "pitch_8m": m8,
        }
        print(
            f"  [{match_id} | {len(eval_frames)}f] {variant_name:<28} ({trk_ms:.2f}ms/f)\n"
            f"    5.0m: GS-HOTA={m5['gs_hota']:.4f} (DetA={m5['deta']:.4f} AssA={m5['assa']:.4f} LocA={m5['loca']:.4f}) "
            f"IDF1={m5['idf1']:.4f} MOTA={m5['mota']:.4f} IDSW={m5['id_switches']:<3} Frag={m5['fragmentations']:<3} "
            f"PredTrks={m5['total_predicted_tracks']:<3} (Matched={m5['pred_unique_identities_matched']:<2}/GT={m5['gt_unique_identities']}) "
            f"MeanLife={m5['mean_track_lifetime_frames']:.1f}f MT={m5['mostly_tracked_mt']}/PT={m5['partially_tracked_pt']}/ML={m5['mostly_lost_ml']}\n"
            f"    8.0m: GS-HOTA={m8['gs_hota']:.4f} (DetA={m8['deta']:.4f} AssA={m8['assa']:.4f} LocA={m8['loca']:.4f}) "
            f"IDF1={m8['idf1']:.4f} MOTA={m8['mota']:.4f} IDSW={m8['id_switches']:<3} Frag={m8['fragmentations']:<3}",
            flush=True,
        )

    # Compute explicit deltas isolating AssA gains vs DetA stability
    for base_key, stitched_key in (
        ("ByteTrack_Baseline", "ByteTrack_FixedCam_Stitched"),
        ("BoTSORT_Baseline", "BoTSORT_FixedCam_Stitched"),
    ):
        b5 = seq_result["variants"][base_key]["pitch_5m"]
        s5 = seq_result["variants"][stitched_key]["pitch_5m"]
        b8 = seq_result["variants"][base_key]["pitch_8m"]
        s8 = seq_result["variants"][stitched_key]["pitch_8m"]
        seq_result["variants"][stitched_key]["delta_vs_baseline_5m"] = {
            "gs_hota_delta": round(s5["gs_hota"] - b5["gs_hota"], 4),
            "deta_delta": round(s5["deta"] - b5["deta"], 4),
            "assa_delta": round(s5["assa"] - b5["assa"], 4),
            "idf1_delta": round(s5["idf1"] - b5["idf1"], 4),
            "mota_delta": round(s5["mota"] - b5["mota"], 4),
            "id_switches_delta": int(s5["id_switches"] - b5["id_switches"]),
            "fragmentations_delta": int(s5["fragmentations"] - b5["fragmentations"]),
            "total_predicted_tracks_delta": int(s5["total_predicted_tracks"] - b5["total_predicted_tracks"]),
            "mean_track_lifetime_frames_delta": round(
                s5["mean_track_lifetime_frames"] - b5["mean_track_lifetime_frames"], 2
            ),
        }
        seq_result["variants"][stitched_key]["delta_vs_baseline_8m"] = {
            "gs_hota_delta": round(s8["gs_hota"] - b8["gs_hota"], 4),
            "deta_delta": round(s8["deta"] - b8["deta"], 4),
            "assa_delta": round(s8["assa"] - b8["assa"], 4),
            "idf1_delta": round(s8["idf1"] - b8["idf1"], 4),
            "mota_delta": round(s8["mota"] - b8["mota"], 4),
            "id_switches_delta": int(s8["id_switches"] - b8["id_switches"]),
            "fragmentations_delta": int(s8["fragmentations"] - b8["fragmentations"]),
            "total_predicted_tracks_delta": int(s8["total_predicted_tracks"] - b8["total_predicted_tracks"]),
            "mean_track_lifetime_frames_delta": round(
                s8["mean_track_lifetime_frames"] - b8["mean_track_lifetime_frames"], 2
            ),
        }

    return seq_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3.3 Player Tracking & Long-Sequence Continuity Benchmark")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/sprint3_3/player_tracking"),
        help="Output directory for tracker_comparison.json and long_sequence_tracking.json",
    )
    parser.add_argument(
        "--long-frames",
        type=int,
        default=375,
        help="Sampled frames for 30s continuous evaluation on 118575 (375 sampled frames @ stride=2 = 750 video frames = 30.0s).",
    )
    parser.add_argument(
        "--disable-frame-appearance",
        action="store_true",
        help="Skip decoding video frames (run pitch-plane + kinematic stitching without HSV+LAB torso appearance).",
    )
    args = parser.parse_args()
    use_app = not args.disable_frame_appearance

    det_cfg = DetectionConfig(
        model_name="yolo11s.pt",
        person_conf=0.28,
        ball_conf=0.14,
        inference_mode="tiled",
        tile_width=1280,
        tile_height=1280,
        tile_overlap=0.22,
        tile_imgsz=1280,
        frame_stride=2,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard 180-sampled-frame comparison on 117092 (Night dev) and 118575 (Day dev)
    print("=== Running Sprint 3.3 Tracker Comparison on 117092 (180 sampled frames) ===", flush=True)
    res_117092 = evaluate_tracker_variants_on_sequence(
        match_id="117092",
        video_path=Path("SoccerTrack-v2/data/videos/117092/117092_panorama_1st_half.mp4"),
        gsr_json=Path("SoccerTrack-v2/data/gsr/117092/117092_1st.json"),
        raw_dir=Path("SoccerTrack-v2/data/raw/117092"),
        det_cfg=det_cfg,
        target_sampled_frames=180,
        stride=2,
        base_cache_path=Path("outputs/sprint3_2/tracker_det_cache_117092.json"),
        use_frame_appearance=use_app,
    )

    print("\n=== Running Sprint 3.3 Tracker Comparison on 118575 (180 sampled frames) ===", flush=True)
    res_118575 = evaluate_tracker_variants_on_sequence(
        match_id="118575",
        video_path=Path("SoccerTrack-v2/data/videos/118575/118575_panorama_1st_half.mp4"),
        gsr_json=Path("SoccerTrack-v2/data/gsr/118575/118575_1st.json"),
        raw_dir=Path("SoccerTrack-v2/data/raw/118575"),
        det_cfg=det_cfg,
        target_sampled_frames=180,
        stride=2,
        base_cache_path=Path("outputs/sprint3_2/tracker_det_cache_118575.json"),
        use_frame_appearance=use_app,
    )

    comparison_out = {
        "benchmark": "sprint3_3_fixed_camera_player_tracking_comparison",
        "description": (
            "Controlled comparison of ByteTrack vs BoT-SORT baseline and Sprint 3.3 fixed-camera stitched variants "
            "on identical cached tiled detections (isolating Association Accuracy AssA from Detection Accuracy DetA)."
        ),
        "matches": {
            "117092": res_117092,
            "118575": res_118575,
        },
    }
    comp_path = args.out_dir / "tracker_comparison.json"
    comp_path.write_text(json.dumps(comparison_out, indent=2), encoding="utf-8")
    print(f"\nSaved tracker comparison to {comp_path}", flush=True)

    # 2. Long-sequence (30s = 750 video frames = 375 sampled frames @ stride=2) evaluation on 118575
    print(f"\n=== Running Sprint 3.3 Long-Sequence ({args.long_frames} sampled frames / 30s) on 118575 ===", flush=True)
    ext_cache_118575 = args.out_dir / f"tracker_det_cache_118575_{args.long_frames}f.json"
    res_long_118575 = evaluate_tracker_variants_on_sequence(
        match_id="118575",
        video_path=Path("SoccerTrack-v2/data/videos/118575/118575_panorama_1st_half.mp4"),
        gsr_json=Path("SoccerTrack-v2/data/gsr/118575/118575_1st.json"),
        raw_dir=Path("SoccerTrack-v2/data/raw/118575"),
        det_cfg=det_cfg,
        target_sampled_frames=args.long_frames,
        stride=2,
        base_cache_path=Path("outputs/sprint3_2/tracker_det_cache_118575.json"),
        extended_cache_path=ext_cache_118575,
        use_frame_appearance=use_app,
    )

    long_out = {
        "benchmark": "sprint3_3_long_sequence_continuous_player_tracking",
        "match_id": "118575",
        "cached_detections_path": str(ext_cache_118575),
        "evaluation": res_long_118575,
    }
    long_path = args.out_dir / "long_sequence_tracking.json"
    long_path.write_text(json.dumps(long_out, indent=2), encoding="utf-8")
    print(f"Saved long-sequence tracking evaluation to {long_path}", flush=True)


if __name__ == "__main__":
    main()
