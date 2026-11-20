"""Reproducible Sprint 3.1 Stage D benchmark: ByteTrack vs BoT-SORT within-shot continuity,
fragmentation, cross-cut leak prevention, team consistency, and runtime on continuous shots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.classification.dataset import JerseyCropDataset
from src.classification.team_discovery import (
    AutomaticTeamDiscoverer,
    TeamDiscoveryObservation,
    collect_track_appearances,
)
from src.config import TeamDiscoveryConfig, load_match_config
from src.evaluation.report import write_csv_rows
from src.evaluation.tracking_eval import compare_tracking_algorithms


def _fit_unsupervised_discoverer_from_crops(cfg_path: str) -> AutomaticTeamDiscoverer:
    """Fit label-free AutomaticTeamDiscoverer on match crops (no GT labels passed to fit)."""
    cfg = load_match_config(cfg_path)
    ds = JerseyCropDataset(Path("data/crops") / cfg.match_id, cfg)
    samples = ds.get_labelled_samples(exclude_unknown=False, verified_only=True)
    tcfg = TeamDiscoveryConfig(
        enabled=True,
        feature_method="hsv_lab",
        clustering_method="kmeans",
        min_track_observations=1,
    )
    obs = [
        TeamDiscoveryObservation(
            tracking_id=i,
            frame_number=m.frame_number,
            timestamp=m.timestamp,
            crop_bgr=c,
            blur_score=max(1.0, float(m.quality.blur_score)),
            grass_ratio=min(0.79, float(m.quality.grass_ratio)),
            occlusion_ratio=min(0.60, float(m.quality.occlusion_ratio)),
        )
        for i, (m, c, _) in enumerate(samples)
    ]
    apps = collect_track_appearances(obs, tcfg)
    return AutomaticTeamDiscoverer(tcfg).fit(apps)


def main() -> None:
    out_dir = Path("outputs/sprint3_1/player_tracking")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "protocol": (
            "Apples-to-apples comparison of ByteTrack vs BoT-SORT with camera-cut tracker reset "
            "and track-level unsupervised team discovery (hsv_lab + KMeans + TrackAppearanceBuffer) "
            "on continuous wide-angle tactical segments from Chelsea vs Sunderland and Norway vs Wales."
        ),
        "formal_identity_metrics": {
            "metrics_available": False,
            "idf1": None,
            "hota": None,
            "validation_blocker": (
                "No independently verified per-frame multi-object identity annotations exist yet; "
                "within-shot fragmentation (IDs per player slot), track lifetime, short-track fraction, "
                "cross-cut ID leaks, and team-label consistency are measured instead."
            ),
        },
        "segments": {},
    }

    frag_rows: List[Dict[str, Any]] = []

    segments = [
        ("chelsea_sunderland_continuous_shot", "configs/chelsea_sunderland.yaml", 840, 150),
        ("norway_wales_continuous_shot", "configs/norway_wales.yaml", 1000, 120),
    ]

    for seg_name, cfg_path, start_frame, max_frames in segments:
        cfg = load_match_config(cfg_path)
        cfg.tracking.start_frame = start_frame
        cfg.tracking.reset_on_camera_cut = True
        cfg.detection.enable_tiled_ball_search = False  # isolate player tracking speed cleanly
        cfg.team_discovery.enabled = True
        cfg.team_discovery.feature_method = "hsv_lab"
        vpath = cfg.resolve_video_path()
        discoverer = _fit_unsupervised_discoverer_from_crops(cfg_path)

        print(f"\nRunning ByteTrack vs BoT-SORT on {seg_name} (start={start_frame}, frames={max_frames})...")
        comp = compare_tracking_algorithms(
            video_path=vpath,
            match_config=cfg,
            classifier=discoverer,
            tracker_configs=("bytetrack.yaml", "botsort.yaml"),
            max_frames=max_frames,
        )
        results["segments"][seg_name] = {
            "match_id": cfg.match_id,
            "start_frame": start_frame,
            "max_frames": max_frames,
            "frame_stride": cfg.detection.frame_stride,
            "comparison": comp,
        }
        for trk_name, diag in comp.items():
            print(
                f"  {trk_name:15s} | ids={diag['distinct_tracking_ids']:3d} "
                f"| short_frac={diag['short_track_fraction']:.3f} "
                f"| mean_obs={diag['mean_track_observations']:5.1f} "
                f"| ws_ids/slot={diag['within_shot_mean_ids_per_player_slot']} "
                f"| ws_team_cons={diag['within_shot_team_label_consistency']} "
                f"| leaks={diag['cross_cut_id_leaks']} "
                f"| stable_chg={diag['smoothed_team_label_changes']} "
                f"| {diag['ms_per_frame']} ms/f"
            )
            frag_rows.append(
                {
                    "segment": seg_name,
                    "tracker": trk_name,
                    "processed_frames": diag["processed_frames"],
                    "camera_cuts_detected": diag["camera_cuts_detected"],
                    "distinct_tracking_ids": diag["distinct_tracking_ids"],
                    "short_tracks_under_5_obs": diag["short_tracks_under_5_obs"],
                    "short_track_fraction": diag["short_track_fraction"],
                    "mean_track_observations": diag["mean_track_observations"],
                    "max_track_observations": diag["max_track_observations"],
                    "cross_cut_id_leaks": diag["cross_cut_id_leaks"],
                    "within_shot_ids_per_player_slot": diag["within_shot_mean_ids_per_player_slot"],
                    "within_shot_short_track_fraction": diag["within_shot_short_track_fraction"],
                    "within_shot_mean_track_lifetime": diag["within_shot_mean_track_lifetime"],
                    "within_shot_team_consistency": diag["within_shot_team_label_consistency"],
                    "stable_label_coverage": diag["stable_label_coverage"],
                    "raw_team_label_changes": diag["raw_team_label_changes"],
                    "smoothed_team_label_changes": diag["smoothed_team_label_changes"],
                    "ms_per_frame": diag["ms_per_frame"],
                }
            )

    (out_dir / "tracking_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv_rows(out_dir / "fragmentation_report.csv", frag_rows)
    print("\nSaved Stage D tracking deliverables to:", out_dir)


if __name__ == "__main__":
    main()

