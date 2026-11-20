"""Tracker execution supporting ByteTrack and BoT-SORT for players while preserving pre-ByteTrack raw ball detections."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.config import DetectionConfig, TrackingConfig
from src.detection.detector import (
    COCO_BALL_CLASS_ID,
    COCO_PERSON_CLASS_ID,
    FootballDetector,
    _nms_ball_detections,
)
from src.possession.ball_tracker import detect_camera_cut, filter_ball_candidates
from src.schemas import DetectionRecord


SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


def _reset_ultralytics_tracker(model: object) -> None:
    """Safely reset Ultralytics internal MOT tracker state across broadcast camera cuts."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor is not None else None
    if trackers:
        for trk in trackers:
            reset_fn = getattr(trk, "reset", None)
            if callable(reset_fn):
                reset_fn()


class MultiObjectTracker:
    """
    Wraps Ultralytics multi-object tracking (`bytetrack.yaml`, `botsort.yaml`, or custom YAML)
    for players (`class 0`) while capturing raw post-NMS football detections (`class 32`) BEFORE
    ByteTrack's two-frame IoU activation filter (`is_activated`) discards fast-moving balls.

    Sprint 3.1 improvements:
    - Optional camera-cut reset (`reset_on_camera_cut=True`) resets internal tracker state on
      broadcast scene cuts so player IDs do not contaminate different players across shots.
    - Pre-filters obvious false-positive ball detections (crowd/scoreboard/torso/pure-turf) before
      deciding whether tiled pitch fallback should execute, preventing a crowd steward FP from
      suppressing fallback recovery of the actual match ball.
    """

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        device: Optional[str | int] = None,
    ) -> None:
        self.det_config = detection_config or DetectionConfig()
        self.trk_config = tracking_config or TrackingConfig()
        trk_name = str(self.trk_config.tracker_config)
        if trk_name not in SUPPORTED_TRACKERS and not Path(trk_name).is_file():
            raise ValueError(
                f"Unsupported tracker '{self.trk_config.tracker_config}'. "
                f"Supported trackers: {SUPPORTED_TRACKERS} or an existing YAML file path."
            )
        self.device: str | int = (
            device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        )

    def track_video(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        fps_override: Optional[float] = None,
    ) -> Iterator[Tuple[int, float, np.ndarray, List[DetectionRecord]]]:
        """
        Stream tracked player detections and raw/recovered ball detections over a video.
        """
        from ultralytics import YOLO

        vpath = Path(video_path)
        if not vpath.is_file():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {vpath}")
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        fps = fps_override if (fps_override and fps_override > 0) else (raw_fps if raw_fps > 0 else 25.0)
        stride = max(1, int(self.det_config.frame_stride))
        start_frame = max(0, int(self.trk_config.start_frame))

        model = YOLO(self.det_config.model_name)
        min_conf = min(self.det_config.person_conf, self.det_config.ball_conf, 0.14)

        captured_raw_boxes: List[object] = []

        def _capture_raw_postprocess_boxes(predictor: object) -> None:
            captured_raw_boxes.clear()
            results_list = getattr(predictor, "results", None)
            if results_list and len(results_list) > 0 and results_list[0].boxes is not None:
                captured_raw_boxes.append(results_list[0].boxes.cpu())

        model.add_callback("on_predict_postprocess_end", _capture_raw_postprocess_boxes)

        aux_detector = FootballDetector(config=self.det_config, device=self.device)
        prev_small: Optional[np.ndarray] = None
        prev_med_h: Optional[float] = None
        last_cut_frame: Optional[int] = None
        track_id_offset: int = 0
        max_global_id: int = 0

        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_offset = 0
            yielded = 0
            while True:
                if max_frames is not None and yielded >= max_frames:
                    break
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                if frame_offset % stride != 0:
                    frame_offset += 1
                    continue

                actual_frame_number = start_frame + frame_offset
                timestamp = round(actual_frame_number / fps, 3)
                frame_offset += 1

                # Reset internal MOT tracker on broadcast camera cuts and advance global ID offset
                # so track IDs are strictly unique across shots (never recycled to 1).
                if self.trk_config.reset_on_camera_cut:
                    small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
                    if prev_small is not None and detect_camera_cut(
                        prev_small, small_curr, corr_threshold=0.52
                    ):
                        if last_cut_frame is None or (actual_frame_number - last_cut_frame) > 6:
                            _reset_ultralytics_tracker(model)
                            track_id_offset = max_global_id
                            last_cut_frame = actual_frame_number
                            prev_med_h = None
                    prev_small = small_curr

                results = model.track(
                    source=frame_bgr,
                    persist=True,
                    tracker=self.trk_config.tracker_config,
                    classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
                    conf=min_conf,
                    imgsz=self.det_config.imgsz,
                    device=self.device,
                    verbose=False,
                )
                tracked_boxes = results[0].boxes if results else None
                tracked_records = FootballDetector.parse_ultralytics_boxes(
                    tracked_boxes,
                    frame_number=actual_frame_number,
                    timestamp=timestamp,
                    person_conf=self.det_config.person_conf,
                    ball_conf=self.det_config.ball_conf,
                )
                player_records = [r for r in tracked_records if r.class_name == "player"]
                ball_records = [r for r in tracked_records if r.class_name == "ball"]

                # Detect wide-to-closeup scale-jump cuts that preserve green HSV histograms
                if self.trk_config.reset_on_camera_cut and player_records:
                    p_heights = [float(r.bbox[3] - r.bbox[1]) for r in player_records if (r.bbox[3] - r.bbox[1]) > 15]
                    curr_med_h = float(np.median(p_heights)) if p_heights else None
                    if (
                        prev_med_h is not None
                        and curr_med_h is not None
                        and detect_camera_cut(
                            None, None, prev_median_player_h=prev_med_h, curr_median_player_h=curr_med_h
                        )
                        and (last_cut_frame is None or (actual_frame_number - last_cut_frame) > 6)
                    ):
                        _reset_ultralytics_tracker(model)
                        track_id_offset = max_global_id
                        last_cut_frame = actual_frame_number
                    if curr_med_h is not None:
                        prev_med_h = curr_med_h

                # Apply monotonic global ID offset across shots
                if track_id_offset > 0:
                    for p in player_records:
                        if p.tracking_id is not None:
                            p.tracking_id = int(p.tracking_id) + track_id_offset
                for p in player_records:
                    if p.tracking_id is not None and int(p.tracking_id) > max_global_id:
                        max_global_id = int(p.tracking_id)

                # Recover any ball detections suppressed by ByteTrack's 2-frame IoU gate
                if captured_raw_boxes:
                    raw_parsed = FootballDetector.parse_ultralytics_boxes(
                        captured_raw_boxes[0],
                        frame_number=actual_frame_number,
                        timestamp=timestamp,
                        person_conf=self.det_config.person_conf,
                        ball_conf=self.det_config.ball_conf,
                    )
                    raw_balls = [r for r in raw_parsed if r.class_name == "ball"]
                    if raw_balls:
                        ball_records = _nms_ball_detections(ball_records + raw_balls)

                # Check whether full-frame produced any plausible on-pitch ball before skipping tiled search
                plausible_balls = (
                    filter_ball_candidates(
                        ball_records,
                        player_records,
                        frame_bgr.shape[:2],
                        min_size=self.det_config.min_ball_box_size_px,
                        max_size=self.det_config.max_ball_box_size_px,
                        frame_bgr=frame_bgr,
                    )
                    if self.det_config.filter_false_positive_balls and ball_records
                    else ball_records
                )

                # If no plausible ball found and tiled fallback is enabled, scan pitch tiles
                if not plausible_balls and self.det_config.enable_tiled_ball_search:
                    tiled_balls = aux_detector.detect_balls_tiled(
                        frame_bgr,
                        frame_number=actual_frame_number,
                        timestamp=timestamp,
                        conf_threshold=self.det_config.tile_conf_threshold,
                        tile_imgsz=960,
                    )
                    if tiled_balls:
                        ball_records = _nms_ball_detections(plausible_balls + tiled_balls)

                yielded += 1
                yield actual_frame_number, timestamp, frame_bgr, (player_records + ball_records)
        finally:
            cap.release()
