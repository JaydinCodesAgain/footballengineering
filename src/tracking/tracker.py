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
from src.schemas import DetectionRecord


SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


class MultiObjectTracker:
    """
    Wraps Ultralytics multi-object tracking (`bytetrack.yaml` or `botsort.yaml`) for players (`class 0`)
    while capturing raw post-NMS football detections (`class 32`) BEFORE ByteTrack's two-frame IoU
    activation filter (`is_activated`) discards fast-moving balls.
    """

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        device: Optional[str | int] = None,
    ) -> None:
        self.det_config = detection_config or DetectionConfig()
        self.trk_config = tracking_config or TrackingConfig()
        if self.trk_config.tracker_config not in SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unsupported tracker '{self.trk_config.tracker_config}'. "
                f"Supported trackers: {SUPPORTED_TRACKERS}"
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

        # Hook executed on `on_predict_postprocess_end` BEFORE `track.on_predict_postprocess_end`
        # overwrites `predictor.results[0].boxes` and drops unactivated fast-moving ball boxes.
        captured_raw_boxes: List[object] = []

        def _capture_raw_postprocess_boxes(predictor: object) -> None:
            captured_raw_boxes.clear()
            results_list = getattr(predictor, "results", None)
            if results_list and len(results_list) > 0 and results_list[0].boxes is not None:
                captured_raw_boxes.append(results_list[0].boxes.cpu())

        model.add_callback("on_predict_postprocess_end", _capture_raw_postprocess_boxes)

        aux_detector = FootballDetector(config=self.det_config, device=self.device)

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

                # Recover any ball detections that were suppressed by ByteTrack's 2-frame IoU gate
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

                # If still no ball found and tiled ball search fallback is enabled, scan pitch tiles
                if not ball_records and self.det_config.enable_tiled_ball_search:
                    tiled_balls = aux_detector.detect_balls_tiled(
                        frame_bgr,
                        frame_number=actual_frame_number,
                        timestamp=timestamp,
                        conf_threshold=self.det_config.tile_conf_threshold,
                        tile_imgsz=960,
                    )
                    if tiled_balls:
                        ball_records = tiled_balls

                yielded += 1
                yield actual_frame_number, timestamp, frame_bgr, (player_records + ball_records)
        finally:
            cap.release()
