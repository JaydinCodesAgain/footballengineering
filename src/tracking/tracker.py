"""Tracker execution supporting ByteTrack and BoT-SORT for players (in both full-frame
and native-resolution multi-tile modes) while preserving pre-ByteTrack raw ball detections,
optional geometric pitch-region filtering, and fixed-camera motion-assisted ball recovery.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
import yaml

from src.config import DetectionConfig, TrackingConfig
from src.detection.detector import (
    COCO_BALL_CLASS_ID,
    COCO_PERSON_CLASS_ID,
    FootballDetector,
    _nms_ball_detections,
)
from src.detection.motion_ball import FixedCameraMotionBallDetector
from src.geometry.pitch_region import PitchRegionFilter
from src.possession.ball_tracker import detect_camera_cut, filter_ball_candidates
from src.schemas import DetectionRecord


SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


class _NumpyDetections:
    """Minimal Ultralytics `Boxes`-compatible NumPy container for direct BYTETracker / BOTSORT updates."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, item) -> "_NumpyDetections":
        return _NumpyDetections(self.xywh[item], self.conf[item], self.cls[item])

    @classmethod
    def from_records(cls, records: Sequence[DetectionRecord]) -> "_NumpyDetections":
        if not records:
            return cls(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        xywh_rows = []
        confs = []
        classes = []
        for r in records:
            x1, y1, x2, y2 = r.bbox
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            cx = float(x1) + w / 2.0
            cy = float(y1) + h / 2.0
            xywh_rows.append([cx, cy, w, h])
            confs.append(float(r.confidence))
            classes.append(float(r.class_id))
        return cls(
            np.asarray(xywh_rows, dtype=np.float32),
            np.asarray(confs, dtype=np.float32),
            np.asarray(classes, dtype=np.float32),
        )


def _build_ultralytics_tracker(
    tracker_config: str,
    frame_rate: int = 25,
    disable_gmc: bool = False,
):
    """Instantiate Ultralytics BYTETracker or BOTSORT directly for custom/tiled detections."""
    import inspect
    import ultralytics
    from ultralytics.trackers import BOTSORT, BYTETracker

    cfg_path = Path(tracker_config)
    if not cfg_path.is_file():
        cfg_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / tracker_config
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if disable_gmc:
        raw_cfg["gmc_method"] = None
    # Match Ultralytics defaults if absent
    raw_cfg.setdefault("fuse_score", True)
    raw_cfg.setdefault("proximity_thresh", 0.5)
    raw_cfg.setdefault("appearance_thresh", 0.25)
    raw_cfg.setdefault("with_reid", False)
    raw_cfg.setdefault("model", "auto")
    args = SimpleNamespace(**raw_cfg)

    tracker_type = str(
        raw_cfg.get("tracker_type", "botsort" if "botsort" in str(tracker_config).lower() else "bytetrack")
    ).lower()
    cls = BYTETracker if tracker_type == "bytetrack" else BOTSORT
    sig = inspect.signature(cls.__init__)
    if "frame_rate" in sig.parameters:
        return cls(args, frame_rate=max(1, int(round(frame_rate))))
    return cls(args)


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
    for players (`class 0`) while supporting both:
    - Standard broadcast full-frame mode (`model.track` + pre-ByteTrack raw ball capture), and
    - Native-resolution overlapping multi-tile mode (`FootballDetector.detect_frame_tiled`
      piped directly into Ultralytics `BYTETracker` / `BOTSORT`), with optional geometric
      fixed-camera pitch-region filtering and motion-assisted small-ball candidate recovery.
    """

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        device: Optional[str | int] = None,
        pitch_region_filter: Optional[PitchRegionFilter] = None,
        fixed_camera: bool = False,
        use_pitch_roi_for_tiling: bool = True,
        filter_players_by_pitch: bool = True,
        filter_balls_by_pitch: bool = True,
        motion_ball_detector: Optional[FixedCameraMotionBallDetector] = None,
    ) -> None:
        self.det_config = detection_config or DetectionConfig()
        self.trk_config = tracking_config or TrackingConfig()
        self.pitch_region_filter = pitch_region_filter
        self.fixed_camera = bool(fixed_camera)
        self.use_pitch_roi_for_tiling = bool(use_pitch_roi_for_tiling)
        self.filter_players_by_pitch = bool(filter_players_by_pitch)
        self.filter_balls_by_pitch = bool(filter_balls_by_pitch)
        self.motion_ball_detector = motion_ball_detector

        trk_name = str(self.trk_config.tracker_config)
        if trk_name not in SUPPORTED_TRACKERS and not Path(trk_name).is_file():
            raise ValueError(
                f"Unsupported tracker '{self.trk_config.tracker_config}'. "
                f"Supported trackers: {SUPPORTED_TRACKERS} or an existing YAML file path."
            )
        self.device: str | int = (
            device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        )

    def _track_video_tiled(
        self,
        vpath: Path,
        max_frames: Optional[int],
        fps_override: Optional[float],
    ) -> Iterator[Tuple[int, float, np.ndarray, List[DetectionRecord]]]:
        """Run native-resolution overlapping multi-tile detection -> optional pitch filtering ->
        direct Ultralytics BYTETracker / BOTSORT association -> motion-assisted ball recovery.
        """
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {vpath}")
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        fps = fps_override if (fps_override and fps_override > 0) else (raw_fps if raw_fps > 0 else 25.0)
        stride = max(1, int(self.det_config.frame_stride))
        start_frame = max(0, int(self.trk_config.start_frame))
        eff_fps = max(1, int(round(fps / float(stride))))

        detector = FootballDetector(config=self.det_config, device=self.device)
        mot_tracker = _build_ultralytics_tracker(
            self.trk_config.tracker_config,
            frame_rate=eff_fps,
            disable_gmc=self.fixed_camera,
        )

        prev_small: Optional[np.ndarray] = None
        prev_med_h: Optional[float] = None
        last_cut_frame: Optional[int] = None
        track_id_offset: int = 0
        max_global_id: int = 0
        do_cut_reset = bool(self.trk_config.reset_on_camera_cut) and not self.fixed_camera

        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_offset = 0
            yielded = 0
            cached_tiling_roi: Optional[Tuple[int, int, int, int]] = None

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

                if do_cut_reset:
                    small_curr = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
                    if prev_small is not None and detect_camera_cut(prev_small, small_curr, corr_threshold=0.52):
                        if last_cut_frame is None or (actual_frame_number - last_cut_frame) > 6:
                            mot_tracker.reset()
                            track_id_offset = max_global_id
                            last_cut_frame = actual_frame_number
                            prev_med_h = None
                    prev_small = small_curr

                if cached_tiling_roi is None and self.pitch_region_filter is not None and self.use_pitch_roi_for_tiling:
                    cached_tiling_roi = self.pitch_region_filter.get_tiling_roi_bbox(frame_bgr.shape[:2])

                raw_dets = detector.detect_frame_tiled(
                    frame_bgr,
                    frame_number=actual_frame_number,
                    timestamp=timestamp,
                    roi_bbox=cached_tiling_roi,
                )

                # Separate players and balls, applying optional geometric pitch-region filtering
                raw_players = [d for d in raw_dets if d.class_name == "player"]
                ball_records = [d for d in raw_dets if d.class_name == "ball"]

                if self.pitch_region_filter is not None:
                    if self.filter_players_by_pitch:
                        raw_players = [
                            p for p in raw_players if self.pitch_region_filter.is_player_in_region(p.bbox, include_margin=True)
                        ]
                    if self.filter_balls_by_pitch:
                        ball_records = [
                            b for b in ball_records if self.pitch_region_filter.is_ball_in_region(b.bbox, include_margin=True)
                        ]

                # Scale-jump cut check for non-fixed cameras
                if do_cut_reset and raw_players:
                    p_heights = [float(r.bbox[3] - r.bbox[1]) for r in raw_players if (r.bbox[3] - r.bbox[1]) > 15]
                    curr_med_h = float(np.median(p_heights)) if p_heights else None
                    if (
                        prev_med_h is not None
                        and curr_med_h is not None
                        and detect_camera_cut(None, None, prev_median_player_h=prev_med_h, curr_median_player_h=curr_med_h)
                        and (last_cut_frame is None or (actual_frame_number - last_cut_frame) > 6)
                    ):
                        mot_tracker.reset()
                        track_id_offset = max_global_id
                        last_cut_frame = actual_frame_number
                    if curr_med_h is not None:
                        prev_med_h = curr_med_h

                # Associate players via BYTETracker / BOTSORT (passing img=None on fixed cameras to skip GMC)
                det_adapter = _NumpyDetections.from_records(raw_players)
                tracked_arr = mot_tracker.update(det_adapter, img=None if self.fixed_camera else frame_bgr)

                player_records: List[DetectionRecord] = []
                if tracked_arr is not None and len(tracked_arr) > 0:
                    for row in tracked_arr:
                        x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
                        tid = int(round(float(row[4]))) + track_id_offset
                        conf = round(float(row[5]), 4)
                        if tid > max_global_id:
                            max_global_id = tid
                        player_records.append(
                            DetectionRecord(
                                frame_number=actual_frame_number,
                                timestamp=timestamp,
                                class_id=COCO_PERSON_CLASS_ID,
                                class_name="player",
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                tracking_id=tid,
                            )
                        )

                # Filter tiled ball detections before optionally augmenting with motion-assisted proposals
                plausible_balls = (
                    filter_ball_candidates(
                        ball_records,
                        player_records if player_records else raw_players,
                        frame_bgr.shape[:2],
                        min_size=self.det_config.min_ball_box_size_px,
                        max_size=self.det_config.max_ball_box_size_px,
                        frame_bgr=frame_bgr,
                        pitch_region_filter=self.pitch_region_filter if self.filter_balls_by_pitch else None,
                    )
                    if self.det_config.filter_false_positive_balls and ball_records
                    else ball_records
                )

                # Fixed-camera motion-assisted small-ball recovery when tiled YOLO finds no plausible ball
                if self.motion_ball_detector is not None:
                    motion_balls = self.motion_ball_detector.detect_candidates(
                        frame_bgr=frame_bgr,
                        frame_number=actual_frame_number,
                        timestamp=timestamp,
                        player_detections=player_records if player_records else raw_players,
                        yolo_detector=detector,
                    )
                    if motion_balls:
                        plausible_balls = _nms_ball_detections(plausible_balls + motion_balls)

                yielded += 1
                yield actual_frame_number, timestamp, frame_bgr, (player_records + plausible_balls)
        finally:
            cap.release()

    def track_video(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        fps_override: Optional[float] = None,
    ) -> Iterator[Tuple[int, float, np.ndarray, List[DetectionRecord]]]:
        """
        Stream tracked player detections and raw/recovered ball detections over a video.
        """
        vpath = Path(video_path)
        if not vpath.is_file():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        cap_probe = cv2.VideoCapture(str(vpath))
        if not cap_probe.isOpened():
            raise ValueError(f"Could not open video file: {vpath}")
        fw = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()

        probe_det = FootballDetector(config=self.det_config, device=self.device)
        if probe_det.should_use_tiled((fh, fw)):
            yield from self._track_video_tiled(vpath, max_frames=max_frames, fps_override=fps_override)
            return

        from ultralytics import YOLO

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
        do_cut_reset = bool(self.trk_config.reset_on_camera_cut) and not self.fixed_camera

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

                if do_cut_reset:
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

                if self.pitch_region_filter is not None and self.filter_players_by_pitch:
                    player_records = [
                        p for p in player_records if self.pitch_region_filter.is_player_in_region(p.bbox, include_margin=True)
                    ]

                if do_cut_reset and player_records:
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

                if track_id_offset > 0:
                    for p in player_records:
                        if p.tracking_id is not None:
                            p.tracking_id = int(p.tracking_id) + track_id_offset
                for p in player_records:
                    if p.tracking_id is not None and int(p.tracking_id) > max_global_id:
                        max_global_id = int(p.tracking_id)

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

                plausible_balls = (
                    filter_ball_candidates(
                        ball_records,
                        player_records,
                        frame_bgr.shape[:2],
                        min_size=self.det_config.min_ball_box_size_px,
                        max_size=self.det_config.max_ball_box_size_px,
                        frame_bgr=frame_bgr,
                        pitch_region_filter=self.pitch_region_filter if self.filter_balls_by_pitch else None,
                    )
                    if self.det_config.filter_false_positive_balls and ball_records
                    else ball_records
                )

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

                if self.motion_ball_detector is not None:
                    motion_balls = self.motion_ball_detector.detect_candidates(
                        frame_bgr=frame_bgr,
                        frame_number=actual_frame_number,
                        timestamp=timestamp,
                        player_detections=player_records,
                        yolo_detector=aux_detector,
                    )
                    if motion_balls:
                        ball_records = _nms_ball_detections(ball_records + motion_balls)

                yielded += 1
                yield actual_frame_number, timestamp, frame_bgr, (player_records + ball_records)
        finally:
            cap.release()
