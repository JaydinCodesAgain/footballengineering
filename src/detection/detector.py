"""YOLO11 wrapper supporting class-specific confidence thresholds for players and footballs."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import torch

from src.config import DetectionConfig
from src.schemas import DetectionRecord


COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32


class FootballDetector:
    """
    Player and ball detector backed by Ultralytics YOLO11.
    Supports distinct confidence thresholds for players (`person_conf`) and balls (`ball_conf`)
    to maximize small-ball recall without inflating false-positive person detections.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        device: Optional[str | int] = None,
    ) -> None:
        self.config = config or DetectionConfig()
        if device is None:
            self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_name)
        return self._model

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        person_conf_override: Optional[float] = None,
        ball_conf_override: Optional[float] = None,
    ) -> List[DetectionRecord]:
        """Run player and ball detection on a single BGR frame."""
        p_conf = float(person_conf_override if person_conf_override is not None else self.config.person_conf)
        b_conf = float(ball_conf_override if ball_conf_override is not None else self.config.ball_conf)
        min_conf = min(p_conf, b_conf)

        results = self.model.predict(
            frame_bgr,
            classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            conf=min_conf,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        return self.parse_ultralytics_boxes(
            results[0].boxes,
            frame_number=frame_number,
            timestamp=timestamp,
            person_conf=p_conf,
            ball_conf=b_conf,
        )

    @staticmethod
    def parse_ultralytics_boxes(
        boxes,
        frame_number: int,
        timestamp: float,
        person_conf: float,
        ball_conf: float,
    ) -> List[DetectionRecord]:
        """Convert Ultralytics Boxes object into filtered `DetectionRecord` instances."""
        records: List[DetectionRecord] = []
        if boxes is None or len(boxes) == 0:
            return records

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item()) if box.conf is not None else 1.0
            if cls_id == COCO_PERSON_CLASS_ID:
                if conf < person_conf:
                    continue
                cls_name = "player"
            elif cls_id == COCO_BALL_CLASS_ID:
                if conf < ball_conf:
                    continue
                cls_name = "ball"
            else:
                continue

            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [int(round(float(c))) for c in coords[:4]]
            track_id = int(box.id.item()) if getattr(box, "id", None) is not None and box.id is not None else None

            records.append(
                DetectionRecord(
                    frame_number=int(frame_number),
                    timestamp=round(float(timestamp), 3),
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=round(conf, 4),
                    tracking_id=track_id,
                )
            )
        return records

