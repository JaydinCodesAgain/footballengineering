"""YOLO11 wrapper supporting class-specific confidence thresholds, high-res ROI search, and tiled pitch ball detection."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch

from src.config import DetectionConfig
from src.evaluation.detection_eval import compute_iou, evaluate_detections_against_ground_truth
from src.schemas import DetectionRecord


COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32


def _nms_ball_detections(
    detections: Sequence[DetectionRecord],
    iou_threshold: float = 0.45,
) -> List[DetectionRecord]:
    """Apply Non-Maximum Suppression (NMS) across ball detections merged from tiles or ROIs."""
    if len(detections) <= 1:
        return list(detections)
    ordered = sorted(detections, key=lambda d: float(d.confidence), reverse=True)
    kept: List[DetectionRecord] = []
    for cand in ordered:
        if all(compute_iou(cand.bbox, existing.bbox) < iou_threshold for existing in kept):
            kept.append(cand)
    return kept


class FootballDetector:
    """
    Player and ball detector backed by Ultralytics YOLO11.

    Supports multiple ball detection strategies:
    1. Standard full-frame inference (`imgsz=1088` or `1280`) with decoupled `person_conf` and `ball_conf`.
    2. Dynamic Region-of-Interest (ROI) high-resolution search around predicted ball coordinates.
    3. Overlapping 2-tile horizontal pitch slicing for tiny distant broadcast footballs.
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
        imgsz_override: Optional[int] = None,
    ) -> List[DetectionRecord]:
        """Run player and ball detection on a single BGR frame."""
        p_conf = float(person_conf_override if person_conf_override is not None else self.config.person_conf)
        b_conf = float(ball_conf_override if ball_conf_override is not None else self.config.ball_conf)
        min_conf = min(p_conf, b_conf)
        imgsz = int(imgsz_override if imgsz_override is not None else self.config.imgsz)

        results = self.model.predict(
            frame_bgr,
            classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            conf=min_conf,
            imgsz=imgsz,
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

    def detect_ball_in_roi(
        self,
        frame_bgr: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        frame_number: int = 0,
        timestamp: float = 0.0,
        conf_threshold: Optional[float] = None,
    ) -> List[DetectionRecord]:
        """
        Run targeted ball detection inside a local `(rx1, ry1, rx2, ry2)` crop and map coordinates
        back to full-frame pixel space.
        """
        rx1, ry1, rx2, ry2 = roi_bbox
        fh, fw = frame_bgr.shape[:2]
        rx1 = max(0, min(fw - 1, int(rx1)))
        ry1 = max(0, min(fh - 1, int(ry1)))
        rx2 = max(rx1 + 1, min(fw, int(rx2)))
        ry2 = max(ry1 + 1, min(fh, int(ry2)))

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
            return []

        b_conf = float(conf_threshold if conf_threshold is not None else self.config.roi_conf_threshold)
        roi_imgsz = max(320, min(960, int(math.ceil(max(crop.shape[:2]) / 32.0) * 32))) if "math" in globals() else 640

        results = self.model.predict(
            crop,
            classes=[COCO_BALL_CLASS_ID],
            conf=b_conf,
            imgsz=roi_imgsz,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        mapped: List[DetectionRecord] = []
        for box in results[0].boxes:
            if int(box.cls.item()) != COCO_BALL_CLASS_ID:
                continue
            conf = float(box.conf.item()) if box.conf is not None else b_conf
            if conf < b_conf:
                continue
            coords = box.xyxy[0].cpu().tolist()
            bx1 = int(round(float(coords[0]))) + rx1
            by1 = int(round(float(coords[1]))) + ry1
            bx2 = int(round(float(coords[2]))) + rx1
            by2 = int(round(float(coords[3]))) + ry1
            mapped.append(
                DetectionRecord(
                    frame_number=int(frame_number),
                    timestamp=round(float(timestamp), 3),
                    class_id=COCO_BALL_CLASS_ID,
                    class_name="ball",
                    bbox=(bx1, by1, bx2, by2),
                    confidence=round(conf, 4),
                    tracking_id=None,
                )
            )
        return _nms_ball_detections(mapped)

    def detect_balls_tiled(
        self,
        frame_bgr: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        conf_threshold: Optional[float] = None,
        tile_imgsz: int = 960,
    ) -> List[DetectionRecord]:
        """
        Run 2-tile overlapping horizontal pitch inference for tiny distant footballs and merge via NMS.
        """
        fh, fw = frame_bgr.shape[:2]
        y0 = int(fh * 0.10)
        y1 = int(fh * 0.96)
        tiles = [
            (0, y0, int(fw * 0.58), y1),
            (int(fw * 0.42), y0, fw, y1),
        ]
        b_conf = float(conf_threshold if conf_threshold is not None else self.config.tile_conf_threshold)

        all_balls: List[DetectionRecord] = []
        for tx1, ty1, tx2, ty2 in tiles:
            crop = frame_bgr[ty1:ty2, tx1:tx2]
            if crop.size == 0:
                continue
            results = self.model.predict(
                crop,
                classes=[COCO_BALL_CLASS_ID],
                conf=b_conf,
                imgsz=tile_imgsz,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                continue
            for box in results[0].boxes:
                if int(box.cls.item()) != COCO_BALL_CLASS_ID:
                    continue
                conf = float(box.conf.item()) if box.conf is not None else b_conf
                if conf < b_conf:
                    continue
                coords = box.xyxy[0].cpu().tolist()
                bx1 = int(round(float(coords[0]))) + tx1
                by1 = int(round(float(coords[1]))) + ty1
                bx2 = int(round(float(coords[2]))) + tx1
                by2 = int(round(float(coords[3]))) + ty1
                all_balls.append(
                    DetectionRecord(
                        frame_number=int(frame_number),
                        timestamp=round(float(timestamp), 3),
                        class_id=COCO_BALL_CLASS_ID,
                        class_name="ball",
                        bbox=(bx1, by1, bx2, by2),
                        confidence=round(conf, 4),
                        tracking_id=None,
                    )
                )

        return _nms_ball_detections(all_balls)

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


def benchmark_ball_detection_strategies(
    frames_with_ids: Sequence[Tuple[int, float, np.ndarray]],
    verified_annotations: Optional[Sequence[Any]] = None,
    iou_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Benchmark the 6 football detection strategies under identical evaluation conditions:
    1. `yolo11n_baseline_c040` (YOLO11n baseline: imgsz=1088, ball_conf=0.40)
    2. `yolo11n_lowconf_filtered_c015` (Low-confidence full-frame: imgsz=1088, ball_conf=0.15 + FP filter)
    3. `yolo11n_highres_1280_c018` (High-resolution detection: imgsz=1280, ball_conf=0.18 + FP filter)
    4. `yolo11n_tiled_pitch_960_c018` (Tiled inference: 2x horizontal pitch tiles @ 960, ball_conf=0.18 + NMS)
    5. `yolo11n_hybrid_1088_plus_tiled_fallback` (Hybrid full-frame 1088 + tiled fallback on missed frames)
    6. `yolo11n_kalman_roi_recovery` (Full-frame 1088 + Kalman-guided 640x640 ROI recovery)

    Records inference latency (ms/frame), ball detection coverage (explicitly distinguished from
    recall), and (when verified manual bounding-box annotations exist) Precision, Recall, F1,
    AP@50, and mAP@50:95.
    """
    from src.possession.ball_tracker import KalmanBallTracker, filter_ball_candidates

    detector = FootballDetector(DetectionConfig(model_name="yolo11n.pt", imgsz=1088))
    n_frames = max(1, len(frames_with_ids))
    verified = [
        a
        for a in (verified_annotations or [])
        if getattr(a, "verified", False) and getattr(a, "source", "") == "manual"
    ]

    strategies = [
        ("yolo11n_baseline_c040", 1088, 0.40, False, "full_only"),
        ("yolo11n_lowconf_filtered_c015", 1088, 0.15, True, "full_only"),
        ("yolo11n_highres_1280_c018", 1280, 0.18, True, "full_only"),
        ("yolo11n_tiled_pitch_960_c018", 960, 0.18, True, "tiled_only"),
        ("yolo11n_hybrid_1088_plus_tiled_fallback", 1088, 0.18, True, "hybrid_tiled"),
        ("yolo11n_kalman_roi_recovery", 1088, 0.18, True, "kalman_roi"),
    ]

    results_table: List[Dict[str, Any]] = []

    for name, imgsz, b_conf, use_fp_filter, mode in strategies:
        t0 = time.perf_counter()
        preds_by_frame: Dict[int, List[DetectionRecord]] = {}
        frames_with_ball = 0
        total_balls = 0
        kalman = KalmanBallTracker("bench") if mode == "kalman_roi" else None

        for fno, ts, frm in frames_with_ids:
            fh, fw = frm.shape[:2]
            if mode == "tiled_only":
                raw_dets = detector.detect_frame(
                    frm, fno, ts, person_conf_override=0.40, ball_conf_override=0.99, imgsz_override=1088
                )
                players = [d for d in raw_dets if d.class_name == "player"]
                balls = detector.detect_balls_tiled(frm, fno, ts, conf_threshold=b_conf, tile_imgsz=imgsz)
            else:
                raw_dets = detector.detect_frame(
                    frm, fno, ts, person_conf_override=0.40, ball_conf_override=b_conf, imgsz_override=imgsz
                )
                players = [d for d in raw_dets if d.class_name == "player"]
                balls = [d for d in raw_dets if d.class_name == "ball"]
                if mode == "hybrid_tiled" and not balls:
                    balls = detector.detect_balls_tiled(frm, fno, ts, conf_threshold=0.16, tile_imgsz=960)
                elif mode == "kalman_roi" and kalman is not None and not balls:
                    roi = kalman.get_predicted_search_roi((fh, fw), roi_size=640)
                    if roi is None and players:
                        # Fallback to player centroid ROI when tracker is not yet initialized
                        cx_p = int(np.mean([(p.bbox[0] + p.bbox[2]) / 2.0 for p in players]))
                        cy_p = int(np.mean([p.bbox[3] for p in players]))
                        x0 = max(0, min(fw - 640, cx_p - 320))
                        y0 = max(0, min(fh - 640, cy_p - 320))
                        roi = (x0, y0, min(fw, x0 + 640), min(fh, y0 + 640))
                    if roi is not None:
                        balls = detector.detect_ball_in_roi(frm, roi, fno, ts, conf_threshold=0.14)

            if use_fp_filter:
                balls = filter_ball_candidates(balls, players, (fh, fw))

            if kalman is not None:
                kalman.step(fno, ts, balls, frame_bgr=frm, player_detections=players)

            if balls:
                frames_with_ball += 1
                total_balls += len(balls)
            preds_by_frame[fno] = players + balls

        elapsed_s = time.perf_counter() - t0
        ms_per_frame = (elapsed_s / float(n_frames)) * 1000.0
        coverage_pct = (frames_with_ball / float(n_frames)) * 100.0

        row: Dict[str, Any] = {
            "strategy": name,
            "imgsz": imgsz,
            "ball_conf": b_conf,
            "fp_filter": use_fp_filter,
            "frames_evaluated": len(frames_with_ids),
            "frames_with_ball_detected": frames_with_ball,
            "total_ball_detections": total_balls,
            "ball_detection_coverage_pct": round(coverage_pct, 2),
            "inference_ms_per_frame": round(ms_per_frame, 1),
            "coverage_vs_recall_note": "Coverage measures % of frames with >=1 detection; Recall requires verified GT boxes.",
        }

        if verified:
            iou_res = evaluate_detections_against_ground_truth(preds_by_frame, verified, iou_threshold=iou_threshold)
            b_m = iou_res["ball"]
            row["ball_precision"] = b_m["precision"]
            row["ball_recall"] = b_m["recall"]
            row["ball_f1"] = b_m["f1"]
            row["ball_ap50"] = b_m["ap50"]
            row["ball_map50_95"] = b_m.get("map50_95")
        else:
            row["ball_precision"] = None
            row["ball_recall"] = None
            row["ball_f1"] = None
            row["ball_ap50"] = None
            row["ball_map50_95"] = None

        results_table.append(row)

    return results_table
