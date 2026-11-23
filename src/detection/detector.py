"""YOLO11 wrapper supporting class-specific confidence thresholds, high-res ROI search, and tiled pitch ball detection."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch

from src.config import DetectionConfig
from src.detection.tiling import (
    TileSpec,
    estimate_active_content_bbox,
    extract_tile_views,
    generate_tile_grid,
    merge_tiled_detections,
)
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
        half: bool = False,
    ) -> None:
        from src.config import resolve_compute_device

        self.config = config or DetectionConfig()
        req_dev = "auto" if device is None else str(device)
        resolved_dev, resolved_half, fell_back = resolve_compute_device(req_dev, bool(half))
        self.device: str | int = resolved_dev
        self.half: bool = resolved_half
        self.fell_back_to_cpu: bool = fell_back
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.config.model_name)
        return self._model

    def should_use_tiled(self, frame_shape: Tuple[int, int]) -> bool:
        """Return True if multi-tile detection should be used for `frame_shape`."""
        mode = str(self.config.inference_mode).strip().lower()
        if mode == "tiled":
            return True
        if mode == "auto":
            return int(frame_shape[1]) >= int(self.config.auto_tile_min_width_px)
        return False

    def detect_frame_auto(
        self,
        frame_bgr: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        person_conf_override: Optional[float] = None,
        ball_conf_override: Optional[float] = None,
    ) -> List[DetectionRecord]:
        """Dispatch to native-resolution multi-tile or standard full-frame detection based on config."""
        if self.should_use_tiled(frame_bgr.shape[:2]):
            return self.detect_frame_tiled(
                frame_bgr,
                frame_number=frame_number,
                timestamp=timestamp,
                roi_bbox=roi_bbox,
                person_conf_override=person_conf_override,
                ball_conf_override=ball_conf_override,
            )
        return self.detect_frame(
            frame_bgr,
            frame_number=frame_number,
            timestamp=timestamp,
            person_conf_override=person_conf_override,
            ball_conf_override=ball_conf_override,
        )

    def detect_frame_tiled(
        self,
        frame_bgr: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_overlap: Optional[float] = None,
        tile_imgsz: Optional[int] = None,
        person_conf_override: Optional[float] = None,
        ball_conf_override: Optional[float] = None,
        include_full_frame: Optional[bool] = None,
    ) -> List[DetectionRecord]:
        """Run overlapping native-resolution multi-tile detection for both players and footballs.

        Slices zero-copy NumPy views from `frame_bgr`, runs batched YOLO inference per tile,
        maps detections back into raw panoramic pixel coordinates, and merges overlaps with
        seam-aware class-specific NMS + containment deduplication.
        """
        fh, fw = frame_bgr.shape[:2]
        p_conf = float(person_conf_override if person_conf_override is not None else self.config.person_conf)
        b_conf = float(ball_conf_override if ball_conf_override is not None else self.config.ball_conf)
        min_conf = min(p_conf, b_conf)

        active_roi = roi_bbox
        if active_roi is None and self.config.tile_auto_content_crop:
            active_roi = estimate_active_content_bbox(frame_bgr)

        tw = int(tile_width if tile_width is not None else self.config.tile_width)
        th = int(tile_height if tile_height is not None else self.config.tile_height)
        ov = float(tile_overlap if tile_overlap is not None else self.config.tile_overlap)
        imgsz = int(tile_imgsz if tile_imgsz is not None else self.config.tile_imgsz)

        tiles = generate_tile_grid(
            (fh, fw),
            tile_width=tw,
            tile_height=th,
            overlap=ov,
            roi_bbox=active_roi,
        )
        if not tiles:
            return []

        views = extract_tile_views(frame_bgr, tiles)
        batch_sz = max(1, int(self.config.tile_batch_size))
        mapped: List[Tuple[DetectionRecord, Optional[TileSpec]]] = []

        for start in range(0, len(tiles), batch_sz):
            batch_tiles = tiles[start : start + batch_sz]
            batch_views = views[start : start + batch_sz]
            pred_kwargs: Dict[str, Any] = {
                "classes": [COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
                "conf": min_conf,
                "imgsz": imgsz,
                "device": self.device,
                "verbose": False,
            }
            if self.half:
                pred_kwargs["half"] = True
            results = self.model.predict(batch_views, **pred_kwargs)
            if not results:
                continue
            for tile, res in zip(batch_tiles, results):
                local_recs = self.parse_ultralytics_boxes(
                    res.boxes,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    person_conf=p_conf,
                    ball_conf=b_conf,
                )
                for rec in local_recs:
                    gx1 = rec.bbox[0] + tile.x1
                    gy1 = rec.bbox[1] + tile.y1
                    gx2 = rec.bbox[2] + tile.x1
                    gy2 = rec.bbox[3] + tile.y1
                    mapped.append(
                        (
                            DetectionRecord(
                                frame_number=rec.frame_number,
                                timestamp=rec.timestamp,
                                class_id=rec.class_id,
                                class_name=rec.class_name,
                                bbox=(gx1, gy1, gx2, gy2),
                                confidence=rec.confidence,
                                tracking_id=None,
                            ),
                            tile,
                        )
                    )

        use_global = (
            bool(include_full_frame)
            if include_full_frame is not None
            else bool(self.config.include_full_frame_in_tiled)
        )
        if use_global:
            global_recs = self.detect_frame(
                frame_bgr,
                frame_number=frame_number,
                timestamp=timestamp,
                person_conf_override=p_conf,
                ball_conf_override=b_conf,
            )
            for rec in global_recs:
                mapped.append((rec, None))

        return merge_tiled_detections(
            mapped,
            frame_shape=(fh, fw),
            player_nms_iou=float(self.config.player_nms_iou),
            player_containment_iom=float(self.config.player_containment_iom),
            ball_nms_iou=float(self.config.ball_nms_iou),
        )

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

        pred_kwargs: Dict[str, Any] = {
            "classes": [COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            "conf": min_conf,
            "imgsz": imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            pred_kwargs["half"] = True
        results = self.model.predict(frame_bgr, **pred_kwargs)
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
        upscale_factor: float = 1.0,
    ) -> List[DetectionRecord]:
        """
        Run targeted ball detection inside a local `(rx1, ry1, rx2, ry2)` crop and map coordinates
        back to full-frame pixel space. Optional `upscale_factor > 1.0` enlarges tiny 9-12 px
        panoramic footballs into COCO YOLO's effective scale range before inference.
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

        scale = max(1.0, float(upscale_factor))
        if scale > 1.01:
            infer_crop = cv2.resize(
                crop,
                (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            infer_crop = crop

        b_conf = float(conf_threshold if conf_threshold is not None else self.config.roi_conf_threshold)
        roi_imgsz = max(320, min(1024, int(math.ceil(max(infer_crop.shape[:2]) / 32.0) * 32)))

        pred_kwargs: Dict[str, Any] = {
            "classes": [COCO_BALL_CLASS_ID],
            "conf": b_conf,
            "imgsz": roi_imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.half:
            pred_kwargs["half"] = True
        results = self.model.predict(infer_crop, **pred_kwargs)
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
            bx1 = int(round(float(coords[0]) / scale)) + rx1
            by1 = int(round(float(coords[1]) / scale)) + ry1
            bx2 = int(round(float(coords[2]) / scale)) + rx1
            by2 = int(round(float(coords[3]) / scale)) + ry1
            if bx2 <= bx1 or by2 <= by1:
                continue
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
    model_name: str = "yolo11n.pt",
) -> List[Dict[str, Any]]:
    """
    Benchmark the 6 football detection strategies under identical evaluation conditions:
    1. `yolo11_baseline_c040` (Baseline: imgsz=1088, ball_conf=0.40)
    2. `yolo11_lowconf_filtered_c015` (Low-confidence full-frame: imgsz=1088, ball_conf=0.15 + FP filter)
    3. `yolo11_highres_1280_c018` (High-resolution detection: imgsz=1280, ball_conf=0.18 + FP filter)
    4. `yolo11_tiled_pitch_960_c018` (Tiled inference: 2x horizontal pitch tiles @ 960, ball_conf=0.18 + NMS)
    5. `yolo11_hybrid_1088_plus_tiled_fallback` (Hybrid full-frame 1088 + tiled fallback on missed frames)
    6. `yolo11_kalman_roi_recovery` (Full-frame 1088 + Kalman-guided 640x640 ROI recovery)

    Records inference latency (ms/frame), ball detection coverage (explicitly distinguished from
    recall), and (when verified manual bounding-box annotations exist) Precision, Recall, F1,
    AP@50, mAP@50:95, and centre localisation error.
    """
    from src.possession.ball_tracker import KalmanBallTracker, filter_ball_candidates

    detector = FootballDetector(DetectionConfig(model_name=model_name, imgsz=1088))
    model_tag = model_name.replace(".pt", "")
    n_frames = max(1, len(frames_with_ids))
    verified = [
        a
        for a in (verified_annotations or [])
        if getattr(a, "verified", False) and getattr(a, "source", "") == "manual"
    ]

    strategies = [
        (f"{model_tag}_baseline_c040", 1088, 0.40, False, "full_only"),
        (f"{model_tag}_lowconf_filtered_c015", 1088, 0.15, True, "full_only"),
        (f"{model_tag}_highres_1280_c018", 1280, 0.18, True, "full_only"),
        (f"{model_tag}_tiled_pitch_960_c018", 960, 0.18, True, "tiled_only"),
        (f"{model_tag}_hybrid_1088_plus_tiled_fallback", 1088, 0.18, True, "hybrid_tiled"),
        (f"{model_tag}_kalman_roi_recovery", 1088, 0.18, True, "kalman_roi"),
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
                if use_fp_filter and balls:
                    balls = filter_ball_candidates(
                        balls,
                        players,
                        (fh, fw),
                        frame_bgr=frm,
                        predicted_pos=kalman.get_predicted_position() if kalman else None,
                    )
                if mode == "hybrid_tiled" and not balls:
                    balls = detector.detect_balls_tiled(frm, fno, ts, conf_threshold=0.16, tile_imgsz=960)
                elif mode == "kalman_roi" and kalman is not None and not balls:
                    roi = kalman.get_predicted_search_roi((fh, fw), roi_size=640)
                    if roi is None and players:
                        cx_p = int(np.mean([(p.bbox[0] + p.bbox[2]) / 2.0 for p in players]))
                        cy_p = int(np.mean([p.bbox[3] for p in players]))
                        x0 = max(0, min(fw - 640, cx_p - 320))
                        y0 = max(0, min(fh - 640, cy_p - 320))
                        roi = (x0, y0, min(fw, x0 + 640), min(fh, y0 + 640))
                    if roi is not None:
                        balls = detector.detect_ball_in_roi(frm, roi, fno, ts, conf_threshold=0.14)

            if use_fp_filter:
                balls = filter_ball_candidates(
                    balls,
                    players,
                    (fh, fw),
                    frame_bgr=frm,
                    predicted_pos=kalman.get_predicted_position() if kalman else None,
                )

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
            "model_name": model_name,
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
            row["ball_center_error_px"] = b_m.get("mean_center_error_px")
        else:
            row["ball_precision"] = None
            row["ball_recall"] = None
            row["ball_f1"] = None
            row["ball_ap50"] = None
            row["ball_map50_95"] = None
            row["ball_center_error_px"] = None

        results_table.append(row)

    return results_table
