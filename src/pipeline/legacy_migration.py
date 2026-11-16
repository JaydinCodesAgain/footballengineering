"""Migration and auditing of legacy index-based crop sets from `footballengineering (2).py`."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch

from src.classification.cropper import (
    compute_blur_score,
    compute_grass_mask,
    compute_occlusion_ratio,
    extract_legacy_jersey_crop,
)
from src.classification.dataset import JerseyCropDataset
from src.config import MatchConfig
from src.schemas import CropQualityMetrics


# ---------------------------------------------------------------------------
# 1. Historical Uncorrected Index Dictionaries from `footballengineering (2).py`
#    (Preserved to reproduce and diagnose the 10.00%–16.67% baseline failure)
# ---------------------------------------------------------------------------
LEGACY_CHELSEA_SUNDERLAND_LABELS: Dict[str, List[int]] = {
    "Chelsea": [
        7, 8, 11, 12, 16, 18, 20, 21, 29, 31, 34,
        45, 46, 52, 53, 54, 56, 57, 60, 63, 65, 68, 71,
    ],
    "Sunderland": [
        0, 1, 2, 4, 10, 13, 22, 24, 25, 30, 32, 33, 36, 37, 38,
        40, 41, 47, 51, 58, 59, 61, 62, 66, 67, 69, 73, 74, 75, 78, 79,
    ],
    "REF": [14, 17, 26, 28, 39, 44, 48, 55, 72],
    "Other": [19, 27, 35, 50, 64],
}

LEGACY_NORWAY_WALES_TRAIN_LABELS: Dict[str, List[int]] = {
    "Norway": [1, 2, 3, 5, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 23],
    "Wales": [0, 4, 6, 7],
    "REF": [19, 24, 26, 28, 29],
}

LEGACY_NORWAY_WALES_EVAL_LABELS: Dict[str, List[int]] = {
    "Norway": [1, 2, 8, 9, 13, 18, 22, 23, 27, 29],
    "Wales": [0, 4, 7, 10, 11, 15, 16, 19, 20, 21, 24, 25, 26, 28],
    "REF": [5, 12, 14, 17],
    "Unknown": [3, 6],
}


# ---------------------------------------------------------------------------
# 2. Visually Audited & Corrected Ground-Truth Labels
#    - Norway (Team_A): Red shirt with dark-blue/white Nordic cross
#    - Wales (Team_B): Yellow away shirt
#    - REF (Referee): Cyan / light-blue UEFA official shirt
#    - Unknown: Pre-match black anthem jackets, goalkeepers, or pitch/blur
# ---------------------------------------------------------------------------
AUDITED_NORWAY_WALES_TRAIN_LABELS: Dict[str, List[int]] = {
    "Norway": [
        30, 42, 48, 49, 50, 52, 53, 54, 56, 59, 62, 67,
        69, 70, 72, 76, 77, 79, 81, 82, 83, 85, 87,
    ],
    "Wales": [
        10, 17, 19, 24, 26, 29, 34, 37, 45, 55, 58, 60,
        63, 64, 65, 66, 74, 84, 86, 88, 89,
    ],
    "REF": [
        4, 6, 7, 28, 31, 33, 35, 36, 41, 57, 68, 71,
    ],
    "Unknown": [
        1, 2, 3, 5, 12, 15, 18, 20, 21, 23, 32, 44, 46,
    ],
}

AUDITED_NORWAY_WALES_VAL_LABELS: Dict[str, List[int]] = {
    "Norway": [2, 9, 10, 12, 14, 15, 16, 18, 21, 23, 27, 28, 29],
    "Wales": [0, 8, 13, 17, 22, 24, 26],
    "REF": [4, 11, 19, 20],
    "Unknown": [1, 3, 5, 6, 7, 25],
}

AUDITED_NORWAY_WALES_TEST_LABELS: Dict[str, List[int]] = {
    "Norway": [32, 34, 35, 37, 52, 54, 55, 56, 57, 60, 69, 71, 73, 75],
    "Wales": [
        33, 36, 38, 40, 42, 43, 47, 48, 49, 51, 58, 61,
        62, 64, 65, 66, 67, 68, 74, 76, 77,
    ],
    "REF": [46, 59, 63, 70, 72],
    "Unknown": [39, 41, 44, 45, 50, 53, 78, 79],
}

AUDITED_CHELSEA_SUNDERLAND_LABELS: Dict[str, List[int]] = {
    "Chelsea": [
        3, 7, 10, 11, 13, 18, 21, 22, 24, 28, 29, 30, 31,
        33, 37, 40, 45, 52, 54, 58, 60, 61, 63, 65, 68, 71,
    ],
    "Sunderland": [
        0, 1, 2, 4, 5, 8, 12, 15, 16, 17, 19, 20, 25, 26, 27,
        32, 34, 35, 36, 38, 41, 44, 46, 51, 53, 56, 57, 59,
        62, 66, 67, 69, 73, 74, 75, 76, 77, 78, 79,
    ],
    "REF": [39, 48, 55],
    "Other": [6, 9, 14, 23, 42, 43, 47, 49, 50, 64, 70, 72],
}

LEGACY_NORWAY_WALES_DETECTION_COUNTS: Dict[int, Dict[str, int]] = {
    1624: {"person_tp": 11, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
    1744: {"person_tp": 4, "person_fn": 0, "ball_tp": 1, "ball_fn": 0},
    1865: {"person_tp": 6, "person_fn": 0, "ball_tp": 1, "ball_fn": 0},
    1986: {"person_tp": 16, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
    2106: {"person_tp": 6, "person_fn": 0, "ball_tp": 1, "ball_fn": 0},
    2227: {"person_tp": 7, "person_fn": 0, "ball_tp": 0, "ball_fn": 1},
    2348: {"person_tp": 10, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
    2469: {"person_tp": 10, "person_fn": 0, "ball_tp": 1, "ball_fn": 0},
    2589: {"person_tp": 2, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
    2710: {"person_tp": 2, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
    2831: {"person_tp": 3, "person_fn": 0, "ball_tp": 0, "ball_fn": 0},
}


def _compute_crop_quality(
    crop_bgr: np.ndarray,
    torso_bbox: Tuple[int, int, int, int],
    other_bboxes: List[Tuple[int, int, int, int]],
    target_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> CropQualityMetrics:
    h, w = crop_bgr.shape[:2]
    blur = compute_blur_score(crop_bgr)
    _, grass_ratio = compute_grass_mask(crop_bgr)
    occ_ratio = compute_occlusion_ratio(torso_bbox, other_bboxes, target_bbox)
    fh, fw = frame_shape
    x1, y1, x2, y2 = target_bbox
    edge = x1 <= 1 or y1 <= 1 or x2 >= fw - 1 or y2 >= fh - 1
    return CropQualityMetrics(
        width=w,
        height=h,
        blur_score=blur,
        grass_ratio=grass_ratio,
        occlusion_ratio=occ_ratio,
        edge_clipped=edge,
        is_valid=True,
        rejection_reason=None,
    )


def migrate_norway_wales_legacy_dataset(
    video_path: str | Path,
    dataset_dir: str | Path,
    match_config: MatchConfig,
    force_rebuild: bool = False,
    use_audited_labels: bool = True,
) -> JerseyCropDataset:
    """
    Extract the Norway-Wales crop candidates from `footballengineering (2).py` (Lines 1046-1103)
    and persist every crop with a deterministic ID, frame number, timestamp, bounding box,
    quality metrics, and split (`train`, `val`, `test`).
    """
    from ultralytics import YOLO

    ds = JerseyCropDataset(dataset_dir, match_config)
    if len(ds.records) > 0 and not force_rebuild:
        return ds

    ds.records.clear()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open Norway vs Wales video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    cap.release()

    train_end = duration * 0.40
    eval_start = duration * 0.55
    stride = 2
    rng = np.random.default_rng(42)
    device = 0 if torch.cuda.is_available() else "cpu"

    detector = YOLO(match_config.detection.model_name)
    results = detector.predict(
        source=str(video_path),
        stream=True,
        classes=[0],
        conf=0.40,
        imgsz=1080,
        vid_stride=stride,
        device=device,
        verbose=False,
    )

    train_items: List[Tuple[np.ndarray, int, float, Tuple[int, int, int, int], Tuple[int, int, int, int], CropQualityMetrics]] = []
    eval_items: List[Tuple[np.ndarray, int, float, Tuple[int, int, int, int], Tuple[int, int, int, int], CropQualityMetrics]] = []

    for i, result in enumerate(results):
        seconds = i * stride / fps
        if i % 8 != 0:
            continue
        if result.boxes is None:
            continue
        if not (seconds < train_end or seconds >= eval_start):
            continue

        boxes = result.boxes
        chosen = rng.permutation(len(boxes))[:3]
        all_bboxes = [
            tuple(int(c) for c in b.xyxy[0].cpu().tolist()[:4]) for b in boxes
        ]
        fh, fw = result.orig_img.shape[:2]

        for j in chosen:
            box = boxes[int(j)]
            coords = [int(c) for c in box.xyxy[0].cpu().tolist()[:4]]
            x1, y1, x2, y2 = coords
            crop = extract_legacy_jersey_crop(result.orig_img, coords, mode="norway_wales")
            if crop is None:
                continue

            pw, ph = x2 - x1, y2 - y1
            torso_box = (
                max(0, int(x1 + pw * 0.25)),
                max(0, int(y1 + ph * 0.18)),
                min(fw, int(x1 + pw * 0.75)),
                min(fh, int(y1 + ph * 0.55)),
            )
            q_metrics = _compute_crop_quality(
                crop, torso_box, all_bboxes, (x1, y1, x2, y2), (fh, fw)
            )
            actual_frame = i * stride

            if seconds < train_end and len(train_items) < 100:
                train_items.append((crop, actual_frame, seconds, (x1, y1, x2, y2), torso_box, q_metrics))
            elif seconds >= eval_start and len(eval_items) < 80:
                eval_items.append((crop, actual_frame, seconds, (x1, y1, x2, y2), torso_box, q_metrics))

    train_source = AUDITED_NORWAY_WALES_TRAIN_LABELS if use_audited_labels else LEGACY_NORWAY_WALES_TRAIN_LABELS
    val_source = AUDITED_NORWAY_WALES_VAL_LABELS if use_audited_labels else LEGACY_NORWAY_WALES_EVAL_LABELS
    test_source = AUDITED_NORWAY_WALES_TEST_LABELS if use_audited_labels else {}

    train_idx_to_label: Dict[int, str] = {}
    for team_name, idxs in train_source.items():
        for idx in idxs:
            train_idx_to_label[idx] = team_name

    val_idx_to_label: Dict[int, str] = {}
    for team_name, idxs in val_source.items():
        for idx in idxs:
            val_idx_to_label[idx] = team_name

    test_idx_to_label: Dict[int, str] = {}
    for team_name, idxs in test_source.items():
        for idx in idxs:
            test_idx_to_label[idx] = team_name

    for idx, (crop, fno, sec, bbox, tbox, qm) in enumerate(train_items):
        raw_lbl = train_idx_to_label.get(idx)
        cid = f"{match_config.match_id}_train_g{idx:03d}_f{fno:06d}_x{bbox[0]:04d}_y{bbox[1]:04d}"
        ds.add_crop(
            crop_bgr=crop,
            frame_number=fno,
            timestamp=sec,
            bbox=bbox,
            torso_bbox=tbox,
            quality=qm,
            ground_truth_label=raw_lbl,
            raw_label=raw_lbl,
            label_verified=raw_lbl is not None,
            split="train",
            legacy_gallery_index=idx,
            crop_id_override=cid,
        )

    for idx, (crop, fno, sec, bbox, tbox, qm) in enumerate(eval_items):
        if idx < 30:
            raw_lbl = val_idx_to_label.get(idx)
            split_name = "val"
        else:
            raw_lbl = test_idx_to_label.get(idx)
            split_name = "test" if idx >= 32 else "unassigned"

        cid = f"{match_config.match_id}_{split_name}_g{idx:03d}_f{fno:06d}_x{bbox[0]:04d}_y{bbox[1]:04d}"
        ds.add_crop(
            crop_bgr=crop,
            frame_number=fno,
            timestamp=sec,
            bbox=bbox,
            torso_bbox=tbox,
            quality=qm,
            ground_truth_label=raw_lbl,
            raw_label=raw_lbl,
            label_verified=raw_lbl is not None,
            split=split_name,
            legacy_gallery_index=idx,
            crop_id_override=cid,
        )

    ds.save()
    return ds


def migrate_chelsea_sunderland_legacy_dataset(
    video_path: str | Path,
    dataset_dir: str | Path,
    match_config: MatchConfig,
    force_rebuild: bool = False,
    use_audited_labels: bool = True,
) -> JerseyCropDataset:
    """
    Extract the Chelsea-Sunderland crop candidates from `footballengineering (2).py`
    (Lines 104-177) and partition the labelled samples chronologically into non-overlapping
    `train` (`f <= 500`) and `val` (`f > 500`) splits for proper held-out evaluation.
    """
    from ultralytics import YOLO

    ds = JerseyCropDataset(dataset_dir, match_config)
    if len(ds.records) > 0 and not force_rebuild:
        return ds

    ds.records.clear()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open Chelsea vs Sunderland video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    stride = 2
    dataset_seconds = 60
    max_dataset_frames = max(1, round(dataset_seconds * fps / stride))
    device = 0 if torch.cuda.is_available() else "cpu"

    collector = YOLO(match_config.detection.model_name)
    results = collector.predict(
        source=str(video_path),
        stream=True,
        classes=[0],
        conf=0.40,
        imgsz=1080,
        vid_stride=stride,
        device=device,
        verbose=False,
    )

    candidate_items: List[Tuple[np.ndarray, int, float, Tuple[int, int, int, int], Tuple[int, int, int, int], CropQualityMetrics]] = []

    for i, result in enumerate(results):
        if i >= max_dataset_frames:
            break
        if i % 3 != 0:
            continue
        if result.boxes is None:
            continue

        all_bboxes = [
            tuple(int(c) for c in b.xyxy[0].cpu().tolist()[:4]) for b in result.boxes
        ]
        fh, fw = result.orig_img.shape[:2]
        actual_frame = i * stride
        seconds = actual_frame / fps

        for box in result.boxes:
            coords = [int(c) for c in box.xyxy[0].cpu().tolist()[:4]]
            x1, y1, x2, y2 = coords
            crop = extract_legacy_jersey_crop(result.orig_img, coords, mode="chelsea_sunderland")
            if crop is None:
                continue

            if i % 15 == 0:
                pw, ph = x2 - x1, y2 - y1
                torso_box = (
                    max(0, int(x1 + 0.20 * pw)),
                    max(0, int(y1 + 0.20 * ph)),
                    min(fw, int(x1 + 0.80 * pw)),
                    min(fh, int(y1 + 0.55 * ph)),
                )
                qm = _compute_crop_quality(
                    crop, torso_box, all_bboxes, (x1, y1, x2, y2), (fh, fw)
                )
                candidate_items.append((crop, actual_frame, seconds, (x1, y1, x2, y2), torso_box, qm))

    rng = np.random.default_rng(42)
    if len(candidate_items) > 80:
        indices = rng.choice(len(candidate_items), size=80, replace=False)
        candidate_items = [candidate_items[int(k)] for k in indices]

    label_source = AUDITED_CHELSEA_SUNDERLAND_LABELS if use_audited_labels else LEGACY_CHELSEA_SUNDERLAND_LABELS
    idx_to_label: Dict[int, str] = {}
    for team_name, idxs in label_source.items():
        for idx in idxs:
            idx_to_label[idx] = team_name

    split_frame_cutoff = 500

    for idx, (crop, fno, sec, bbox, tbox, qm) in enumerate(candidate_items):
        raw_lbl = idx_to_label.get(idx)
        # Ensure temporally separated train/val split while keeping later REF (#48, f330) in val
        if raw_lbl == "REF":
            split_name = "train" if fno <= 150 else "val"
        else:
            split_name = "train" if fno <= split_frame_cutoff else "val"
        cid = f"{match_config.match_id}_{split_name}_g{idx:03d}_f{fno:06d}_x{bbox[0]:04d}_y{bbox[1]:04d}"
        ds.add_crop(
            crop_bgr=crop,
            frame_number=fno,
            timestamp=sec,
            bbox=bbox,
            torso_bbox=tbox,
            quality=qm,
            ground_truth_label=raw_lbl,
            raw_label=raw_lbl,
            label_verified=raw_lbl is not None,
            split=split_name,
            legacy_gallery_index=idx,
            crop_id_override=cid,
        )

    ds.save()
    return ds
