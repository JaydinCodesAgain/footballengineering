"""Dataset management, manifest persistence, label review, and leakage-free splitting."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.config import MatchConfig, SplitConfig
from src.schemas import CropQualityMetrics, InternalLabel, JerseyCropMetadata


def make_crop_id(
    match_id: str,
    frame_number: int,
    bbox: Tuple[int, int, int, int],
    tracking_id: Optional[int] = None,
) -> str:
    """Create a deterministic, human-readable crop identifier decoupled from gallery order."""
    x1, y1, x2, y2 = bbox
    trk_part = f"_t{tracking_id:04d}" if tracking_id is not None else ""
    return f"{match_id}_f{frame_number:06d}{trk_part}_x{x1:04d}_y{y1:04d}_w{x2 - x1:03d}_h{y2 - y1:03d}"


class JerseyCropDataset:
    """Manages persistent jersey crop images, metadata manifests, label updates, and splits."""

    def __init__(self, root_dir: str | Path, match_config: MatchConfig) -> None:
        self.root_dir = Path(root_dir)
        self.match_config = match_config
        self.images_dir = self.root_dir / "images"
        self.manifest_json_path = self.root_dir / "manifest.json"
        self.manifest_csv_path = self.root_dir / "manifest.csv"
        self.records: Dict[str, JerseyCropMetadata] = {}
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_json_path.is_file():
            self.load()

    def add_crop(
        self,
        crop_bgr: np.ndarray,
        frame_number: int,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        torso_bbox: Tuple[int, int, int, int],
        quality: CropQualityMetrics,
        tracking_id: Optional[int] = None,
        ground_truth_label: Optional[str] = None,
        raw_label: Optional[str] = None,
        label_verified: bool = False,
        split: str = "unassigned",
        legacy_gallery_index: Optional[int] = None,
        crop_id_override: Optional[str] = None,
    ) -> JerseyCropMetadata:
        """Save a crop image to disk and register its metadata in the dataset."""
        crop_id = crop_id_override or make_crop_id(
            self.match_config.match_id, frame_number, bbox, tracking_id
        )
        rel_img_path = f"images/{crop_id}.png"
        abs_img_path = self.root_dir / rel_img_path
        cv2.imwrite(str(abs_img_path), crop_bgr)

        internal_label: Optional[str] = None
        if ground_truth_label is not None and str(ground_truth_label).strip() != "":
            internal_label = self.match_config.to_internal_label(ground_truth_label)

        is_labelled = internal_label is not None
        meta = JerseyCropMetadata(
            crop_id=crop_id,
            match_id=self.match_config.match_id,
            frame_number=int(frame_number),
            timestamp=round(float(timestamp), 3),
            tracking_id=tracking_id,
            bbox=bbox,
            torso_bbox=torso_bbox,
            ground_truth_label=internal_label,
            raw_label=raw_label or (
                self.match_config.to_display_label(internal_label) if internal_label else None
            ),
            is_labelled=is_labelled,
            label_verified=bool(label_verified and is_labelled),
            split=split,
            image_path=rel_img_path,
            quality=quality,
            legacy_gallery_index=legacy_gallery_index,
        )
        self.records[crop_id] = meta
        return meta

    def load_crop_image(self, crop_id: str) -> np.ndarray:
        """Load the BGR image array for a registered crop ID."""
        if crop_id not in self.records:
            raise KeyError(f"Crop ID '{crop_id}' not found in dataset.")
        meta = self.records[crop_id]
        img_path = self.root_dir / meta.image_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Crop image file missing: {img_path}")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode crop image: {img_path}")
        return img

    def update_label(
        self,
        crop_id: str,
        new_label: Optional[str],
        verified: bool = True,
    ) -> JerseyCropMetadata:
        """Update or clear the ground-truth label for a crop by its stable ID."""
        if crop_id not in self.records:
            raise KeyError(f"Cannot update unknown crop_id '{crop_id}'.")
        rec = self.records[crop_id]
        if new_label is None or str(new_label).strip() == "":
            rec.ground_truth_label = None
            rec.raw_label = None
            rec.is_labelled = False
            rec.label_verified = False
        else:
            internal = self.match_config.to_internal_label(new_label)
            rec.ground_truth_label = internal
            rec.raw_label = self.match_config.to_display_label(internal)
            rec.is_labelled = True
            rec.label_verified = verified
        return rec

    def save(self) -> None:
        """Persist metadata records to both JSON and human-editable CSV manifests."""
        ordered = sorted(
            self.records.values(),
            key=lambda r: (r.split, r.legacy_gallery_index if r.legacy_gallery_index is not None else 999999, r.frame_number, r.crop_id),
        )
        payload = {
            "match_id": self.match_config.match_id,
            "match_name": self.match_config.match_name,
            "total_crops": len(ordered),
            "labelled_crops": sum(1 for r in ordered if r.is_labelled),
            "verified_crops": sum(1 for r in ordered if r.label_verified),
            "crops": [r.to_dict() for r in ordered],
        }
        with self.manifest_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        fieldnames = [
            "crop_id",
            "match_id",
            "split",
            "legacy_gallery_index",
            "frame_number",
            "timestamp",
            "tracking_id",
            "ground_truth_label",
            "raw_label",
            "is_labelled",
            "label_verified",
            "bbox",
            "torso_bbox",
            "blur_score",
            "grass_ratio",
            "occlusion_ratio",
            "image_path",
        ]
        with self.manifest_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ordered:
                writer.writerow(
                    {
                        "crop_id": r.crop_id,
                        "match_id": r.match_id,
                        "split": r.split,
                        "legacy_gallery_index": r.legacy_gallery_index if r.legacy_gallery_index is not None else "",
                        "frame_number": r.frame_number,
                        "timestamp": r.timestamp,
                        "tracking_id": r.tracking_id if r.tracking_id is not None else "",
                        "ground_truth_label": r.ground_truth_label or "",
                        "raw_label": r.raw_label or "",
                        "is_labelled": r.is_labelled,
                        "label_verified": r.label_verified,
                        "bbox": json.dumps(list(r.bbox)),
                        "torso_bbox": json.dumps(list(r.torso_bbox)),
                        "blur_score": round(r.quality.blur_score, 2),
                        "grass_ratio": round(r.quality.grass_ratio, 3),
                        "occlusion_ratio": round(r.quality.occlusion_ratio, 3),
                        "image_path": r.image_path,
                    }
                )

    def load(self) -> None:
        """Load dataset records from `manifest.json`."""
        with self.manifest_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.records.clear()
        for item in data.get("crops", []):
            rec = JerseyCropMetadata.from_dict(item)
            self.records[rec.crop_id] = rec

    def import_labels_from_csv(self, csv_path: str | Path) -> int:
        """
        Import reviewed or corrected labels from a CSV file.
        Returns the number of modified records.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        updated = 0
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("crop_id", "").strip()
                if not cid or cid not in self.records:
                    continue
                new_gt = row.get("ground_truth_label", "").strip()
                new_raw = row.get("raw_label", "").strip()
                verified_str = row.get("label_verified", "true").strip().lower()
                verified = verified_str in ("true", "1", "yes", "y")

                label_to_apply = new_gt or new_raw
                rec = self.records[cid]
                if label_to_apply:
                    internal = self.match_config.to_internal_label(label_to_apply)
                    if rec.ground_truth_label != internal or rec.label_verified != verified:
                        self.update_label(cid, internal, verified=verified)
                        updated += 1
                elif rec.is_labelled:
                    self.update_label(cid, None, verified=False)
                    updated += 1

                split_val = row.get("split", "").strip()
                if split_val in ("train", "val", "test", "unassigned") and rec.split != split_val:
                    rec.split = split_val
                    updated += 1

        if updated > 0:
            self.save()
        return updated

    def get_labelled_samples(
        self,
        split: Optional[str] = None,
        exclude_unknown: bool = False,
        verified_only: bool = False,
    ) -> List[Tuple[JerseyCropMetadata, np.ndarray, str]]:
        """
        Return list of `(metadata, crop_bgr, internal_label)` for labelled samples.
        """
        samples: List[Tuple[JerseyCropMetadata, np.ndarray, str]] = []
        ordered = sorted(
            self.records.values(),
            key=lambda r: (
                r.legacy_gallery_index if r.legacy_gallery_index is not None else 999999,
                r.frame_number,
                r.crop_id,
            ),
        )
        for rec in ordered:
            if not rec.is_labelled or rec.ground_truth_label is None:
                continue
            if verified_only and not rec.label_verified:
                continue
            if split is not None and rec.split != split:
                continue
            if exclude_unknown and rec.ground_truth_label == InternalLabel.UNKNOWN.value:
                continue
            img = self.load_crop_image(rec.crop_id)
            samples.append((rec, img, rec.ground_truth_label))
        return samples

    def get_unlabelled_samples(self) -> List[Tuple[JerseyCropMetadata, np.ndarray]]:
        """Return list of `(metadata, crop_bgr)` for unlabelled crops."""
        samples: List[Tuple[JerseyCropMetadata, np.ndarray]] = []
        for rec in sorted(self.records.values(), key=lambda r: (r.frame_number, r.crop_id)):
            if not rec.is_labelled:
                samples.append((rec, self.load_crop_image(rec.crop_id)))
        return samples

    def class_distribution(self, split: Optional[str] = None) -> Dict[str, int]:
        """Count labelled samples per internal class for a given split (or all splits)."""
        counts: Counter[str] = Counter()
        for rec in self.records.values():
            if not rec.is_labelled or rec.ground_truth_label is None:
                continue
            if split is not None and rec.split != split:
                continue
            counts[rec.ground_truth_label] += 1
        return dict(counts)


def assign_temporal_and_track_splits(
    records: Sequence[JerseyCropMetadata],
    video_duration_seconds: float,
    split_config: Optional[SplitConfig] = None,
    min_temporal_gap_seconds: float = 1.5,
) -> Dict[str, str]:
    """
    Assign leakage-free `train`, `val`, and `test` splits using temporal boundaries and tracking IDs.

    Rules:
    1. Crops before `train_end_fraction * duration` are assigned to `train`.
    2. Crops between `val_start_fraction * duration` and `val_end_fraction * duration` are assigned to `val`.
    3. Crops at or after `test_start_fraction * duration` are assigned to `test`.
    4. Any crop in the buffer zone or sharing a `tracking_id` with an earlier split is isolated to prevent
       same-player identity leakage across splits.
    """
    cfg = split_config or SplitConfig()
    if video_duration_seconds <= 0:
        raise ValueError("video_duration_seconds must be positive.")

    train_end = cfg.train_end_fraction * video_duration_seconds
    val_start = max(train_end + min_temporal_gap_seconds, cfg.val_start_fraction * video_duration_seconds)
    val_end = cfg.val_end_fraction * video_duration_seconds
    test_start = max(val_end + min_temporal_gap_seconds, cfg.test_start_fraction * video_duration_seconds)

    assignments: Dict[str, str] = {}
    track_to_split: Dict[int, str] = {}

    # Sort chronologically
    ordered = sorted(records, key=lambda r: (r.timestamp, r.frame_number, r.crop_id))
    for rec in ordered:
        t = rec.timestamp
        if t < train_end:
            candidate_split = "train"
        elif val_start <= t < val_end:
            candidate_split = "val"
        elif t >= test_start:
            candidate_split = "test"
        else:
            candidate_split = "unassigned"

        # Prevent track ID leakage across splits
        if rec.tracking_id is not None and candidate_split != "unassigned":
            existing_split = track_to_split.get(rec.tracking_id)
            if existing_split is not None and existing_split != candidate_split:
                # Keep the track in its first assigned split or mark unassigned if across buffer
                candidate_split = "unassigned"
            else:
                track_to_split[rec.tracking_id] = candidate_split

        assignments[rec.crop_id] = candidate_split

    return assignments


def validate_no_split_leakage(
    records: Sequence[JerseyCropMetadata],
    min_frame_gap: int = 5,
) -> Dict[str, object]:
    """
    Verify that train, val, and test splits have no overlapping crop IDs, no shared tracking IDs,
    and no temporally adjacent frames across splits.
    """
    by_split: Dict[str, List[JerseyCropMetadata]] = {"train": [], "val": [], "test": []}
    for rec in records:
        if rec.split in by_split and rec.is_labelled:
            by_split[rec.split].append(rec)

    shared_tracks: List[Tuple[str, str, int]] = []
    temporal_violations: List[Tuple[str, str, int, int]] = []

    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in split_pairs:
        r1 = by_split[s1]
        r2 = by_split[s2]

        tracks1 = {r.tracking_id for r in r1 if r.tracking_id is not None}
        tracks2 = {r.tracking_id for r in r2 if r.tracking_id is not None}
        for tid in sorted(tracks1 & tracks2):
            shared_tracks.append((s1, s2, tid))

        for a in r1:
            for b in r2:
                if a.match_id == b.match_id and abs(a.frame_number - b.frame_number) < min_frame_gap:
                    temporal_violations.append((a.crop_id, b.crop_id, a.frame_number, b.frame_number))

    is_clean = len(shared_tracks) == 0 and len(temporal_violations) == 0
    return {
        "is_clean": is_clean,
        "shared_tracks": shared_tracks,
        "temporal_violations": temporal_violations,
        "counts_per_split": {k: len(v) for k, v in by_split.items()},
    }

