"""Audit measurable diversity limits of the local Sprint 3.5 review batch.

This intentionally does not infer whether a frame is a replay, close-up, or
contains a small football. Those require human review or verified ball boxes.
It reports only source metadata, luminance summaries, and perceptual-near-
duplicate candidates for a reviewer to inspect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.annotation_workflow import DetectionAnnotationStore


def _dhash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def audit_review_batch(manifest_path: Path, project_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: list[dict[str, Any]] = []
    all_dimensions: set[tuple[int, int]] = set()

    for source in manifest["splits"]:
        source_root = Path(source["annotation_root"])
        if not source_root.is_absolute():
            source_root = project_root / source_root
        store = DetectionAnnotationStore(source_root, str(source["match_id"]))
        annotations = [store.frames[key] for key in sorted(store.frames)]
        brightness: list[float] = []
        hashes: list[int] = []
        dimensions: set[tuple[int, int]] = set()
        unreadable = 0
        for annotation in annotations:
            image = cv2.imread(str(source_root / annotation.image_rel_path))
            if image is None:
                unreadable += 1
                continue
            height, width = image.shape[:2]
            dimensions.add((width, height))
            all_dimensions.add((width, height))
            brightness.append(float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()))
            hashes.append(_dhash(image))

        distances = [(left ^ right).bit_count() for left, right in zip(hashes, hashes[1:])]
        sources.append(
            {
                "match_id": source["match_id"],
                "split": source["split"],
                "review_frames": len(annotations),
                "image_dimensions": sorted([list(item) for item in dimensions]),
                "unreadable_images": unreadable,
                "mean_grayscale_luminance": round(float(np.mean(brightness)), 2) if brightness else None,
                "grayscale_luminance_stddev": round(float(np.std(brightness)), 2) if brightness else None,
                "adjacent_sparse_pairs": len(distances),
                "near_duplicate_candidates_dhash_le_5": sum(distance <= 5 for distance in distances),
                "semantic_scene_status": "not_auto_classified; reviewer must reject close-ups, replays, and transitions",
                "small_ball_status": "unavailable until manual ball boxes are reviewed",
            }
        )

    panoramic_sources = [item["match_id"] for item in sources if any(width / height > 2.0 for width, height in item["image_dimensions"])]
    return {
        "purpose": "representativeness audit, not detector evaluation",
        "total_review_frames": sum(int(item["review_frames"]) for item in sources),
        "image_dimensions_present": sorted([list(item) for item in all_dimensions]),
        "panoramic_sources": panoramic_sources,
        "limitations": [
            "Three sources are 1280x720 highlight-style recordings; they do not establish fixed-camera grassroots generalisation.",
            "The 4K SoccerTrack clip is previously inspected training-only context, not an untouched final test source.",
            "Close-ups, replays, transitions, active play, lighting category, and small-ball prevalence are not inferred automatically.",
            "No verified labels exist yet, so no category recall, hard-negative rate, or illumination-stratified ball metric is available.",
        ],
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/sprint3_5/training_data_manifest.json")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sprint3_5/batch_representativeness_audit.json")
    args = parser.parse_args()
    result = audit_review_batch(args.manifest, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
