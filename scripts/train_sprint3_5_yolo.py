"""Run the approved Sprint 3.5 YOLO11s experiment after manual-review gates pass.

The local environment does not install an ``yolo`` console executable, so this
launcher uses the installed Ultralytics Python API. It is dry-run by default
and refuses training unless the latest reviewed-data gate is eligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build_train_kwargs(config: dict[str, Any], dataset_yaml: Path, device_override: str | None = None) -> dict[str, Any]:
    """Translate the reviewed experiment configuration into Ultralytics kwargs."""
    if config.get("class_names") != ["ball"]:
        raise ValueError("Sprint 3.5 training is intentionally a one-class ['ball'] experiment.")
    training = dict(config["training"])
    augmentation = dict(config["augmentation"])
    training.pop("pretrained", None)  # The .pt base model already carries pretrained weights.
    if device_override is not None:
        training["device"] = device_override
    return {
        "data": str(dataset_yaml.resolve()),
        "project": str((ROOT / "outputs/sprint3_5/ultralytics_runs").resolve()),
        "name": str(config["experiment_name"]),
        "exist_ok": False,
        **training,
        **augmentation,
    }


def _require_eligible_gate(gate_path: Path) -> None:
    if not gate_path.is_file():
        raise FileNotFoundError(f"Training gate status is missing: {gate_path}. Run the dataset exporter first.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("training_eligible", False):
        blockers = " | ".join(str(item) for item in gate.get("blockers", []))
        raise ValueError(f"Training is blocked by reviewed-data gate: {blockers}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sprint3_5_ball_training.yaml")
    parser.add_argument("--dataset-yaml", type=Path, default=ROOT / "data/sprint3_5_ball_yolo/dataset.yaml")
    parser.add_argument("--gate-status", type=Path, default=ROOT / "outputs/sprint3_5/training_gate_status.json")
    parser.add_argument("--device", help="Optional Ultralytics device override, for example 0 or cpu.")
    parser.add_argument("--validate-model", action="store_true", help="Load yolo11s.pt but do not train.")
    parser.add_argument("--execute", action="store_true", help="Run training only after gate and dataset validation.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    kwargs = build_train_kwargs(config, args.dataset_yaml, args.device)
    print(json.dumps({"base_model": config["base_model"], "train_kwargs": kwargs}, indent=2))

    if args.validate_model:
        from ultralytics import YOLO

        model = YOLO(str(config["base_model"]))
        print(f"Loaded candidate base model: task={model.task}, checkpoint={model.ckpt_path}")

    if not args.execute:
        return

    _require_eligible_gate(args.gate_status)
    if not args.dataset_yaml.is_file():
        raise FileNotFoundError(f"Reviewed YOLO dataset YAML is missing: {args.dataset_yaml}")
    from ultralytics import YOLO

    model = YOLO(str(config["base_model"]))
    model.train(**kwargs)


if __name__ == "__main__":
    main()
