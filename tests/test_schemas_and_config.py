"""Tests for schemas, bounding-box validation, configuration loading, and label integrity."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config import MatchConfig, load_match_config
from src.schemas import BoundingBox, InternalLabel


def test_bounding_box_geometry_and_validation() -> None:
    bbox = BoundingBox.from_coords([10.2, 20.4, 50.1, 100.4])
    assert bbox.to_tuple() == (10, 20, 50, 100)
    assert bbox.width == 40
    assert bbox.height == 80
    assert bbox.area == 3200
    assert bbox.center == (30.0, 60.0)
    assert pytest.approx(bbox.aspect_ratio) == 0.5
    assert bbox.is_valid(min_width=10, min_height=20)

    invalid_box = BoundingBox(x1=50, y1=50, x2=40, y2=30)
    assert invalid_box.width == 0
    assert invalid_box.height == 0
    assert not invalid_box.is_valid()

    with pytest.raises(ValueError):
        BoundingBox.from_coords([1, 2, 3])


def test_config_loading_and_label_mapping() -> None:
    root = Path(__file__).resolve().parent.parent
    nw_cfg = load_match_config(root / "configs" / "norway_wales.yaml")
    cs_cfg = load_match_config(root / "configs" / "chelsea_sunderland.yaml")

    assert nw_cfg.match_id == "norway_wales"
    assert nw_cfg.to_internal_label("Norway") == InternalLabel.TEAM_A.value
    assert nw_cfg.to_internal_label("Wales") == InternalLabel.TEAM_B.value
    assert nw_cfg.to_internal_label("REF") == InternalLabel.REFEREE.value
    assert nw_cfg.to_internal_label("Unknown") == InternalLabel.UNKNOWN.value
    assert nw_cfg.to_display_label(InternalLabel.TEAM_A.value) == "Norway"

    assert cs_cfg.match_id == "chelsea_sunderland"
    assert cs_cfg.to_internal_label("Chelsea") == InternalLabel.TEAM_A.value
    assert cs_cfg.to_internal_label("Sunderland") == InternalLabel.TEAM_B.value
    assert cs_cfg.to_internal_label("Other") == InternalLabel.UNKNOWN.value

    # Cross-match incompatible label must raise ValueError
    with pytest.raises(ValueError, match="not compatible"):
        nw_cfg.to_internal_label("Chelsea")

    with pytest.raises(ValueError, match="not compatible"):
        cs_cfg.to_internal_label("Norway")

