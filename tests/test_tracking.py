"""Tests for TemporalTeamStabilizer, soft label locking, contradiction unlock, and identity resets."""

from __future__ import annotations

from src.config import TrackingConfig
from src.evaluation.tracking_eval import evaluate_manual_identity_continuity
from src.schemas import ClassificationPrediction, InternalLabel
from src.tracking.stabilizer import TemporalTeamStabilizer


def _make_pred(label: str, confidence: float = 0.85) -> ClassificationPrediction:
    is_unk = label == InternalLabel.UNKNOWN.value
    return ClassificationPrediction(
        predicted_label=label,
        raw_nearest_label=label,
        nearest_distance=0.10 if not is_unk else 0.75,
        second_nearest_distance=0.45 if not is_unk else 0.80,
        margin=0.35 if not is_unk else 0.05,
        confidence_score=confidence if not is_unk else 0.0,
        is_unknown=is_unk,
    )


def test_stabilizer_minimum_evidence_and_soft_locking() -> None:
    cfg = TrackingConfig(
        history_length=9,
        min_votes=3,
        min_agreement=0.60,
        confidence_weighted=True,
        lock_confidence_threshold=0.75,
        lock_min_observations=5,
        reclassify_contradiction_streak=4,
    )
    stabilizer = TemporalTeamStabilizer(cfg)
    bbox = (100, 100, 140, 200)

    # First two observations: below min_votes=3 -> returns Unknown
    s1, _, l1 = stabilizer.update(10, 1, bbox, _make_pred(InternalLabel.TEAM_A.value))
    s2, _, l2 = stabilizer.update(10, 2, bbox, _make_pred(InternalLabel.TEAM_A.value))
    assert s1 == InternalLabel.UNKNOWN.value and not l1
    assert s2 == InternalLabel.UNKNOWN.value and not l2

    # Third observation reaches min_votes=3 -> emits Team_A
    s3, conf3, l3 = stabilizer.update(10, 3, bbox, _make_pred(InternalLabel.TEAM_A.value))
    assert s3 == InternalLabel.TEAM_A.value
    assert conf3 >= 0.90
    assert not l3

    # Observations 4 and 5 reach lock_min_observations=5 -> locks Team_A
    stabilizer.update(10, 4, bbox, _make_pred(InternalLabel.TEAM_A.value))
    s5, _, l5 = stabilizer.update(10, 5, bbox, _make_pred(InternalLabel.TEAM_A.value))
    assert s5 == InternalLabel.TEAM_A.value
    assert l5 is True

    # Transient Unknown or single noisy Team_B frame does not break lock
    s6, _, l6 = stabilizer.update(10, 6, bbox, _make_pred(InternalLabel.UNKNOWN.value))
    s7, _, l7 = stabilizer.update(10, 7, bbox, _make_pred(InternalLabel.TEAM_B.value, confidence=0.50))
    assert s6 == InternalLabel.TEAM_A.value and l6 is True
    assert s7 == InternalLabel.TEAM_A.value and l7 is True


def test_stabilizer_unlocks_on_sustained_contradiction() -> None:
    cfg = TrackingConfig(
        history_length=7,
        min_votes=3,
        min_agreement=0.55,
        lock_confidence_threshold=0.75,
        lock_min_observations=4,
        reclassify_contradiction_streak=3,
    )
    stabilizer = TemporalTeamStabilizer(cfg)
    bbox = (100, 100, 140, 200)

    for f in range(1, 5):
        stabilizer.update(5, f, bbox, _make_pred(InternalLabel.TEAM_A.value, 0.85))
    assert stabilizer.tracks[5].locked_label == InternalLabel.TEAM_A.value

    # Feed sustained high-confidence Team_B observations to trigger unlock and reclassification
    for f in range(5, 11):
        lbl, _, _ = stabilizer.update(5, f, bbox, _make_pred(InternalLabel.TEAM_B.value, 0.95))

    assert lbl == InternalLabel.TEAM_B.value


def test_stabilizer_resets_on_large_spatial_jump_or_missing_frames() -> None:
    cfg = TrackingConfig(
        history_length=7,
        min_votes=3,
        min_agreement=0.60,
        max_missing_frames_before_reset=20,
        max_spatial_jump_ratio=2.5,
    )
    stabilizer = TemporalTeamStabilizer(cfg)
    bbox_left = (50, 100, 90, 180)
    for f in range(1, 5):
        stabilizer.update(22, f, bbox_left, _make_pred(InternalLabel.TEAM_A.value))

    assert stabilizer.tracks[22].current_stable_label == InternalLabel.TEAM_A.value

    # Track 22 suddenly jumps across the entire frame (e.g. ID switched to a distant player)
    bbox_far_right = (900, 100, 940, 180)
    lbl_after_jump, _, _ = stabilizer.update(22, 5, bbox_far_right, _make_pred(InternalLabel.TEAM_B.value))
    assert stabilizer.tracks[22].identity_resets == 1
    assert lbl_after_jump == InternalLabel.UNKNOWN.value  # Reset requires min_votes again

    diag = stabilizer.summarize_diagnostics(processed_frames=5)
    assert diag["identity_resets_triggered"] == 1


def test_manual_identity_continuity_audit() -> None:
    empty_res = evaluate_manual_identity_continuity({})
    assert empty_res["identity_ground_truth_available"] is False

    seq_res = evaluate_manual_identity_continuity(
        {
            "Player_1": [12, 12, 12, None, 12, 19, 19],
            "Player_2": [4, 4, 4, 4, 4],
        }
    )
    assert seq_res["identity_ground_truth_available"] is True
    assert seq_res["total_id_switches"] == 1
    assert seq_res["per_player"]["Player_1"]["id_switches"] == 1
    assert seq_res["per_player"]["Player_2"]["id_switches"] == 0

