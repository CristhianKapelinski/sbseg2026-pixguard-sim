"""Unit tests for the evaluation metrics, especially the headline metric."""

from __future__ import annotations

import numpy as np

from pixguard_sim.metrics import (
    batch_metrics,
    pre_deadline_flag_fraction,
    pre_deadline_with_ci,
    recall_by_group,
    wilson_interval,
)


def test_batch_metrics_perfect() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    m = batch_metrics(y, scores, threshold=0.5, n_bootstrap=50)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == 1.0


def test_batch_metrics_no_fraud_pr_auc_zero() -> None:
    y = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])
    assert batch_metrics(y, scores, 0.5, n_bootstrap=10)["pr_auc"] == 0.0


def test_wilson_interval_brackets_point_estimate() -> None:
    lo, hi = wilson_interval(9, 10)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0


def test_wilson_interval_zero_n_is_degenerate() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_pre_deadline_with_ci_reports_counts_and_bounds() -> None:
    y = np.array([1, 1, 1, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    decision = np.array([100, 500, 2000, 100])
    out = pre_deadline_with_ci(y, scores, decision, threshold=0.5, deadline_ms=1000)
    assert out["k"] == 2 and out["n"] == 3
    assert out["ci_lo"] <= out["value"] <= out["ci_hi"]


def test_pre_deadline_counts_only_in_time_flagged_frauds() -> None:
    y = np.array([1, 1, 1, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    # event 0 decides at 100ms, event 1 at 500ms, event 2 at 2000ms.
    decision = np.array([100, 500, 2000, 100])
    # Deadline 1000ms: events 0 and 1 are in time, event 2 is late.
    frac = pre_deadline_flag_fraction(y, scores, decision, threshold=0.5, deadline_ms=1000)
    assert frac == 2 / 3


def test_pre_deadline_late_decision_excluded() -> None:
    y = np.array([1])
    scores = np.array([1.0])
    decision = np.array([5000])
    # Flagged but after the deadline: not actionable, fraction must be 0.
    assert pre_deadline_flag_fraction(y, scores, decision, 0.5, 1000) == 0.0


def test_pre_deadline_unflagged_excluded() -> None:
    y = np.array([1])
    scores = np.array([0.1])
    decision = np.array([100])
    # In time but below threshold: not flagged, fraction must be 0.
    assert pre_deadline_flag_fraction(y, scores, decision, 0.5, 1000) == 0.0


def test_pre_deadline_no_fraud_returns_zero() -> None:
    y = np.array([0, 0])
    scores = np.array([0.9, 0.9])
    decision = np.array([10, 10])
    assert pre_deadline_flag_fraction(y, scores, decision, 0.5, 1000) == 0.0


def test_recall_by_group_per_layer() -> None:
    y = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    groups = np.array([0, 0, 1, 1])
    out = recall_by_group(y, scores, groups, threshold=0.5)
    assert out[0] == 0.5
    assert out[1] == 0.5
