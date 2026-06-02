"""Unit tests for the harness, detectors, and adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pixguard_sim.adapters import (
    MissingSourceDataError,
    adapt_pix_fraud_br,
    adapt_tide,
    synthesize_tide_shaped,
)
from pixguard_sim.config import GeneratorConfig, HarnessConfig
from pixguard_sim.detectors import RuleThresholdDetector, make_ml_detector
from pixguard_sim.generator import generate_events
from pixguard_sim.harness import evaluate_all, train_eval_split
from pixguard_sim.schema import EVENT_COLUMNS

_SMALL = GeneratorConfig(n_accounts=300, n_legit_events=2000, seed=7)


def test_train_eval_split_stratified_and_disjoint() -> None:
    frame = generate_events(_SMALL)
    train, eval_ = train_eval_split(frame, seed=7, eval_fraction=0.4)
    assert train["is_fraud"].sum() > 0
    assert eval_["is_fraud"].sum() > 0
    assert len(train) + len(eval_) == len(frame)


def test_rule_detector_scores_in_unit_interval() -> None:
    frame = generate_events(_SMALL)
    det = RuleThresholdDetector().fit(frame)
    scores = det.score(frame)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_decision_latency_is_relative_to_initiation() -> None:
    frame = generate_events(_SMALL)
    det = make_ml_detector("lr", seed=7, inference_budget_ms=500)
    latency = det.decision_latency_ms(frame)
    # Latency is measured from each event's own initiation, so it equals the
    # detector's inference budget regardless of absolute t_init.
    assert np.all(latency == 500.0)
    assert len(latency) == len(frame)


def test_evaluate_all_produces_reports() -> None:
    frame = generate_events(_SMALL)
    cfg = HarnessConfig()
    reports = evaluate_all(
        [RuleThresholdDetector(), make_ml_detector("lr", 7, 50)], frame, cfg, seed=7
    )
    assert len(reports) == 2
    for r in reports:
        assert set(r.batch.keys()) == {"precision", "recall", "f1", "pr_auc"}
        assert set(r.pre_deadline_fraction.keys()) == set(cfg.deadline_sweep_ms)


def test_adapt_tide_roundtrips_schema() -> None:
    tide = synthesize_tide_shaped(n_events=500, fraud_rate=0.05, seed=3)
    frame = adapt_tide(tide)
    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert frame["is_fraud"].sum() > 0


def test_pending_adapter_raises_without_source() -> None:
    try:
        adapt_pix_fraud_br("/nonexistent/path/pix_fraud_br.csv")
    except MissingSourceDataError as exc:
        assert "PENDING" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected MissingSourceDataError")


def test_adapt_tide_missing_column_raises() -> None:
    bad = pd.DataFrame({"transaction_id": [1], "src_account": [0]})
    try:
        adapt_tide(bad)
    except KeyError as exc:
        assert "missing Tide columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")
