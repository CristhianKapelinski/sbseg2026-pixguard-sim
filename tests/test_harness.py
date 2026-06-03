"""Unit tests for the harness, detectors, and the real-schema adapters.

The adapter tests build small frames carrying the *real* column names of each
released dataset (Tide, pix-fraud-br, AMLSim) so the schema mapping is exercised
with no network and no large downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pixguard_sim.adapters import (
    adapt_amlsim,
    adapt_pix_fraud_br,
    adapt_tide,
)
from pixguard_sim.config import GeneratorConfig, HarnessConfig
from pixguard_sim.detectors import RuleThresholdDetector, make_ml_detector
from pixguard_sim.detectors.base import measure_score_latency_ms
from pixguard_sim.generator import generate_events
from pixguard_sim.harness import evaluate_all, train_eval_split
from pixguard_sim.schema import EVENT_COLUMNS

_SMALL = GeneratorConfig(n_accounts=300, n_legit_events=2000, seed=7)


def _tide_frame(n: int = 400, fraud_rate: float = 0.05, seed: int = 3) -> pd.DataFrame:
    """A small frame in Tide's real released transaction schema."""
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < fraud_rate)
    return pd.DataFrame({
        "src": [f"individual_{i % 50}" for i in range(n)],
        "dest": [f"account_{i % 80}" for i in range(n)],
        "edge_type": "transaction",
        "amount": np.round(rng.lognormal(5, 1, n), 2),
        "currency": "EUR",
        "is_fraudulent": fraud,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h").astype(str),
        "transaction_type": "TransactionType.PAYMENT",
    })


def _pfb_frame(n: int = 400, fraud_rate: float = 0.05, seed: int = 5) -> pd.DataFrame:
    """A small frame in pix-fraud-br's real released schema."""
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame({
        "id_pagador": [f"***.{i:03d}.000-**" for i in range(n)],
        "id_recebedor": [f"***.{(i * 7) % 1000:03d}.111-**" for i in range(n)],
        "tipo_transacao": "chave_pix",
        "valor_brl": np.round(rng.lognormal(11, 1, n), 2),
        "saldo_anterior_pagador": rng.lognormal(10, 1, n),
        "saldo_posterior_pagador": rng.lognormal(10, 1, n),
        "saldo_anterior_recebedor": rng.lognormal(9, 1, n),
        "saldo_posterior_recebedor": rng.lognormal(9, 1, n),
        "datetime_brasilia": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "hora_dia": rng.integers(0, 24, n),
        "dia_semana": "segunda",
        "dia_util": True,
        "horario_noturno": rng.random(n) < 0.3,
        "acima_limite_noturno": False,
        "razao_saldo_residual": rng.random(n),
        "proporcao_valor_recebedor": rng.random(n),
        "fraude": fraud,
    })


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


def test_decision_latency_is_measured_not_assumed() -> None:
    """Decision latency must come from a real timed scoring call, not a budget."""
    frame = generate_events(_SMALL)
    det = make_ml_detector("lr", seed=7, inference_budget_ms=500).fit(frame)
    # Before any timed scoring, the latency defaults to instantaneous, never the
    # hand-set budget.
    assert det.measured_latency_ms == 0.0
    scores, total_ms, per_event_ms = measure_score_latency_ms(det, frame)
    assert len(scores) == len(frame)
    assert total_ms >= 0.0
    assert per_event_ms >= 0.0
    # decision_latency_ms now returns the measured per-event latency, not 500.
    latency = det.decision_latency_ms(frame)
    assert np.all(latency == per_event_ms)
    assert per_event_ms != 500.0
    assert len(latency) == len(frame)


def test_evaluate_all_reports_carry_confidence_intervals() -> None:
    frame = generate_events(_SMALL)
    cfg = HarnessConfig(n_bootstrap=50)
    reports = evaluate_all(
        [RuleThresholdDetector(), make_ml_detector("lr", 7, 50)], frame, cfg, seed=7
    )
    assert len(reports) == 2
    for r in reports:
        # Batch block carries point estimates plus Wilson and bootstrap CIs.
        assert {"precision", "recall", "f1", "pr_auc"} <= set(r.batch.keys())
        assert {"recall_ci_lo", "recall_ci_hi", "f1_ci_lo", "pr_auc_ci_hi"} <= set(
            r.batch.keys()
        )
        # Each pre-deadline entry is a proportion with a CI and N.
        for entry in r.pre_deadline_fraction.values():
            assert {"value", "ci_lo", "ci_hi", "n"} <= set(entry.keys())
            assert entry["ci_lo"] <= entry["value"] <= entry["ci_hi"]


def test_adapt_tide_real_schema_roundtrips() -> None:
    frame = adapt_tide(_tide_frame())
    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert frame["is_fraud"].sum() > 0


def test_adapt_tide_drops_non_transaction_edges() -> None:
    df = _tide_frame(n=100)
    df.loc[:49, "edge_type"] = "ownership"
    frame = adapt_tide(df)
    assert len(frame) == 50


def test_adapt_pix_fraud_br_real_schema_and_features() -> None:
    frame = adapt_pix_fraud_br(_pfb_frame())
    assert set(EVENT_COLUMNS) <= set(frame.columns)
    assert "razao_saldo_residual" in frame.columns
    assert frame["is_fraud"].sum() > 0


def test_adapt_amlsim_real_schema() -> None:
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2022-01-01", periods=20, freq="min").astype(str),
        "From Bank": 0,
        "Account": [f"A{i}" for i in range(20)],
        "To Bank": 1,
        "Account.1": [f"B{i}" for i in range(20)],
        "Amount Received": np.arange(20, dtype=float) * 10,
        "Is Laundering": ([0] * 18) + [1, 1],
    })
    frame = adapt_amlsim(df)
    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert frame["is_fraud"].sum() == 2


def test_adapt_tide_missing_column_raises() -> None:
    bad = pd.DataFrame({"src": [1], "dest": [0]})
    try:
        adapt_tide(bad)
    except KeyError as exc:
        assert "missing Tide columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")
