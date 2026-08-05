"""Regression tests: no feature column may give the label away.

Two label leaks shipped before these tests existed, one in the adapters and one
in the generator, and the whole suite stayed green through both. Both had the
same shape: a feature whose values were reachable from only one class, so a
threshold on it separated fraud from legitimate traffic exactly and a detector
scored the generator's bookkeeping instead of behaviour.

The rule these tests encode is that every feature a detector sees must be
reachable from BOTH classes, with no region where one class is impossible by
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixguard_sim.adapters import _synth_signals
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.generator import generate_events
from pixguard_sim.schema import FEATURE_COLUMNS

# A feature is a deterministic indicator when a class simply cannot reach part
# of its range. Sampling noise puts a few tail events beyond the other class's
# observed maximum even for honestly overlapping distributions, so the check
# allows a small tail and fails on a wholesale split.
MAX_FRAUD_SHARE_ABOVE_LEGIT: float = 0.20


def _leak_share(values: np.ndarray, is_fraud: np.ndarray) -> float:
    """Share of fraud events sitting above every legitimate event."""
    legit = values[is_fraud == 0]
    fraud = values[is_fraud == 1]
    if legit.size == 0 or fraud.size == 0:
        return 0.0
    return float((fraud > legit.max()).mean())


def test_generator_features_are_reachable_from_both_classes() -> None:
    """No generated feature may put fraud in a region legitimate traffic cannot."""
    frame = generate_events(GeneratorConfig())
    is_fraud = frame["is_fraud"].to_numpy()
    offenders = {
        column: share
        for column in FEATURE_COLUMNS
        if (share := _leak_share(frame[column].to_numpy(dtype=float), is_fraud))
        > MAX_FRAUD_SHARE_ABOVE_LEGIT
    }
    assert not offenders, (
        "feature columns separate the classes by construction: "
        f"{offenders}; every column a detector sees must overlap across classes"
    )


@pytest.mark.parametrize("rate", [0.001, 0.01, 0.1])
def test_adapter_signals_are_reachable_from_both_classes(rate: float) -> None:
    """Signals synthesized for a foreign schema must not encode the label."""
    rng = np.random.default_rng(7)
    is_fraud = (rng.random(50_000) < rate).astype(int)
    device_changed, new_payee, velocity = _synth_signals(is_fraud, seed=7)
    for name, values in (
        ("device_changed", device_changed),
        ("new_payee", new_payee),
        ("payer_velocity_1h", velocity),
    ):
        legit = values[is_fraud == 0]
        assert legit.size and legit.max() >= values[is_fraud == 1].max(), (
            f"{name} is reachable only from fraud at rate {rate}: a detector "
            "reading this bit inherits the label"
        )


def test_adapter_device_signal_is_not_a_perfect_indicator() -> None:
    """The exact bug that shipped: device_changed drawn for fraud only."""
    rng = np.random.default_rng(11)
    is_fraud = (rng.random(50_000) < 0.01).astype(int)
    device_changed, _, _ = _synth_signals(is_fraud, seed=11)
    flagged = device_changed == 1
    assert flagged.any(), "signal never fires; the test would pass vacuously"
    precision = float(is_fraud[flagged].mean())
    assert precision < 0.9, (
        f"device_changed=1 implies fraud with probability {precision:.3f}; "
        "it must be drawn from both classes"
    )
