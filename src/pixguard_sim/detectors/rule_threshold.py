"""Industry-style rule-threshold baseline detector.

A transparent floor detector: it flags an event when the amount exceeds a
threshold or velocity is high, optionally combined with a new-payee signal. It
keys only on observable behavioural signals (amount, trailing-hour velocity,
new-payee), never on the per-scenario label columns (``is_remote_session``,
``coercion_flag``, ``med_layer``), so it carries no label leakage. It
deliberately ignores the deep-layer MED structure and the coercion-only signal,
which is why it is expected to miss deep MED layers and coercion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pixguard_sim.detectors.base import Detector


class RuleThresholdDetector(Detector):
    """Threshold rule on amount, velocity, and high-risk binary signals."""

    name = "rule_threshold"

    def __init__(
        self,
        amount_threshold_brl: float = 1500.0,
        velocity_threshold: int = 5,
        inference_budget_ms: int = 20,
    ) -> None:
        """Initialize the rule thresholds.

        Args:
            amount_threshold_brl: Amount above which an event is suspicious.
            velocity_threshold: Trailing-hour transfer count above which an
                event is suspicious.
            inference_budget_ms: Fixed in-rail decision latency.
        """
        self.amount_threshold_brl = amount_threshold_brl
        self.velocity_threshold = velocity_threshold
        self.inference_budget_ms = inference_budget_ms

    def fit(self, frame: pd.DataFrame) -> RuleThresholdDetector:
        """No-op fit; the rule is fixed by construction."""
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a graded rule score in ``[0, 1]`` per event.

        Keys only on observable behavioural signals (amount, velocity,
        new-payee); it does not read the leaked per-scenario label columns.
        """
        amount = frame["amount_brl"].to_numpy()
        velocity = frame["payer_velocity_1h"].to_numpy()
        new_payee = frame["new_payee"].to_numpy()

        score = np.zeros(len(frame), dtype="float64")
        score += 0.5 * (amount > self.amount_threshold_brl)
        score += 0.3 * (velocity > self.velocity_threshold)
        score += 0.2 * (new_payee == 1)
        return np.clip(score, 0.0, 1.0)
