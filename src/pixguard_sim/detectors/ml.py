"""Scikit-learn baseline detectors (LR / RF / GB).

A thin adapter that wraps a scikit-learn classifier behind the
:class:`~pixguard_sim.detectors.base.Detector` protocol, mirroring the baseline
family of the open PIX generator prior art (logistic regression, random
forest, gradient boosting) so the comparison is like-for-like. The detector is
fit on the numeric feature columns of the schema and exposes a calibrated-style
probability as its score.

The ``inference_budget_ms`` argument models the detector's decision latency on
the event's relative timeline. Two detectors with near-identical batch
precision/recall but different inference budgets are separated by the harness's
pre-deadline flag fraction; this is the headline discriminative claim, so the
budget is a first-class, configurable property of the detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pixguard_sim.detectors.base import Detector
from pixguard_sim.schema import FEATURE_COLUMNS

_FACTORIES = {
    "lr": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=120, max_depth=12, random_state=seed, n_jobs=1
    ),
    "gb": lambda seed: GradientBoostingClassifier(random_state=seed),
}


class SklearnDetector(Detector):
    """A scikit-learn classifier behind the detector protocol."""

    def __init__(
        self,
        estimator: ClassifierMixin,
        name: str,
        inference_budget_ms: int = 0,
    ) -> None:
        """Wrap a (constructed, unfitted) scikit-learn estimator.

        Args:
            estimator: Any classifier exposing ``fit``/``predict_proba``.
            name: Human-readable detector name used in reports.
            inference_budget_ms: Decision latency on the relative timeline.
        """
        self.estimator = estimator
        self.name = name
        self.inference_budget_ms = inference_budget_ms

    def fit(self, frame: pd.DataFrame) -> SklearnDetector:
        """Fit on the schema's numeric feature columns."""
        x = frame[list(FEATURE_COLUMNS)].to_numpy()
        y = frame["is_fraud"].to_numpy()
        self.estimator.fit(x, y)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probability per event."""
        x = frame[list(FEATURE_COLUMNS)].to_numpy()
        proba = self.estimator.predict_proba(x)
        # Column index of the positive (fraud) class.
        classes = list(self.estimator.classes_)
        pos = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, pos].astype("float64")


def make_ml_detector(
    kind: str, seed: int, inference_budget_ms: int = 0, name: str | None = None
) -> SklearnDetector:
    """Construct a baseline ML detector by short name.

    Args:
        kind: One of ``"lr"``, ``"rf"``, ``"gb"``.
        seed: RNG seed for the estimator, for determinism.
        inference_budget_ms: Decision latency on the relative timeline.
        name: Optional override for the detector name.

    Returns:
        A constructed (unfitted) :class:`SklearnDetector`.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in _FACTORIES:
        raise ValueError(f"unknown ML detector kind: {kind!r}")
    estimator = _FACTORIES[kind](seed)
    return SklearnDetector(
        estimator=estimator,
        name=name or kind,
        inference_budget_ms=inference_budget_ms,
    )
