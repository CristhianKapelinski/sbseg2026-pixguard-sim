"""Tabular baseline detectors (LR / RF / GB / XGBoost).

A thin adapter that wraps a tabular classifier behind the
:class:`~pixguard_sim.detectors.base.Detector` protocol, mirroring the baseline
family of the open PIX generator prior art (logistic regression, random
forest, gradient boosting, XGBoost) so the comparison is like-for-like. The
detector is fit on the numeric feature columns of the schema and exposes a
probability as its score.

The ``inference_budget_ms`` argument is retained only as descriptive metadata.
The harness scores the pre-deadline flag fraction on each detector's *measured*
per-event scoring latency (timed by ``measure_score_latency_ms``), not on an
assumed budget, so the deadline metric reflects observed inference speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _make_xgb(seed: int) -> ClassifierMixin:
    """Construct an XGBoost classifier, importing lazily.

    XGBoost is the strongest tabular baseline reported by the prior art, so it
    is included for a like-for-like comparison. It is imported lazily so the
    core package does not hard-depend on it.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )


from pixguard_sim.detectors.base import Detector  # noqa: E402
from pixguard_sim.schema import FEATURE_COLUMNS  # noqa: E402

_FACTORIES = {
    "lr": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=120, max_depth=12, random_state=seed, n_jobs=1
    ),
    "gb": lambda seed: GradientBoostingClassifier(random_state=seed),
    "xgb": _make_xgb,
}


class SklearnDetector(Detector):
    """A scikit-learn classifier behind the detector protocol."""

    def __init__(
        self,
        estimator: ClassifierMixin,
        name: str,
        inference_budget_ms: int = 0,
        features: tuple[str, ...] | None = None,
    ) -> None:
        """Wrap a (constructed, unfitted) scikit-learn estimator.

        Args:
            estimator: Any classifier exposing ``fit``/``predict_proba``.
            name: Human-readable detector name used in reports.
            inference_budget_ms: Decision latency on the relative timeline.
            features: Feature columns to fit on; defaults to the shared schema
                feature columns. A generator with extra engineered features
                (e.g. pix-fraud-br) can pass them so the prior-art reproduction
                uses identical signal.
        """
        self.estimator = estimator
        self.name = name
        self.inference_budget_ms = inference_budget_ms
        self.features = tuple(features) if features else FEATURE_COLUMNS

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Select the configured feature columns present in the frame."""
        cols = [c for c in self.features if c in frame.columns]
        return frame[cols].to_numpy()

    def fit(self, frame: pd.DataFrame) -> SklearnDetector:
        """Fit on the configured numeric feature columns."""
        x = self._matrix(frame)
        y = frame["is_fraud"].to_numpy()
        self.estimator.fit(x, y)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probability per event."""
        x = self._matrix(frame)
        proba = self.estimator.predict_proba(x)
        # Column index of the positive (fraud) class.
        classes = list(self.estimator.classes_)
        pos = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, pos].astype("float64")


def make_ml_detector(
    kind: str,
    seed: int,
    inference_budget_ms: int = 0,
    name: str | None = None,
    features: tuple[str, ...] | None = None,
) -> SklearnDetector:
    """Construct a baseline ML detector by short name.

    Args:
        kind: One of ``"lr"``, ``"rf"``, ``"gb"``, ``"xgb"``.
        seed: RNG seed for the estimator, for determinism.
        inference_budget_ms: Decision latency on the relative timeline.
        name: Optional override for the detector name.
        features: Optional feature columns (defaults to the shared schema set).

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
        features=features,
    )
