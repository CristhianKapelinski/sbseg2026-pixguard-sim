"""Detector plugin protocol.

The harness depends only on this abstract surface, never on a concrete model.
A detector is fit once on a labeled training frame, then asked to score an
evaluation frame. Scores are floats in ``[0, 1]`` where higher means more
fraud-like; the harness applies a configurable threshold for binary metrics
and uses the raw scores for PR-AUC.

A detector also reports its own *decision latency* via
:meth:`Detector.decision_latency_ms`: the time, on the event's relative
timeline, at which the detector would emit its decision for an event. This is
what lets the harness compute the pre-deadline flag fraction without assuming
all detectors decide instantaneously.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    """Abstract base class for PIX fraud detectors."""

    name: str = "detector"

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> Detector:
        """Fit the detector on a labeled training frame.

        Args:
            frame: Training events with an ``is_fraud`` label column.

        Returns:
            ``self``, fitted.
        """

    @abstractmethod
    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a fraud score in ``[0, 1]`` per row of ``frame``."""

    def decision_latency_ms(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the decision latency per event, relative to its initiation.

        The decision deadline is a window measured from each event's own
        initiation (``t_init_ms``). This method therefore returns the latency
        *since initiation*, not an absolute timeline value: the default models
        an at-initiation decision that completes after the detector's fixed
        inference budget. Subclasses with a heavier pipeline (e.g. graph
        lookups across MED layers) override this to add per-event latency.
        """
        return np.full(len(frame), float(self.inference_budget_ms))

    inference_budget_ms: int = 0
