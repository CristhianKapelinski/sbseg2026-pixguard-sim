"""Detector plugin protocol.

The harness depends only on this abstract surface, never on a concrete model.
A detector is fit once on a labeled training frame, then asked to score an
evaluation frame. Scores are floats in ``[0, 1]`` where higher means more
fraud-like; the harness applies a configurable threshold for binary metrics
and uses the raw scores for PR-AUC.

A detector also reports its own *decision latency* via
:meth:`Detector.decision_latency_ms`: the time, on the event's relative
timeline, at which the detector would emit its decision for an event. This
latency is **measured**, not assumed: :func:`measure_score_latency_ms` times
the detector's real scoring call and records the mean per-event wall-clock
inference latency on :attr:`Detector.measured_latency_ms`, which
:meth:`decision_latency_ms` then returns. This is what lets the harness compute
the pre-deadline flag fraction from each detector's observed speed rather than a
hand-set budget.
"""

from __future__ import annotations

import time
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
        initiation (``t_init_ms``). If the detector recorded per-event latencies
        (e.g. an LLM scoring one event at a time), those are returned. Otherwise
        the mean per-event latency from the last
        :func:`measure_score_latency_ms` call is broadcast to all events.
        """
        per_event = getattr(self, "per_event_latencies_ms", None)
        if per_event is not None and len(per_event) == len(frame):
            return per_event.astype("float64")
        return np.full(len(frame), float(self.measured_latency_ms))

    # Mean per-event scoring wall-clock latency (ms), set by
    # ``measure_score_latency_ms`` after a real scoring call. ``inference_budget_ms``
    # is descriptive metadata for reports; it does not enter the pre-deadline
    # metric, which reads the measured latency above.
    measured_latency_ms: float = 0.0
    inference_budget_ms: int = 0


def measure_score_latency_ms(
    detector: Detector, frame: pd.DataFrame
) -> tuple[np.ndarray, float, float]:
    """Score a frame while timing the real per-event inference latency.

    Wraps the detector's own :meth:`Detector.score` in a wall-clock timer and
    records the mean per-event latency (total scoring time / number of events)
    on ``detector.measured_latency_ms`` so the harness's pre-deadline metric
    uses the latency the detector actually exhibited rather than an assumed
    budget.

    Args:
        detector: A fitted detector.
        frame: The evaluation frame to score.

    Returns:
        ``(scores, total_score_ms, per_event_ms)``.
    """
    n = max(len(frame), 1)
    t0 = time.perf_counter()
    scores = detector.score(frame)
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_event_ms = total_ms / n
    detector.measured_latency_ms = float(per_event_ms)
    return scores, float(total_ms), float(per_event_ms)
