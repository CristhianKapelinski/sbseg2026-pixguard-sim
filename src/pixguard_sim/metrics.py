"""Evaluation metrics.

Provides the conventional batch metrics (precision, recall, F1, PR-AUC) and the
headline latency-aware metric of PixGuard-Sim: the *pre-deadline flag
fraction*. All metrics are computed with plain NumPy/scikit-learn and return
plain floats so results serialize cleanly to JSON.

Pre-deadline flag fraction (definition): among true-fraud events, the fraction
for which the detector both (i) emits a positive flag (score at or above the
threshold) and (ii) emits its decision within the configured decision window
``deadline_ms`` measured from each event's own initiation, where the per-event
decision latency is the detector's inference budget plus any per-event pipeline
overhead. A fraud caught only after the deadline is not actionable for
pre-transaction blocking and does not count.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def batch_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute precision, recall, F1, and PR-AUC.

    Args:
        y_true: Ground-truth labels (0/1).
        scores: Detector scores in ``[0, 1]``.
        threshold: Decision threshold for the binary metrics.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``pr_auc``.
    """
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pr_auc = (
        float(average_precision_score(y_true, scores))
        if y_true.sum() > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
    }


def pre_deadline_flag_fraction(
    y_true: np.ndarray,
    scores: np.ndarray,
    decision_time_ms: np.ndarray,
    threshold: float,
    deadline_ms: int,
) -> float:
    """Fraction of true frauds flagged before the decision deadline.

    Args:
        y_true: Ground-truth labels (0/1).
        scores: Detector scores in ``[0, 1]``.
        decision_time_ms: Per-event decision latency since initiation.
        threshold: Decision threshold turning a score into a flag.
        deadline_ms: The configurable pre-transaction decision deadline.

    Returns:
        Pre-deadline flag fraction over true-fraud events; 0.0 if no frauds.
    """
    fraud_mask = y_true == 1
    n_fraud = int(fraud_mask.sum())
    if n_fraud == 0:
        return 0.0
    flagged = scores >= threshold
    in_time = decision_time_ms <= deadline_ms
    caught = fraud_mask & flagged & in_time
    return float(caught.sum()) / float(n_fraud)


def recall_by_group(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict[object, float]:
    """Recall computed separately within each group label.

    Used for the recall-by-MED-layer and per-scenario breakdowns.
    """
    out: dict[object, float] = {}
    flagged = scores >= threshold
    for group in sorted(set(groups.tolist())):
        mask = (groups == group) & (y_true == 1)
        n = int(mask.sum())
        out[group] = float((mask & flagged).sum()) / n if n > 0 else 0.0
    return out
