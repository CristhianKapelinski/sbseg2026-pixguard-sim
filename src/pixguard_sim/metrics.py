"""Evaluation metrics with confidence intervals.

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

Every reported proportion (recall, pre-deadline flag fraction, per-group
recall) carries a Wilson 95% confidence interval so the population size N is
explicit and no point estimate is read as exact. PR-AUC and F1, which are not
single proportions, carry a seed-pinned bootstrap 95% confidence interval.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# Standard-normal quantile for a two-sided 95% interval.
_Z95 = 1.959963984540054


def wilson_interval(
    successes: int, n: int, z: float = _Z95
) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    The Wilson interval is preferred over the normal approximation for the
    small-count, near-boundary proportions that arise under heavy class
    imbalance (e.g. recall over a few hundred fraud events, or a fraction that
    is exactly 0 or 1).

    Args:
        successes: Number of successes (e.g. frauds caught).
        n: Number of trials (e.g. true-fraud events). Zero returns ``(0, 0)``.
        z: Standard-normal quantile (default: two-sided 95%).

    Returns:
        ``(lo, hi)`` clipped to ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def _proportion(successes: int, n: int) -> dict[str, float | int]:
    """Point estimate plus Wilson 95% CI and N for a proportion."""
    value = successes / n if n > 0 else 0.0
    lo, hi = wilson_interval(successes, n)
    return {"value": float(value), "ci_lo": lo, "ci_hi": hi, "n": int(n),
            "k": int(successes)}


def batch_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute precision, recall, F1, and PR-AUC with confidence intervals.

    Recall and precision are proportions, so they carry a Wilson 95% CI. F1 and
    PR-AUC are not single proportions, so they carry a seed-pinned bootstrap 95%
    CI over ``n_bootstrap`` resamples of the evaluation set.

    Args:
        y_true: Ground-truth labels (0/1).
        scores: Detector scores in ``[0, 1]``.
        threshold: Decision threshold for the binary metrics.
        n_bootstrap: Number of bootstrap resamples for F1/PR-AUC CIs.
        seed: RNG seed for the bootstrap, for determinism.

    Returns:
        Dict with point estimates (``precision``/``recall``/``f1``/``pr_auc``),
        Wilson CIs for recall/precision, bootstrap CIs for f1/pr_auc, and N.
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

    n_pos = int((y_true == 1).sum())
    n_pred_pos = int((y_pred == 1).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    recall_lo, recall_hi = wilson_interval(tp, n_pos)
    prec_lo, prec_hi = wilson_interval(tp, n_pred_pos) if n_pred_pos else (0.0, 0.0)

    f1_lo, f1_hi, prauc_lo, prauc_hi = _bootstrap_f1_prauc(
        y_true, scores, threshold, n_bootstrap, seed
    )

    return {
        "precision": float(precision),
        "precision_ci_lo": prec_lo,
        "precision_ci_hi": prec_hi,
        "recall": float(recall),
        "recall_ci_lo": recall_lo,
        "recall_ci_hi": recall_hi,
        "f1": float(f1),
        "f1_ci_lo": f1_lo,
        "f1_ci_hi": f1_hi,
        "pr_auc": pr_auc,
        "pr_auc_ci_lo": prauc_lo,
        "pr_auc_ci_hi": prauc_hi,
        "n_eval": int(len(y_true)),
        "n_fraud": n_pos,
    }


def _bootstrap_f1_prauc(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Seed-pinned bootstrap 95% CIs for F1 and PR-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or n_bootstrap <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    f1s: list[float] = []
    praucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        sc = scores[idx]
        if yt.sum() == 0:
            continue
        yp = (sc >= threshold).astype(int)
        _, _, f1b, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        f1s.append(float(f1b))
        praucs.append(float(average_precision_score(yt, sc)))
    if not f1s:
        return (0.0, 0.0, 0.0, 0.0)
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    pr_lo, pr_hi = np.percentile(praucs, [2.5, 97.5])
    return (float(f1_lo), float(f1_hi), float(pr_lo), float(pr_hi))


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
    return pre_deadline_flag_count(
        y_true, scores, decision_time_ms, threshold, deadline_ms
    )[2]


def pre_deadline_flag_count(
    y_true: np.ndarray,
    scores: np.ndarray,
    decision_time_ms: np.ndarray,
    threshold: float,
    deadline_ms: int,
) -> tuple[int, int, float]:
    """Return ``(caught, n_fraud, fraction)`` for the pre-deadline metric.

    Exposes the raw counts so a Wilson CI can be attached by the caller.
    """
    fraud_mask = y_true == 1
    n_fraud = int(fraud_mask.sum())
    if n_fraud == 0:
        return (0, 0, 0.0)
    flagged = scores >= threshold
    in_time = decision_time_ms <= deadline_ms
    caught = int((fraud_mask & flagged & in_time).sum())
    return (caught, n_fraud, caught / n_fraud)


def pre_deadline_with_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    decision_time_ms: np.ndarray,
    threshold: float,
    deadline_ms: int,
) -> dict[str, float | int]:
    """Pre-deadline flag fraction with a Wilson 95% CI and N."""
    caught, n_fraud, _ = pre_deadline_flag_count(
        y_true, scores, decision_time_ms, threshold, deadline_ms
    )
    return _proportion(caught, n_fraud)


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


def recall_by_group_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict[object, dict[str, float | int]]:
    """Per-group recall with a Wilson 95% CI and N for each group."""
    out: dict[object, dict[str, float | int]] = {}
    flagged = scores >= threshold
    for group in sorted(set(groups.tolist())):
        mask = (groups == group) & (y_true == 1)
        n = int(mask.sum())
        caught = int((mask & flagged).sum())
        out[group] = _proportion(caught, n)
    return out
