"""Latency-aware evaluation harness (the core contribution).

The harness is detector- and generator-agnostic: it consumes any labeled event
frame in the PIX-native schema and any object satisfying the
:class:`~pixguard_sim.detectors.base.Detector` protocol. For each detector it
reports the conventional batch metrics, the headline pre-deadline flag fraction
over a sweep of deadlines, the per-scenario breakdown, and the recall-by-MED-
layer breakdown for the multi-hop scenarios.

A train/evaluation split is drawn deterministically from the master seed. The
harness never exposes the label, scenario, or identifiers to a detector; the
detector sees only the schema's feature columns and the event timeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pixguard_sim.config import HarnessConfig
from pixguard_sim.detectors.base import Detector
from pixguard_sim.metrics import (
    batch_metrics,
    pre_deadline_with_ci,
    recall_by_group_ci,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectorReport:
    """Evaluation report for a single detector.

    Every proportion (``pre_deadline_fraction``, per-scenario recall,
    recall-by-MED-layer) is stored with a Wilson 95% CI and its N; the batch
    block carries Wilson CIs on recall/precision and bootstrap CIs on F1/PR-AUC.
    """

    detector: str
    inference_budget_ms: int
    batch: dict[str, float]
    pre_deadline_fraction: dict[int, dict[str, float | int]]
    per_scenario: dict[str, dict[str, float]]
    recall_by_med_layer: dict[int, dict[str, float | int]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the report."""
        return {
            "detector": self.detector,
            "inference_budget_ms": self.inference_budget_ms,
            "batch": self.batch,
            "pre_deadline_fraction": {
                str(k): v for k, v in self.pre_deadline_fraction.items()
            },
            "per_scenario": self.per_scenario,
            "recall_by_med_layer": {
                str(k): v for k, v in self.recall_by_med_layer.items()
            },
        }


def train_eval_split(
    frame: pd.DataFrame, seed: int, eval_fraction: float = 0.4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split a frame into train and evaluation sets.

    The split is stratified on the label so both sets carry fraud.
    """
    rng = np.random.default_rng(seed)
    train_parts, eval_parts = [], []
    for label in (0, 1):
        subset = frame[frame["is_fraud"] == label]
        idx = rng.permutation(len(subset))
        cut = int(len(subset) * (1.0 - eval_fraction))
        train_parts.append(subset.iloc[idx[:cut]])
        eval_parts.append(subset.iloc[idx[cut:]])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed)
    eval_ = pd.concat(eval_parts).sample(frac=1.0, random_state=seed)
    return train.reset_index(drop=True), eval_.reset_index(drop=True)


def evaluate_detector(
    detector: Detector,
    train: pd.DataFrame,
    eval_: pd.DataFrame,
    cfg: HarnessConfig,
    seed: int = 0,
) -> DetectorReport:
    """Fit a detector on ``train`` and evaluate it on ``eval_``.

    Args:
        detector: The detector to fit and score.
        train: Training events.
        eval_: Evaluation events.
        cfg: Harness configuration (deadlines, threshold, bootstrap count).
        seed: Seed for the bootstrap confidence intervals, for determinism.

    Returns:
        A :class:`DetectorReport` with batch metrics, the pre-deadline flag
        fraction over the configured deadline sweep, the per-scenario
        breakdown, and the recall-by-MED-layer breakdown, all with CIs.
    """
    detector.fit(train)
    scores = detector.score(eval_)
    decision_time = detector.decision_latency_ms(eval_)
    y_true = eval_["is_fraud"].to_numpy()
    thr = cfg.score_threshold

    batch = batch_metrics(y_true, scores, thr, n_bootstrap=cfg.n_bootstrap, seed=seed)

    pre_deadline = {
        int(d): pre_deadline_with_ci(y_true, scores, decision_time, thr, d)
        for d in cfg.deadline_sweep_ms
    }

    per_scenario: dict[str, dict[str, float]] = {}
    scenarios = eval_["scenario"].to_numpy()
    for scenario in sorted(set(scenarios.tolist())):
        if scenario == "legit":
            continue
        mask = scenarios == scenario
        sc_recall = recall_by_group_ci(
            y_true[mask], scores[mask], np.zeros(mask.sum()), thr
        ).get(0.0, {"value": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0, "k": 0})
        sc_pre = pre_deadline_with_ci(
            y_true[mask],
            scores[mask],
            decision_time[mask],
            thr,
            cfg.deadline_ms,
        )
        per_scenario[scenario] = {
            "recall": sc_recall,
            "pre_deadline_fraction": sc_pre,
            "n_fraud": int((y_true[mask] == 1).sum()),
        }

    med_mask = eval_["scenario"].to_numpy() == "fake_med_refund"
    recall_layer = recall_by_group_ci(
        y_true[med_mask],
        scores[med_mask],
        eval_["med_layer"].to_numpy()[med_mask],
        thr,
    )
    recall_by_layer = {int(k): v for k, v in recall_layer.items()}

    report = DetectorReport(
        detector=detector.name,
        inference_budget_ms=detector.inference_budget_ms,
        batch=batch,
        pre_deadline_fraction=pre_deadline,
        per_scenario=per_scenario,
        recall_by_med_layer=recall_by_layer,
    )
    logger.info(
        "evaluated %-26s P=%.3f R=%.3f F1=%.3f PR-AUC=%.3f budget=%dms",
        detector.name,
        batch["precision"],
        batch["recall"],
        batch["f1"],
        batch["pr_auc"],
        detector.inference_budget_ms,
    )
    return report


def evaluate_all(
    detectors: list[Detector],
    frame: pd.DataFrame,
    cfg: HarnessConfig,
    seed: int,
) -> list[DetectorReport]:
    """Evaluate a list of detectors on one shared train/eval split."""
    train, eval_ = train_eval_split(frame, seed=seed)
    logger.info(
        "split: train=%d (fraud=%d) eval=%d (fraud=%d)",
        len(train),
        int(train["is_fraud"].sum()),
        len(eval_),
        int(eval_["is_fraud"].sum()),
    )
    return [evaluate_detector(d, train, eval_, cfg, seed=seed) for d in detectors]
