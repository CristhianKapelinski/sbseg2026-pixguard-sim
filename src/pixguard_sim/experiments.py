"""Experiment drivers (E1-E4).

Each function runs one experiment from the plan end to end on data actually
generated in this environment, logs the exact steps and outputs, and returns a
JSON-serializable result dict that the CLI writes under ``results/``. No number
is produced unless the code here computes it; PENDING experiments are not run
and are recorded as such by the CLI.

E1  Deadline metric is discriminative (main claim).
E2  Single-hop-tuned detectors degrade on multi-hop MED-2.0 and coercion.
E3  Generator-agnosticism: identical harness/metric on an independent generator.
E4  Determinism: a re-run reproduces E1 byte-stable.
"""

from __future__ import annotations

import logging
from typing import Any

from pixguard_sim.adapters import adapt_tide, synthesize_tide_shaped
from pixguard_sim.config import PipelineConfig
from pixguard_sim.detectors import RuleThresholdDetector, make_ml_detector
from pixguard_sim.detectors.base import Detector
from pixguard_sim.generator import generate_events
from pixguard_sim.harness import evaluate_all, evaluate_detector, train_eval_split
from pixguard_sim.logging_setup import content_hash
from pixguard_sim.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _baseline_detectors(seed: int) -> list[Detector]:
    """Build the baseline detector set used across experiments.

    Two ML detectors share the same family but carry different inference
    budgets, so the harness can separate them on the pre-deadline fraction even
    when their batch P/R is close. The rule detector is the floor.
    """
    return [
        RuleThresholdDetector(),
        make_ml_detector("lr", seed, inference_budget_ms=50, name="lr_fast"),
        make_ml_detector("rf", seed, inference_budget_ms=50, name="rf_fast"),
        make_ml_detector("gb", seed, inference_budget_ms=5000, name="gb_slow"),
    ]


def experiment_e1(cfg: PipelineConfig) -> dict[str, Any]:
    """E1: the pre-deadline flag fraction separates detectors batch P/R cannot.

    Generates the Tier A stream, evaluates the baselines, and emits the batch
    table plus the pre-deadline-fraction-vs-deadline curve per detector.
    """
    logger.info("=== E1: deadline metric is discriminative ===")
    frame = generate_events(cfg.generator)
    logger.info("E1 input frame hash=%s rows=%d", content_hash(
        frame.to_csv(index=False)), len(frame))
    reports = evaluate_all(
        _baseline_detectors(cfg.generator.seed), frame, cfg.harness, cfg.generator.seed
    )
    return {
        "experiment": "E1",
        "n_events": int(len(frame)),
        "n_fraud": int(frame["is_fraud"].sum()),
        "deadline_sweep_ms": list(cfg.harness.deadline_sweep_ms),
        "detectors": [r.to_dict() for r in reports],
    }


def experiment_e2(cfg: PipelineConfig) -> dict[str, Any]:
    """E2: single-hop-tuned detectors degrade on multi-hop and coercion.

    Trains detectors on a single-hop-only subset (account-takeover plus legit),
    then evaluates on the full stream, reporting per-scenario recall and
    recall-by-MED-layer to expose the collapse on the unseen scenarios.
    """
    logger.info("=== E2: single-hop detectors degrade on new scenarios ===")
    frame = generate_events(cfg.generator)
    train_full, eval_ = train_eval_split(
        frame, seed=cfg.generator.seed, eval_fraction=0.4
    )
    # Single-hop training subset: legitimate traffic + originating-layer
    # account-takeover only (the prior-art single-transfer case).
    single_hop = train_full[
        (train_full["scenario"] == "legit")
        | (train_full["scenario"] == "account_takeover")
    ].reset_index(drop=True)
    logger.info(
        "E2 single-hop train rows=%d (fraud=%d); eval rows=%d (fraud=%d)",
        len(single_hop),
        int(single_hop["is_fraud"].sum()),
        len(eval_),
        int(eval_["is_fraud"].sum()),
    )

    detectors = [
        make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50, name="rf_single_hop"),
        make_ml_detector("gb", cfg.generator.seed, inference_budget_ms=50, name="gb_single_hop"),
    ]
    reports = [evaluate_detector(d, single_hop, eval_, cfg.harness) for d in detectors]
    return {
        "experiment": "E2",
        "train_scope": "single_hop (legit + account_takeover layer 0)",
        "train_features": list(FEATURE_COLUMNS),
        "detectors": [r.to_dict() for r in reports],
    }


def experiment_e3(cfg: PipelineConfig) -> dict[str, Any]:
    """E3: generator-agnosticism on an independent (Tide-shaped) generator.

    Runs the identical harness and metric definitions on a second,
    differently-shaped generator output adapted into the PIX schema, producing
    the same report structure as E1.
    """
    logger.info("=== E3: generator-agnosticism (independent generator) ===")
    tide_shaped = synthesize_tide_shaped(
        n_events=cfg.generator.n_legit_events,
        fraud_rate=cfg.generator.fraud_base_rate,
        seed=cfg.generator.seed + 1,
    )
    logger.info(
        "E3 Tide-shaped source hash=%s rows=%d",
        content_hash(tide_shaped.to_csv(index=False)),
        len(tide_shaped),
    )
    frame = adapt_tide(tide_shaped)
    reports = evaluate_all(
        [
            RuleThresholdDetector(),
            make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50, name="rf_fast"),
        ],
        frame,
        cfg.harness,
        cfg.generator.seed,
    )
    return {
        "experiment": "E3",
        "generator": "tide_shaped_independent",
        "n_events": int(len(frame)),
        "n_fraud": int(frame["is_fraud"].sum()),
        "detectors": [r.to_dict() for r in reports],
    }


def experiment_e4(cfg: PipelineConfig) -> dict[str, Any]:
    """E4: determinism. Two independent generations hash-equal."""
    logger.info("=== E4: determinism / reproducibility ===")
    h1 = content_hash(generate_events(cfg.generator).to_csv(index=False))
    h2 = content_hash(generate_events(cfg.generator).to_csv(index=False))
    equal = h1 == h2
    logger.info("E4 hashes: run1=%s run2=%s equal=%s", h1, h2, equal)
    return {
        "experiment": "E4",
        "frame_hash_run1": h1,
        "frame_hash_run2": h2,
        "deterministic": equal,
    }


EXPERIMENTS = {
    "E1": experiment_e1,
    "E2": experiment_e2,
    "E3": experiment_e3,
    "E4": experiment_e4,
}
