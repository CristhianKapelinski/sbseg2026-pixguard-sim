"""Experiment drivers.

Each function runs one experiment end to end on data actually obtained in this
run (the in-repo synthetic generator or a real, released third-party dataset
loaded through a checksum-verified adapter), logs the exact steps and outputs,
and returns a JSON-serializable result dict that the CLI writes under
``results/``. No number is produced unless the code here computes it.

In-repo source (the harness's own PIX layer, the scenario source):
  E1  Deadline metric is discriminative (the spine claim, a latency property).
  E2  Single-hop-tuned detectors degrade on multi-hop MED-2.0 and coercion.
  E4  Determinism: a re-run reproduces the stream byte-stable.

Cross-generator (independently-authored generators, the credibility anchor):
  E3  Real Tide HI/LI datasets: honest, sub-perfect, spread-out numbers.
  E5  Real pix-fraud-br: reproduce the prior-art tabular baselines + add the
      deadline metric.
  E6  Cross-generator transfer (train on one generator, test on another).
  E7  GPU GraphSAGE baseline across generators (graph-aware detector).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from pixguard_sim import data_io
from pixguard_sim.adapters import (
    PIX_FRAUD_BR_FEATURES,
    adapt_pix_fraud_br,
    adapt_tide,
)
from pixguard_sim.config import PipelineConfig
from pixguard_sim.detectors import (
    GraphSageDetector,
    RuleThresholdDetector,
    make_ml_detector,
    torch_available,
)
from pixguard_sim.detectors.base import Detector, measure_score_latency_ms
from pixguard_sim.generator import generate_events
from pixguard_sim.harness import evaluate_all, evaluate_detector, train_eval_split
from pixguard_sim.logging_setup import content_hash
from pixguard_sim.metrics import batch_metrics, pre_deadline_with_ci
from pixguard_sim.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _baseline_detectors(seed: int) -> list[Detector]:
    """Build the baseline detector set used across experiments.

    The detector family spans the rule floor plus logistic regression, random
    forest, and gradient boosting. The pre-deadline metric is fed the *measured*
    per-event scoring latency of each detector (see
    :func:`measure_score_latency_ms`); the ``inference_budget_ms`` argument is
    retained only as descriptive metadata and no longer drives the deadline
    metric. The ``gb_slow`` name is kept for continuity but the latency it is
    scored on is whatever its scoring call actually takes.
    """
    return [
        RuleThresholdDetector(),
        make_ml_detector("lr", seed, inference_budget_ms=50, name="lr_fast"),
        make_ml_detector("rf", seed, inference_budget_ms=50, name="rf_fast"),
        make_ml_detector("gb", seed, inference_budget_ms=5000, name="gb_slow"),
    ]


# ---------------------------------------------------------------------------
# In-repo PIX-layer experiments (spine + scenarios)
# ---------------------------------------------------------------------------


def experiment_e1(cfg: PipelineConfig) -> dict[str, Any]:
    """E1: the pre-deadline flag fraction separates detectors batch P/R cannot."""
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
    """E2: single-hop-tuned detectors degrade on multi-hop and coercion."""
    logger.info("=== E2: single-hop detectors degrade on new scenarios ===")
    frame = generate_events(cfg.generator)
    train_full, eval_ = train_eval_split(
        frame, seed=cfg.generator.seed, eval_fraction=0.4
    )
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
        make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50,
                         name="rf_single_hop"),
        make_ml_detector("gb", cfg.generator.seed, inference_budget_ms=50,
                         name="gb_single_hop"),
    ]
    reports = [
        evaluate_detector(d, single_hop, eval_, cfg.harness, seed=cfg.generator.seed)
        for d in detectors
    ]
    return {
        "experiment": "E2",
        "train_scope": "single_hop (legit + account_takeover layer 0)",
        "train_features": list(FEATURE_COLUMNS),
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


# ---------------------------------------------------------------------------
# Cross-generator experiments on real, released datasets
# ---------------------------------------------------------------------------


def _load_tide(cfg: PipelineConfig, ratio: str) -> pd.DataFrame:
    """Load, verify, subsample, and adapt a real Tide dataset split.

    Args:
        cfg: Pipeline config (data dir, subsample size, seed).
        ratio: ``"HI"`` (0.19% illicit) or ``"LI"`` (0.10% illicit).
    """
    root = data_io.data_root(cfg.data.data_dir)
    spec = next(f for f in data_io.TIDE_FILES if f.key == f"tide_tx_{ratio.lower()}")
    data_io.verify(spec, root)
    path = data_io.resolve(spec, root)
    cols = ["src", "dest", "edge_type", "amount", "is_fraudulent", "timestamp"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = df[df["edge_type"] == "transaction"]
    df = data_io.stratified_subsample(
        df, "is_fraudulent", cfg.data.tide_subsample_n, cfg.generator.seed
    )
    logger.info("Tide-%s loaded rows=%d source_hash=%s", ratio, len(df),
                content_hash(df.to_csv(index=False)))
    return adapt_tide(df, seed=cfg.generator.seed)


def _load_pix_fraud_br(cfg: PipelineConfig) -> pd.DataFrame:
    """Load, verify by row/fraud count, subsample, and adapt pix-fraud-br."""
    root = data_io.data_root(cfg.data.data_dir)
    path = root / data_io.PIX_FRAUD_BR_FILE.relpath
    if not path.exists():
        raise data_io.MissingSourceDataError(
            f"pix-fraud-br not found: {path} "
            f"(source: {data_io.PIX_FRAUD_BR_FILE.source_url})"
        )
    df = pd.read_parquet(path)
    n_fraud = int(df["fraude"].sum())
    logger.info("pix-fraud-br loaded rows=%d fraud=%d (provider: 2000000/15376)",
                len(df), n_fraud)
    df = data_io.stratified_subsample(
        df, "fraude", cfg.data.pix_subsample_n, cfg.generator.seed
    )
    return adapt_pix_fraud_br(df, seed=cfg.generator.seed)


def experiment_e3(cfg: PipelineConfig) -> dict[str, Any]:
    """E3: cross-generator credibility on the real released Tide HI/LI sets.

    Trains and evaluates the tabular baselines and the rule floor on each Tide
    split through a checksum-verified adapter. Under 0.10-0.19% illicit ratios
    the results vary across generators and task difficulty, which is the
    credibility signal: detectors do not trivially reach 1.00.
    """
    logger.info("=== E3: real Tide HI/LI cross-generator ===")
    out: dict[str, Any] = {"experiment": "E3", "splits": {}}
    for ratio in ("HI", "LI"):
        frame = _load_tide(cfg, ratio)
        detectors = [
            RuleThresholdDetector(),
            make_ml_detector("lr", cfg.generator.seed, inference_budget_ms=50,
                            name="lr_fast"),
            make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50,
                            name="rf_fast"),
            make_ml_detector("xgb", cfg.generator.seed, inference_budget_ms=50,
                            name="xgb_fast"),
        ]
        reports = evaluate_all(detectors, frame, cfg.harness, cfg.generator.seed)
        out["splits"][ratio] = {
            "n_events": int(len(frame)),
            "n_fraud": int(frame["is_fraud"].sum()),
            "fraud_rate": float(frame["is_fraud"].mean()),
            "detectors": [r.to_dict() for r in reports],
        }
    return out


def experiment_e5(cfg: PipelineConfig) -> dict[str, Any]:
    """E5: reproduce prior-art baselines on pix-fraud-br + add the deadline metric.

    Fits LR/RF/GB/XGBoost on pix-fraud-br's own engineered features (the same
    signal the prior art used) and reports PR-AUC against the published values
    as the cross-generator transfer test, then scores the deadline metric.
    """
    logger.info("=== E5: pix-fraud-br prior-art reproduction ===")
    frame = _load_pix_fraud_br(cfg)
    feats = tuple(FEATURE_COLUMNS) + PIX_FRAUD_BR_FEATURES
    detectors = [
        RuleThresholdDetector(),
        make_ml_detector("lr", cfg.generator.seed, inference_budget_ms=50,
                        name="lr_fast", features=feats),
        make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50,
                        name="rf_fast", features=feats),
        make_ml_detector("gb", cfg.generator.seed, inference_budget_ms=5000,
                        name="gb_slow", features=feats),
        make_ml_detector("xgb", cfg.generator.seed, inference_budget_ms=50,
                        name="xgb_fast", features=feats),
    ]
    reports = evaluate_all(detectors, frame, cfg.harness, cfg.generator.seed)
    return {
        "experiment": "E5",
        "n_events": int(len(frame)),
        "n_fraud": int(frame["is_fraud"].sum()),
        "fraud_rate": float(frame["is_fraud"].mean()),
        "features_used": list(feats),
        "published_baselines_prauc": {
            "xgboost": 0.8646, "note": "HF card, 200k validation sample, ROC-AUC 0.9950"
        },
        "deadline_sweep_ms": list(cfg.harness.deadline_sweep_ms),
        "detectors": [r.to_dict() for r in reports],
    }


def experiment_e6(cfg: PipelineConfig) -> dict[str, Any]:
    """E6: cross-generator transfer (train on one generator, test on another).

    A detector that only memorised one generator's quirks drops sharply when
    evaluated on a different generator. Trains a random forest on the in-repo
    PIX layer and on pix-fraud-br, then evaluates each on the other generator's
    held-out events, reporting the honest transfer gap (no 1.00s).
    """
    logger.info("=== E6: cross-generator transfer ===")
    inrepo = generate_events(cfg.generator)
    pfb = _load_pix_fraud_br(cfg)

    shared = list(FEATURE_COLUMNS)  # only the columns both generators share
    out: dict[str, Any] = {"experiment": "E6", "transfers": []}

    def _transfer(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        det = make_ml_detector(
            "rf", cfg.generator.seed, inference_budget_ms=50,
            name=f"rf_{name}", features=tuple(shared)
        )
        tr, _ = train_eval_split(train_df, seed=cfg.generator.seed)
        det.fit(tr)
        scores, score_total_ms, per_event_ms = measure_score_latency_ms(det, test_df)
        y = test_df["is_fraud"].to_numpy()
        m = batch_metrics(y, scores, cfg.harness.score_threshold,
                         n_bootstrap=cfg.harness.n_bootstrap, seed=cfg.generator.seed)
        dt = det.decision_latency_ms(test_df)
        pre = pre_deadline_with_ci(y, scores, dt, cfg.harness.score_threshold,
                                  cfg.harness.deadline_ms)
        out["transfers"].append({
            "name": name,
            "n_test": int(len(test_df)),
            "n_test_fraud": int(y.sum()),
            "measured_latency_ms": round(per_event_ms, 6),
            "measured_score_total_ms": round(score_total_ms, 3),
            "batch": m,
            "pre_deadline_1000ms": pre,
        })
        logger.info("E6 %-22s F1=%.3f PR-AUC=%.3f recall=%.3f latency=%.4fms/event",
                    name, m["f1"], m["pr_auc"], m["recall"], per_event_ms)

    # In-distribution references first, then the cross-generator transfers.
    _transfer("inrepo_to_inrepo", inrepo, train_eval_split(
        inrepo, seed=cfg.generator.seed)[1])
    _transfer("pfb_to_pfb", pfb, train_eval_split(pfb, seed=cfg.generator.seed)[1])
    _transfer("inrepo_to_pfb", inrepo, pfb)
    _transfer("pfb_to_inrepo", pfb, inrepo)
    return out


def experiment_e7(cfg: PipelineConfig) -> dict[str, Any]:
    """E7: GPU GraphSAGE baseline across the in-repo and Tide generators.

    Runs a graph-aware GraphSAGE detector on the available accelerator and
    compares it to the tabular random forest on identical inputs, reporting the
    measured wall-clock fit/score time so the latency is observed, not modeled.
    """
    logger.info("=== E7: GraphSAGE GPU baseline ===")
    if not torch_available():
        raise RuntimeError("PyTorch is required for the GNN baseline (E7)")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, Any] = {"experiment": "E7", "device": device, "datasets": {}}

    sources = {"inrepo": generate_events(cfg.generator)}
    try:
        sources["tide_hi"] = _load_tide(cfg, "HI")
    except (data_io.MissingSourceDataError, data_io.ChecksumMismatchError) as exc:
        logger.warning("E7 Tide split unavailable: %s", exc)

    for src_name, frame in sources.items():
        train, eval_ = train_eval_split(frame, seed=cfg.generator.seed)
        results = {}
        for det in (
            GraphSageDetector(cfg.generator.seed, inference_budget_ms=50,
                            name="gnn_sage"),
            make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50,
                            name="rf_fast"),
        ):
            t0 = time.time()
            det.fit(train)
            fit_s = time.time() - t0
            # Real measured per-event scoring latency (total scoring wall-time /
            # n events), the same observable used for the deadline metric.
            scores, score_total_ms, per_event_ms = measure_score_latency_ms(
                det, eval_)
            score_s = score_total_ms / 1000.0
            y = eval_["is_fraud"].to_numpy()
            m = batch_metrics(y, scores, cfg.harness.score_threshold,
                             n_bootstrap=cfg.harness.n_bootstrap, seed=cfg.generator.seed)
            dt = det.decision_latency_ms(eval_)
            pre = {
                int(d): pre_deadline_with_ci(
                    y, scores, dt, cfg.harness.score_threshold, d)
                for d in cfg.harness.deadline_sweep_ms
            }
            results[det.name] = {
                "batch": m,
                "fit_seconds": round(fit_s, 3),
                "score_seconds": round(score_s, 3),
                "measured_latency_ms": round(per_event_ms, 6),
                "measured_score_total_ms": round(score_total_ms, 3),
                "n_scored": int(len(eval_)),
                "pre_deadline_fraction": {str(k): v for k, v in pre.items()},
            }
            logger.info(
                "E7 %-10s on %-10s F1=%.3f PR-AUC=%.3f fit=%.2fs "
                "latency=%.4fms/event",
                det.name, src_name, m["f1"], m["pr_auc"], fit_s, per_event_ms)
        out["datasets"][src_name] = {
            "n_events": int(len(frame)),
            "n_fraud": int(frame["is_fraud"].sum()),
            "deadline_sweep_ms": list(cfg.harness.deadline_sweep_ms),
            "detectors": results,
        }
    return out


def _fraud_enriched_subsample(
    frame: pd.DataFrame, n: int, target_fraud: int, seed: int
) -> pd.DataFrame:
    """Draw a seeded subsample of ``n`` rows with a fixed fraud count.

    At the in-repo 1.2% base rate a 1000-row stratified draw carries only ~12
    fraud events, which is too few for a credible recall/PR-AUC or a tight
    pre-deadline CI (whose N is the fraud count). This draws ``target_fraud``
    true frauds and ``n - target_fraud`` legitimate events instead, so the LLM
    and tabular detectors are scored on the same labelled events with enough
    fraud to read the metrics honestly. The enriched prevalence is reported in
    the result so it is never mistaken for the operational base rate.
    """
    rng = np.random.default_rng(seed)
    pos = frame[frame["is_fraud"] == 1]
    neg = frame[frame["is_fraud"] == 0]
    n_pos = min(target_fraud, len(pos))
    n_neg = min(n - n_pos, len(neg))
    pos_idx = rng.choice(len(pos), size=n_pos, replace=False)
    neg_idx = rng.choice(len(neg), size=n_neg, replace=False)
    sub = pd.concat([pos.iloc[pos_idx], neg.iloc[neg_idx]])
    return sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def experiment_e8(cfg: PipelineConfig) -> dict[str, Any]:
    """E8: does a genuinely slow LLM detector miss a realistic deadline?

    Honest test of whether the pre-deadline flag fraction is meaningful with a
    real slow detector. An off-the-shelf small instruct LLM scores each event
    one transaction at a time (the way an online check would have to call it);
    its *measured* per-event generation latency is compared with the
    sub-microsecond latency of a tabular random forest on the SAME labelled,
    seeded, label-stratified subsample. If the LLM's measured latency exceeds a
    realistic pre-settlement deadline while the RF's does not, the deadline
    metric separates them despite comparable batch accuracy; otherwise it does
    not, and that is reported honestly.
    """
    import os

    from pixguard_sim.detectors.llm import LLMDetector

    # The LLM scores one event per forward generation, so the eval set is a
    # bounded subsample (not the full ~16k events). N, the target fraud count,
    # the model id, and the deadline sweep are read from the environment so the
    # driver can be pointed at a larger N or a stronger model without code
    # changes. The subsample is fraud-enriched (fixed fraud count) so recall,
    # PR-AUC, and the pre-deadline CI (whose N is the fraud count) are credible.
    n_sub = int(os.environ.get("PIXGUARD_E8_N", "1000"))
    target_fraud = int(os.environ.get("PIXGUARD_E8_FRAUD", "150"))
    model_id = os.environ.get("PIXGUARD_E8_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    sweep_env = os.environ.get("PIXGUARD_E8_DEADLINES", "200,1000,2000,5000")
    sweep = [int(x) for x in sweep_env.split(",")]
    logger.info("=== E8: slow LLM detector vs fast tabular, real latency ===")

    frame = generate_events(cfg.generator)
    train, eval_full = train_eval_split(frame, seed=cfg.generator.seed)
    eval_ = _fraud_enriched_subsample(
        eval_full, n_sub, target_fraud, cfg.generator.seed
    )
    y = eval_["is_fraud"].to_numpy()
    thr = cfg.harness.score_threshold
    logger.info(
        "E8 subsample N=%d (fraud=%d, enriched rate=%.4f) from eval pool=%d "
        "(operational base rate ~%.4f)",
        len(eval_), int(y.sum()), float(y.mean()), len(eval_full),
        float(eval_full["is_fraud"].mean()),
    )

    def _run(det: Detector) -> dict[str, Any]:
        det.fit(train)
        scores, total_ms, per_event_ms = measure_score_latency_ms(det, eval_)
        dt = det.decision_latency_ms(eval_)
        m = batch_metrics(y, scores, thr, n_bootstrap=cfg.harness.n_bootstrap,
                          seed=cfg.generator.seed)
        pre = {
            int(d): pre_deadline_with_ci(y, scores, dt, thr, d) for d in sweep
        }
        lat = getattr(det, "per_event_latencies_ms", None)
        lat_stats = None
        if lat is not None and len(lat) > 0:
            lat_stats = {
                "mean_ms": float(np.mean(lat)),
                "median_ms": float(np.median(lat)),
                "p95_ms": float(np.percentile(lat, 95)),
                "min_ms": float(np.min(lat)),
                "max_ms": float(np.max(lat)),
            }
        logger.info(
            "E8 %-12s F1=%.3f PR-AUC=%.3f recall=%.3f mean_lat=%.4fms/event",
            det.name, m["f1"], m["pr_auc"], m["recall"], per_event_ms,
        )
        return {
            "detector": det.name,
            "measured_mean_latency_ms": round(per_event_ms, 6),
            "measured_score_total_ms": round(total_ms, 3),
            "per_event_latency_stats_ms": lat_stats,
            # How many completions nothing could be read from. These fall back
            # to the threshold value, which counts as a flag, so a recall of
            # 1.000 next to a high count is the parser, not the model.
            "unparsed_completions": int(getattr(det, "unparsed_count", 0)),
            "batch": m,
            "pre_deadline_fraction": {str(k): v for k, v in pre.items()},
        }

    # Two LLM regimes plus the fast tabular reference, all on the SAME events.
    # The terse LLM emits a single probability token; the reasoning LLM thinks
    # step by step before concluding (the plausible way to make a small model
    # usable), which costs more generated tokens and therefore more latency.
    llm_terse = LLMDetector(
        model_id=model_id, name="llm_terse", reasoning=False, max_new_tokens=8
    )
    llm_reason = LLMDetector(
        model_id=model_id, name="llm_reasoning", reasoning=True,
        max_new_tokens=320,
    )
    rf = make_ml_detector("rf", cfg.generator.seed, inference_budget_ms=50,
                          name="rf_fast")

    detectors_out = [_run(llm_terse), _run(llm_reason), _run(rf)]

    return {
        "experiment": "E8",
        "question": (
            "With real measured latency, does a slow LLM detector miss a "
            "realistic pre-settlement deadline that a fast tabular detector "
            "meets?"
        ),
        "model_id": model_id,
        "subsample_is_fraud_enriched": True,
        "n_subsample": int(len(eval_)),
        "n_fraud_subsample": int(y.sum()),
        "enriched_fraud_rate_subsample": float(y.mean()),
        "operational_base_rate": float(eval_full["is_fraud"].mean()),
        "n_eval_pool": int(len(eval_full)),
        "score_threshold": thr,
        "deadline_sweep_ms": sweep,
        "detectors": detectors_out,
    }


EXPERIMENTS = {
    "E1": experiment_e1,
    "E2": experiment_e2,
    "E3": experiment_e3,
    "E4": experiment_e4,
    "E5": experiment_e5,
    "E6": experiment_e6,
    "E7": experiment_e7,
    "E8": experiment_e8,
}
