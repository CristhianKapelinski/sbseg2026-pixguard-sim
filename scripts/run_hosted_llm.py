#!/usr/bin/env python3
"""Score the E8 timing slice with hosted reasoning models.

E8 measures a small instruct model on the machine that runs the harness. An
institution that wants a language model in the loop is more likely to call a
hosted one, and then the decision latency is a network round trip plus whatever
the provider spends deliberating. This script scores the SAME events as E8,
drawn with the same seed, so the two are directly comparable, and writes
``results/e9_hosted.json``.

It also records the concurrency check: per-request latency issued serially
against the same requests issued with a bounded number in flight. The run is
shortened by the bound only because that check shows the bound does not change
the quantity being measured.

Usage: python scripts/run_hosted_llm.py [--models m1,m2] [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixguard_sim.config import PipelineConfig  # noqa: E402
from pixguard_sim.detectors.base import Detector  # noqa: E402
from pixguard_sim.detectors.llm_api import (  # noqa: E402
    HostedLLMDetector,
    measure_concurrency_effect,
)
from pixguard_sim.experiments import _fraud_enriched_subsample  # noqa: E402
from pixguard_sim.generator import generate_events  # noqa: E402
from pixguard_sim.harness import train_eval_split  # noqa: E402
from pixguard_sim.metrics import batch_metrics, pre_deadline_with_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("hosted")

# The regulator's ordinary authorization budget for the paying institution,
# 95th percentile (Manual de Tempos do Pix v7.0). Latency is reported against
# it so the number transfers to a reader on other hardware and other networks.
AUTH_BUDGET_MS = 1500.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="deepseek-v4-flash,deepseek-v4-pro")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--fraud", type=int, default=150)
    ap.add_argument("--in-flight", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="cap high enough that the model reaches a verdict")
    ap.add_argument("--out", default="results/e9_hosted.json")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    frame = generate_events(cfg.generator)
    _, eval_full = train_eval_split(frame, seed=cfg.generator.seed)
    eval_ = _fraud_enriched_subsample(eval_full, args.n, args.fraud, cfg.generator.seed)
    y = eval_["is_fraud"].to_numpy()
    thr = cfg.harness.score_threshold
    logger.info("slice: %d events, %d fraud", len(eval_), int(y.sum()))

    out: dict[str, object] = {
        "experiment": "E9",
        "question": "does a hosted reasoning model fit the authorization budget?",
        "n_subsample": len(eval_),
        "n_fraud_subsample": int(y.sum()),
        "operational_base_rate": float(eval_full["is_fraud"].mean()),
        "authorization_budget_ms": AUTH_BUDGET_MS,
        "detectors": [],
    }

    for model in args.models.split(","):
        det = HostedLLMDetector(model=model.strip(), max_in_flight=args.in_flight,
                                max_new_tokens=args.max_tokens)
        check = measure_concurrency_effect(det, eval_, n=8)
        logger.info("%s concurrency check: %s", det.name, check)
        scores = det.score(eval_)
        lat = det.per_event_latencies_ms
        stats = {
            "min_ms": float(lat.min()), "median_ms": float(np.median(lat)),
            "mean_ms": float(lat.mean()), "p95_ms": float(np.percentile(lat, 95)),
            "max_ms": float(lat.max()),
        }
        out["detectors"].append({
            "detector": det.name,
            "model": det.model,
            "batch": batch_metrics(y, scores, thr, cfg.harness.n_bootstrap,
                                   seed=cfg.generator.seed),
            "per_event_latency_stats_ms": stats,
            "budget_share_p95": stats["p95_ms"] / AUTH_BUDGET_MS,
            "new_tokens_median": int(np.median(det.new_tokens)) if det.new_tokens else None,
            "unparsed_completions": det.unparsed_count,
            "failed_requests": det.failed_count,
            "concurrency_check": check,
            # Latency is measured per event, so the deadline metric uses each
            # event's own round trip rather than a broadcast mean.
            "pre_deadline_1500ms": pre_deadline_with_ci(
                y, scores, lat, thr, int(AUTH_BUDGET_MS)
            ),
        })
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("%s done; results so far written to %s", det.name, p)

    logger.info("wrote %s", Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
