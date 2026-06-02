"""Synthetic PIX event-stream generator (Tier A).

Orchestrates the base graph and the four scenario generators into a single
labeled, deterministic event stream. The number of fraud events is derived from
the configured base rate and split across scenarios by the configured weights;
legitimate events form the negative class. Every generated event is SYNTHETIC
and labeled with ground truth; this is never a real-world measurement.

The generator logs each scenario's event count and the resulting fraud base
rate so the run is fully auditable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pixguard_sim.base_graph import build_base_graph
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.scenarios import (
    generate_account_takeover,
    generate_coercion,
    generate_fake_med_refund,
    generate_legit,
    generate_mule_chain,
)
from pixguard_sim.schema import PixEvent, events_to_frame

logger = logging.getLogger(__name__)

# Scenarios whose unit of generation is a multi-event chain rather than a
# single event; their requested count is divided by an average chain length so
# the total fraud-event count tracks the configured base rate.
_AVG_CHAIN_LEN = {"mule_chain": 3.0, "fake_med_refund": 3.5}


def generate_events(cfg: GeneratorConfig) -> pd.DataFrame:
    """Generate the full labeled synthetic PIX event stream.

    Args:
        cfg: Generator configuration (seed, sizes, scenario weights, etc.).

    Returns:
        A DataFrame with the schema of :data:`pixguard_sim.schema.EVENT_COLUMNS`,
        shuffled into a single stream and re-indexed.
    """
    rng = np.random.default_rng(cfg.seed)
    base = build_base_graph(cfg.n_accounts, seed=cfg.seed)

    n_fraud_total = int(round(cfg.n_legit_events * cfg.fraud_base_rate /
                              (1.0 - cfg.fraud_base_rate)))
    weights = cfg.scenario_weights
    weight_sum = sum(weights.values())
    targets = {
        name: int(round(n_fraud_total * w / weight_sum))
        for name, w in weights.items()
    }

    events: list[PixEvent] = generate_legit(
        rng, base, cfg, cfg.n_legit_events
    )

    # account_takeover and coercion emit one event per unit; mule_chain and
    # fake_med_refund emit a chain, so divide their unit count accordingly.
    ato = generate_account_takeover(rng, base, cfg, targets["account_takeover"])
    crc = generate_coercion(rng, base, cfg, targets["coercion"])
    mc = generate_mule_chain(
        rng, base, cfg, max(1, int(targets["mule_chain"] / _AVG_CHAIN_LEN["mule_chain"]))
    )
    med = generate_fake_med_refund(
        rng,
        base,
        cfg,
        max(1, int(targets["fake_med_refund"] / _AVG_CHAIN_LEN["fake_med_refund"])),
    )
    events.extend(ato)
    events.extend(crc)
    events.extend(mc)
    events.extend(med)

    frame = events_to_frame(events)
    # Deterministic shuffle so scenarios are interleaved without leaking order.
    frame = frame.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    counts = frame["scenario"].value_counts().to_dict()
    n_fraud = int(frame["is_fraud"].sum())
    logger.info(
        "generated %d events (fraud=%d, rate=%.4f)",
        len(frame),
        n_fraud,
        n_fraud / len(frame),
    )
    for name in sorted(counts):
        logger.info("  scenario %-18s = %d events", name, counts[name])
    return frame
