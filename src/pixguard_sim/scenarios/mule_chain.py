"""Mule-chain scenario.

A social-engineering or takeover-originated transfer that is then layered
through a short chain of mule accounts to frustrate recovery. This is the
generic laundering-topology analog (fan-out through mules). Unlike the
fake-MED-refund scenario, the chain here is the dispersal of the original
funds rather than a refund-flow abuse, and the originating event already shows
strong new-payee and amount signals; later hops decay in signal strength.
"""

from __future__ import annotations

import numpy as np

from pixguard_sim.base_graph import BaseGraph
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.scenarios._common import (
    device_changed,
    resolve_payee_key,
    sample_amount,
    settle_time,
)
from pixguard_sim.schema import PixEvent


def generate_mule_chain(
    rng: np.random.Generator,
    base: BaseGraph,
    cfg: GeneratorConfig,
    n_chains: int,
    id_prefix: str = "MC",
) -> list[PixEvent]:
    """Generate mule-chain fraud events.

    Args:
        rng: Seeded RNG.
        base: Base interaction graph.
        cfg: Generator configuration.
        n_chains: Number of distinct chains to generate. Each chain emits
            one originating event plus a random number of dispersal hops.
        id_prefix: Prefix for generated event ids.

    Returns:
        A flat list of :class:`PixEvent` records across all chains.
    """
    accounts = list(base.profiles.keys())
    mules = base.mule_candidates() or accounts
    mu, sigma = cfg.fraud_amount_lognormal
    events: list[PixEvent] = []
    event_counter = 0

    for _ in range(n_chains):
        victim = int(rng.choice(accounts))
        depth = int(rng.integers(2, 5))  # dispersal hops (2..4)
        amount = sample_amount(rng, mu, sigma)
        t = int(rng.integers(0, 86_400_000))
        payer = victim
        for layer in range(depth):
            mule = int(rng.choice(mules))
            if mule == payer:
                mule = int(rng.choice(mules))
            # Amount decays across hops as funds are split.
            hop_amount = round(amount * (0.8**layer), 2)
            events.append(
                PixEvent(
                    event_id=f"{id_prefix}{event_counter:07d}",
                    scenario="mule_chain",
                    is_fraud=1,
                    payer_account=payer,
                    payee_account=mule,
                    payee_dict_key=resolve_payee_key(base, mule),
                    amount_brl=hop_amount,
                    t_init_ms=t,
                    t_settle_ms=settle_time(rng, t),
                    med_layer=layer,
                    device_changed=device_changed(
                        rng,
                        cfg.device_change_prob_fraud if layer == 0 else 0.1,
                    ),
                    new_payee=1 if layer == 0 else int(rng.random() < 0.5),
                    payer_velocity_1h=int(rng.integers(3, 10)),
                    is_remote_session=1 if layer == 0 else 0,
                    coercion_flag=0,
                )
            )
            event_counter += 1
            delay = int(
                rng.integers(cfg.inter_hop_delay_ms[0], cfg.inter_hop_delay_ms[1])
            )
            t = t + delay
            payer = mule
    return events
