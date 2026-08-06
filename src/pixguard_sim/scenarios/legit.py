"""Legitimate PIX traffic generator.

Legitimate transfers flow between an account and one of its habitual payees
(graph neighbors), at modest amounts, almost always from the account's usual
device and to a known payee. This provides the negative class against which
the fraud scenarios are scored, and gives the device/new-payee signals their
discriminative power without being trivially separable.
"""

from __future__ import annotations

import numpy as np

from pixguard_sim.base_graph import BaseGraph
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.scenarios._common import (
    device_changed,
    new_payee,
    payer_velocity,
    resolve_payee_key,
    sample_amount,
    settle_time,
)
from pixguard_sim.schema import PixEvent


def generate_legit(
    rng: np.random.Generator,
    base: BaseGraph,
    cfg: GeneratorConfig,
    n_events: int,
    id_prefix: str = "L",
) -> list[PixEvent]:
    """Generate ``n_events`` legitimate PIX transfers.

    Args:
        rng: Seeded RNG.
        base: Base interaction graph.
        cfg: Generator configuration.
        n_events: Number of legitimate events to produce.
        id_prefix: Prefix for generated event ids.

    Returns:
        A list of legitimate :class:`PixEvent` records.
    """
    accounts = list(base.profiles.keys())
    mu, sigma = cfg.legit_amount_lognormal
    events: list[PixEvent] = []

    for i in range(n_events):
        payer = int(rng.choice(accounts))
        profile = base.profiles[payer]
        if profile.known_payees and rng.random() < 0.85:
            payee = int(rng.choice(profile.known_payees))
            is_new = 0
        else:
            payee = int(rng.choice(accounts))
            is_new = new_payee(rng, cfg.new_payee_prob_legit)
        if payee == payer:
            payee = (payee + 1) % base.n_accounts

        t_init = int(rng.integers(0, 86_400_000))
        events.append(
            PixEvent(
                event_id=f"{id_prefix}{i:07d}",
                scenario="legit",
                is_fraud=0,
                payer_account=payer,
                payee_account=payee,
                payee_dict_key=resolve_payee_key(base, payee),
                amount_brl=sample_amount(rng, mu, sigma),
                t_init_ms=t_init,
                t_settle_ms=settle_time(rng, t_init),
                med_layer=0,
                device_changed=device_changed(rng, cfg.device_change_prob_legit),
                new_payee=is_new,
                payer_velocity_1h=payer_velocity(rng, 1.5),
                is_remote_session=0,
                coercion_flag=0,
            )
        )
    return events
