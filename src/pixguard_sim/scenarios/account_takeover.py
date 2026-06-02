"""Account-takeover scenario.

Maps onto the taxonomy's software- and remote-access-based group (the "ghost
hand" / remote-access scam): the attacker controls the victim's device and
initiates a transfer to a single mule account. The distinctive signals are a
remote-access session flag, a device change, a new payee, and an above-typical
amount, all at the originating layer (``med_layer == 0``).
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


def generate_account_takeover(
    rng: np.random.Generator,
    base: BaseGraph,
    cfg: GeneratorConfig,
    n_events: int,
    id_prefix: str = "ATO",
) -> list[PixEvent]:
    """Generate account-takeover fraud events.

    Each event is a single transfer from a victim account, under a remote
    session, to a candidate mule account.
    """
    accounts = list(base.profiles.keys())
    mules = base.mule_candidates() or accounts
    mu, sigma = cfg.fraud_amount_lognormal
    events: list[PixEvent] = []

    for i in range(n_events):
        victim = int(rng.choice(accounts))
        mule = int(rng.choice(mules))
        if mule == victim:
            mule = int(rng.choice(mules))
        t_init = int(rng.integers(0, 86_400_000))
        events.append(
            PixEvent(
                event_id=f"{id_prefix}{i:07d}",
                scenario="account_takeover",
                is_fraud=1,
                payer_account=victim,
                payee_account=mule,
                payee_dict_key=resolve_payee_key(base, mule),
                amount_brl=sample_amount(rng, mu, sigma),
                t_init_ms=t_init,
                t_settle_ms=settle_time(rng, t_init),
                med_layer=0,
                device_changed=device_changed(rng, cfg.device_change_prob_fraud),
                new_payee=1,
                payer_velocity_1h=int(rng.integers(2, 8)),
                is_remote_session=1,
                coercion_flag=0,
            )
        )
    return events
