"""Fake-MED multi-hop refund scenario (MED-2.0).

Maps onto the taxonomy's refund/benefit/urgency social-engineering group (the
"false scheduling" and "wrong Pix" refund scams). The attacker induces a
refund and then traces the funds through up to ``med_max_depth`` subsequent
account layers, mirroring the MED-2.0 refund-tracing depth the regulator
defines (up to five layers). The distinctive structure is the deep refund
chain: each layer is a separate event with a monotonically increasing
``med_layer`` index and an inter-hop delay, so the harness can report recall by
MED layer.

This scenario is one of the two PIX-native scenarios that no open prior
artifact covers; the single-hop open generator models only the first transfer.
"""

from __future__ import annotations

import numpy as np

from pixguard_sim.base_graph import BaseGraph
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.scenarios._common import (
    device_changed,
    payer_velocity,
    resolve_payee_key,
    sample_amount,
    settle_time,
)
from pixguard_sim.schema import PixEvent


def generate_fake_med_refund(
    rng: np.random.Generator,
    base: BaseGraph,
    cfg: GeneratorConfig,
    n_chains: int,
    id_prefix: str = "MED",
) -> list[PixEvent]:
    """Generate fake-MED multi-hop refund chains.

    Each chain emits one event per refund layer, from layer 0 (the induced
    refund) through a depth sampled up to ``cfg.med_max_depth``. Signal
    strength decays with depth: the deeper layers look increasingly like
    ordinary transfers, which is precisely what makes deep-layer recall hard
    and motivates the MED-layer breakdown.
    """
    accounts = list(base.profiles.keys())
    mules = base.mule_candidates() or accounts
    mu, sigma = cfg.fraud_amount_lognormal
    events: list[PixEvent] = []
    event_counter = 0

    for _ in range(n_chains):
        victim = int(rng.choice(accounts))
        depth = int(rng.integers(2, cfg.med_max_depth + 1))  # 2..med_max_depth
        amount = sample_amount(rng, mu, sigma)
        t = int(rng.integers(0, 86_400_000))
        payer = victim
        for layer in range(depth):
            mule = int(rng.choice(mules))
            if mule == payer:
                mule = int(rng.choice(mules))
            # Signal decay: device-change and new-payee probabilities fall as
            # the chain deepens, so deep layers are harder to flag.
            decay = 0.7**layer
            hop_amount = round(amount * (0.85**layer), 2)
            events.append(
                PixEvent(
                    event_id=f"{id_prefix}{event_counter:07d}",
                    scenario="fake_med_refund",
                    is_fraud=1,
                    payer_account=payer,
                    payee_account=mule,
                    payee_dict_key=resolve_payee_key(base, mule),
                    amount_brl=hop_amount,
                    t_init_ms=t,
                    t_settle_ms=settle_time(rng, t),
                    med_layer=layer,
                    device_changed=device_changed(
                        rng, cfg.device_change_prob_fraud * decay
                    ),
                    new_payee=int(rng.random() < cfg.new_payee_prob_fraud * decay),
                    payer_velocity_1h=payer_velocity(rng, 2.9),
                    is_remote_session=0,
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
