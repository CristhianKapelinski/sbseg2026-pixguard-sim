"""Coercion (duress-initiated transfer) scenario.

Maps onto the taxonomy's physical-interaction group (robbery, express
kidnapping, extortion), where the victim is physically coerced into initiating
a transfer from their own device. The defining difficulty for detectors is
that, unlike account-takeover, coercion uses the victim's *own* device and a
genuine session: there is typically no device change and no remote-access
flag. The only distinctive signals are a coercion indicator (e.g. duress under
behavioral biometrics), an unusual amount, and a new payee. This scenario is
the second PIX-native scenario absent from open prior artifacts.
"""

from __future__ import annotations

import numpy as np

from pixguard_sim.base_graph import BaseGraph
from pixguard_sim.config import GeneratorConfig
from pixguard_sim.scenarios._common import (
    resolve_payee_key,
    sample_amount,
    settle_time,
    payer_velocity,
)
from pixguard_sim.schema import PixEvent


def generate_coercion(
    rng: np.random.Generator,
    base: BaseGraph,
    cfg: GeneratorConfig,
    n_events: int,
    id_prefix: str = "CRC",
) -> list[PixEvent]:
    """Generate coercion fraud events.

    Each event is a single high-value transfer from the victim's own device
    (no device change, no remote session) to a mule account, carrying the
    coercion flag.
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
                scenario="coercion",
                is_fraud=1,
                payer_account=victim,
                payee_account=mule,
                payee_dict_key=resolve_payee_key(base, mule),
                # Coerced transfers skew to higher amounts (drain the account).
                amount_brl=sample_amount(rng, mu + 0.3, sigma),
                t_init_ms=t_init,
                t_settle_ms=settle_time(rng, t_init),
                med_layer=0,
                device_changed=0,
                new_payee=1,
                payer_velocity_1h=payer_velocity(rng, 1.0),
                is_remote_session=0,
                coercion_flag=1,
            )
        )
    return events
