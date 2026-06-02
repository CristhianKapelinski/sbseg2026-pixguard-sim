"""PIX-native scenario generators.

Each module here emits :class:`~pixguard_sim.schema.PixEvent` records for one
fraud scenario, mapped onto a thematic group of the public PIX-fraud taxonomy
(documented as an explicit translation in DOCUMENTATION.md). The generators are
pure functions of an RNG and the base graph, so the whole stream is
deterministic given the master seed.
"""

from __future__ import annotations

from pixguard_sim.scenarios.account_takeover import generate_account_takeover
from pixguard_sim.scenarios.coercion import generate_coercion
from pixguard_sim.scenarios.fake_med_refund import generate_fake_med_refund
from pixguard_sim.scenarios.legit import generate_legit
from pixguard_sim.scenarios.mule_chain import generate_mule_chain

__all__ = [
    "generate_account_takeover",
    "generate_coercion",
    "generate_fake_med_refund",
    "generate_legit",
    "generate_mule_chain",
]
