"""Shared helpers for scenario generators.

Centralizes the small primitives every scenario needs (amount sampling, the
relative settlement timeline, DICT-key resolution of a payee, and the per-event
device/new-payee signal draws) so the four scenario modules stay focused on
their distinctive structure and share no duplicated logic.
"""

from __future__ import annotations

import numpy as np

from pixguard_sim.base_graph import BaseGraph

# PIX settles within seconds; we model an irrevocable-settlement offset on the
# event's relative timeline. This is the window inside which a pre-transaction
# alert must fire to be actionable.
SETTLE_OFFSET_MS: tuple[int, int] = (300, 2000)


def sample_amount(rng: np.random.Generator, mu: float, sigma: float) -> float:
    """Draw a positive BRL amount from a log-normal distribution."""
    return float(np.round(rng.lognormal(mean=mu, sigma=sigma), 2))


def settle_time(rng: np.random.Generator, t_init_ms: int) -> int:
    """Return the irrevocable-settlement time for a transfer initiated at t."""
    offset = int(rng.integers(SETTLE_OFFSET_MS[0], SETTLE_OFFSET_MS[1]))
    return t_init_ms + offset


def resolve_payee_key(base: BaseGraph, payee_account: int) -> str:
    """Resolve a payee account to its DICT key (DICT key resolution)."""
    return base.profiles[payee_account].dict_key


def device_changed(rng: np.random.Generator, prob: float) -> int:
    """Draw the device-change signal with the given probability."""
    return int(rng.random() < prob)


def new_payee(rng: np.random.Generator, prob: float) -> int:
    """Draw the new-payee signal with the given probability."""
    return int(rng.random() < prob)
