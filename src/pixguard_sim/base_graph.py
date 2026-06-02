"""Lightweight base interaction graph.

Rather than depending on a heavy external AML simulator, PixGuard-Sim builds a
thin base graph itself, reusing the standard AMLSim/Tide idea of a scale-free
account-interaction topology over which transactions flow. The graph fixes,
per account, a usual device fingerprint, a small set of habitual counterparties
(its "known payees"), and a flag marking a fraction of accounts as candidate
mule accounts. The PIX-native event generators draw on this structure so that
legitimate activity is graph-consistent and fraud scenarios can route funds
through plausible mule paths.

The topology is generated with NetworkX's Barabasi-Albert model under a fixed
seed, making the whole base deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountProfile:
    """Static profile of one account in the base graph."""

    account_id: int
    dict_key: str
    usual_device: str
    known_payees: tuple[int, ...]
    is_mule_candidate: bool


@dataclass(frozen=True)
class BaseGraph:
    """A generated base interaction graph and its account profiles."""

    graph: nx.Graph
    profiles: dict[int, AccountProfile]

    @property
    def n_accounts(self) -> int:
        """Number of accounts in the graph."""
        return len(self.profiles)

    def mule_candidates(self) -> list[int]:
        """Return account ids flagged as candidate mule accounts."""
        return [a for a, p in self.profiles.items() if p.is_mule_candidate]


def _dict_key(rng: np.random.Generator, account_id: int) -> str:
    """Synthesize a DICT-style key for an account.

    The key emulates a DICT random-key (EVP) UUID-like token; it is purely
    synthetic and carries no real identifier.
    """
    token = rng.integers(0, 16**12, dtype=np.int64)
    return f"evp-{account_id:06d}-{token:012x}"


def build_base_graph(
    n_accounts: int,
    seed: int,
    mule_fraction: float = 0.08,
    attach: int = 3,
) -> BaseGraph:
    """Build a deterministic scale-free base interaction graph.

    Args:
        n_accounts: Number of accounts (nodes).
        seed: RNG seed controlling both the topology and the profiles.
        mule_fraction: Fraction of accounts marked as candidate mule accounts.
        attach: Barabasi-Albert attachment parameter (edges per new node).

    Returns:
        A :class:`BaseGraph` with one :class:`AccountProfile` per account.
    """
    if n_accounts <= attach:
        raise ValueError("n_accounts must exceed the attachment parameter")

    graph = nx.barabasi_albert_graph(n_accounts, attach, seed=seed)
    rng = np.random.default_rng(seed)

    n_mules = int(round(n_accounts * mule_fraction))
    mule_ids = set(rng.choice(n_accounts, size=n_mules, replace=False).tolist())

    profiles: dict[int, AccountProfile] = {}
    for account_id in graph.nodes():
        neighbors = tuple(sorted(graph.neighbors(account_id)))
        # Habitual payees: a deterministic subset of graph neighbors.
        if neighbors:
            k = min(len(neighbors), int(rng.integers(1, 4)))
            known = tuple(sorted(rng.choice(neighbors, size=k, replace=False).tolist()))
        else:
            known = ()
        device = f"dev-{rng.integers(0, 16**8, dtype=np.int64):08x}"
        profiles[account_id] = AccountProfile(
            account_id=account_id,
            dict_key=_dict_key(rng, account_id),
            usual_device=device,
            known_payees=known,
            is_mule_candidate=account_id in mule_ids,
        )

    logger.info(
        "base graph built: accounts=%d edges=%d mule_candidates=%d seed=%d",
        n_accounts,
        graph.number_of_edges(),
        n_mules,
        seed,
    )
    return BaseGraph(graph=graph, profiles=profiles)
