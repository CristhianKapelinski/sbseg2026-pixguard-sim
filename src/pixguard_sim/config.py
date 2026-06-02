"""Configuration objects for the PixGuard-Sim pipeline.

This module centralizes every tunable parameter so that no behavior is
hardcoded in the generators, detectors, or harness. Configs are plain frozen
dataclasses that are trivially serializable (for logging the resolved config
and content-hashing it) and constructible from a YAML/JSON-free dictionary so
the CLI can load them from a versioned config file without extra dependencies.

Design note (methodology baked in here): the decision ``deadline`` is a
researcher parameter, not a legal latency. No Brazilian regulation fixes a
latency in seconds; the mandate is "block as soon as identified". The harness
therefore evaluates over a *range* of deadlines, and this config holds the
default sweep used by the experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# Canonical scenario identifiers used across the package. They map onto the
# four thematic groups of the public PIX-fraud taxonomy (see DOCUMENTATION.md
# for the explicit, justified translation).
SCENARIOS: tuple[str, ...] = (
    "mule_chain",
    "account_takeover",
    "fake_med_refund",
    "coercion",
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters of the synthetic PIX event generator.

    Attributes:
        seed: Master RNG seed; the whole pipeline is deterministic given it.
        n_accounts: Number of accounts in the base interaction graph.
        n_legit_events: Number of legitimate (non-fraud) PIX events to draw.
        fraud_base_rate: Target fraction of fraud events among all events.
            Anchored to the order of magnitude reported by the open PIX
            generator prior art (sub-1%); see DOCUMENTATION.md.
        scenario_weights: Relative frequency of each fraud scenario.
        med_max_depth: Maximum number of mule layers in a multi-hop MED-2.0
            refund chain. Anchored to the regulator's five-layer figure.
        inter_hop_delay_ms: (min, max) delay between successive hops of a
            multi-hop chain, in milliseconds of the event's relative timeline.
        legit_amount_lognormal: (mu, sigma) of the log-normal legitimate
            amount distribution, in BRL.
        fraud_amount_lognormal: (mu, sigma) of the log-normal fraud amount
            distribution, in BRL (heavier mean than legitimate).
        device_change_prob_fraud: Probability a fraud event shows a device
            change relative to the account's usual device.
        device_change_prob_legit: Same probability for a legitimate event.
        new_payee_prob_fraud: Probability the payee DICT key is new to the
            payer for a fraud event.
        new_payee_prob_legit: Same probability for a legitimate event.
    """

    seed: int = 20260202
    n_accounts: int = 4000
    n_legit_events: int = 40000
    fraud_base_rate: float = 0.012
    scenario_weights: dict[str, float] = field(
        default_factory=lambda: {
            "mule_chain": 0.30,
            "account_takeover": 0.30,
            "fake_med_refund": 0.20,
            "coercion": 0.20,
        }
    )
    med_max_depth: int = 5
    inter_hop_delay_ms: tuple[int, int] = (200, 5000)
    legit_amount_lognormal: tuple[float, float] = (4.6, 1.0)
    fraud_amount_lognormal: tuple[float, float] = (6.2, 1.1)
    device_change_prob_fraud: float = 0.62
    device_change_prob_legit: float = 0.06
    new_payee_prob_fraud: float = 0.88
    new_payee_prob_legit: float = 0.18


@dataclass(frozen=True)
class HarnessConfig:
    """Parameters of the latency-aware evaluation harness.

    Attributes:
        deadline_ms: Default single decision deadline, in milliseconds of the
            event's relative timeline, used when a single value is needed.
        deadline_sweep_ms: Ordered range of deadlines over which the
            pre-deadline flag fraction is reported (the headline curve).
        score_threshold: Decision threshold applied to a detector's score to
            turn it into a binary flag for precision/recall.
    """

    deadline_ms: int = 1000
    deadline_sweep_ms: tuple[int, ...] = (
        100,
        250,
        500,
        1000,
        2000,
        5000,
        10000,
    )
    score_threshold: float = 0.5


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level config bundling generator and harness settings."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the resolved configuration."""
        return asdict(self)

    def to_json(self) -> str:
        """Return a stable, sorted JSON string for logging and hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a (possibly partial) plain dictionary."""
        gen = GeneratorConfig(**data.get("generator", {}))
        har = HarnessConfig(**_coerce_harness(data.get("harness", {})))
        return cls(generator=gen, harness=har)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a JSON file; fields omitted take defaults."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def with_overrides(self, **generator_overrides: Any) -> PipelineConfig:
        """Return a copy with the given generator fields overridden."""
        return replace(
            self, generator=replace(self.generator, **generator_overrides)
        )


def _coerce_harness(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce JSON lists into the tuples the harness config expects."""
    out = dict(data)
    if "deadline_sweep_ms" in out:
        out["deadline_sweep_ms"] = tuple(int(x) for x in out["deadline_sweep_ms"])
    return out
