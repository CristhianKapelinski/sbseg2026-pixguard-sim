"""Unit tests for the synthetic generator and base graph determinism."""

from __future__ import annotations

from pixguard_sim.base_graph import build_base_graph
from pixguard_sim.config import SCENARIOS, GeneratorConfig
from pixguard_sim.generator import generate_events
from pixguard_sim.logging_setup import content_hash
from pixguard_sim.schema import EVENT_COLUMNS

_SMALL = GeneratorConfig(n_accounts=300, n_legit_events=1500, seed=7)


def test_base_graph_is_deterministic() -> None:
    g1 = build_base_graph(200, seed=11)
    g2 = build_base_graph(200, seed=11)
    assert g1.n_accounts == g2.n_accounts == 200
    assert g1.profiles[5].dict_key == g2.profiles[5].dict_key
    assert g1.profiles[5].usual_device == g2.profiles[5].usual_device


def test_generate_schema_and_label_presence() -> None:
    frame = generate_events(_SMALL)
    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert frame["is_fraud"].isin([0, 1]).all()
    assert frame["is_fraud"].sum() > 0
    # All canonical fraud scenarios are present.
    present = set(frame["scenario"].unique())
    for scenario in SCENARIOS:
        assert scenario in present


def test_generation_is_deterministic() -> None:
    h1 = content_hash(generate_events(_SMALL).to_csv(index=False))
    h2 = content_hash(generate_events(_SMALL).to_csv(index=False))
    assert h1 == h2


def test_med_layers_reach_depth() -> None:
    frame = generate_events(_SMALL)
    med = frame[frame["scenario"] == "fake_med_refund"]
    # Multi-hop chains must produce layers beyond the originating layer.
    assert med["med_layer"].max() >= 1


def test_coercion_uses_own_device_no_remote() -> None:
    frame = generate_events(_SMALL)
    crc = frame[frame["scenario"] == "coercion"]
    assert (crc["device_changed"] == 0).all()
    assert (crc["is_remote_session"] == 0).all()
    assert (crc["coercion_flag"] == 1).all()


def test_fraud_base_rate_in_range() -> None:
    frame = generate_events(_SMALL)
    rate = frame["is_fraud"].mean()
    # Within a tolerance band of the configured base rate (chains inflate it).
    assert 0.005 <= rate <= 0.05
