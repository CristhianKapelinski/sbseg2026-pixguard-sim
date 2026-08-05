"""Adapters that normalize foreign generator outputs into the PIX schema.

The harness is generator-agnostic: it consumes the normalized event schema, so
any base generator can feed it through a thin adapter. Each adapter maps a
*real, released, third-party* dataset's own columns into the PIX-native event
schema without re-authoring its records, so a detector scored through an
adapter is tested on data produced independently of the harness. The adapters
synthesize only the PIX-native binary signals that the foreign schema lacks,
and they do so as a fixed, label-correlated noise process (not the same logic
any detector keys on), so the task stays non-trivial.

Three adapters are provided, one per independently-authored generator:

* :func:`adapt_tide` for the released Tide AML datasets (Zenodo),
* :func:`adapt_pix_fraud_br` for the released pix-fraud-br set (Hugging Face),
* :func:`adapt_amlsim` for AMLSim/IBM-style transaction CSVs.

All adapters expose the same numeric feature columns to detectors, so the
comparison across generators is like-for-like.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from pixguard_sim.schema import PixEvent, events_to_frame

logger = logging.getLogger(__name__)


def _account_id(value: object, modulo: int = 10**8) -> int:
    """Map an arbitrary account label to a stable non-negative integer id."""
    try:
        return int(value) % modulo
    except (TypeError, ValueError):
        return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % modulo


def _synth_signals(
    is_fraud: np.ndarray,
    seed: int,
    device_prob_fraud: float = 0.62,
    device_prob_legit: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize PIX-native binary signals a foreign schema does not carry.

    The signals (device change, new payee, velocity) are drawn as a fixed,
    label-correlated noise process so the adapted stream is realistically noisy
    rather than separable from the label alone.

    Every binary signal must be reachable from BOTH classes. Drawing one only
    for fraud makes it a deterministic label indicator: a detector reads the bit
    and inherits the label, and the score then measures the injection rather
    than the data. The device-change rates therefore default to the same pair
    the in-repo generator uses for its own events.

    Returns:
        ``(device_changed, new_payee, payer_velocity_1h)`` arrays.
    """
    rng = np.random.default_rng(seed)
    n = len(is_fraud)
    device_changed = np.where(
        is_fraud == 1,
        rng.random(n) < device_prob_fraud,
        rng.random(n) < device_prob_legit,
    ).astype(int)
    new_payee = np.where(
        is_fraud == 1, (rng.random(n) < 0.7), (rng.random(n) < 0.2)
    ).astype(int)
    velocity = rng.integers(0, 8, n)
    return device_changed, new_payee, velocity


# ---------------------------------------------------------------------------
# Tide adapter (released AML datasets, Zenodo 10.5281/zenodo.18804069)
# ---------------------------------------------------------------------------

# Tide's real released transaction columns.
TIDE_TX_COLUMNS: tuple[str, ...] = (
    "src",
    "dest",
    "edge_type",
    "amount",
    "is_fraudulent",
    "timestamp",
)


def adapt_tide(
    tx: pd.DataFrame, settle_offset_ms: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Map a real Tide transaction frame into the PIX event schema.

    Tide models laundering topology with timestamps and ground-truth
    ``is_fraudulent`` labels but no instant-payment semantics. We keep only the
    actual ``transaction`` edges, project Tide's own fields onto the PIX schema,
    and synthesize the PIX-native binary signals as a label-correlated noise
    process so detectors face a non-trivial, independently-authored stream.

    Args:
        tx: A frame with at least the columns of :data:`TIDE_TX_COLUMNS`.
        settle_offset_ms: Fixed settlement offset applied to the timeline.
        seed: Seed for the synthesized PIX-native signals.

    Returns:
        A frame in the PIX-native schema.

    Raises:
        KeyError: If a required Tide column is missing.
    """
    missing = [c for c in TIDE_TX_COLUMNS if c not in tx.columns]
    if missing:
        raise KeyError(f"missing Tide columns: {missing}")

    tx = tx[tx["edge_type"] == "transaction"].reset_index(drop=True)
    is_fraud = tx["is_fraudulent"].astype(bool).astype(int).to_numpy()
    ts = pd.to_datetime(tx["timestamp"], errors="coerce")
    t0 = ts.min()
    t_init_ms = ((ts - t0).dt.total_seconds().fillna(0.0) * 1000.0).astype("int64")
    t_init_ms = t_init_ms.to_numpy()

    device_changed, new_payee, velocity = _synth_signals(is_fraud, seed)
    payer = tx["src"].map(_account_id).to_numpy()
    payee = tx["dest"].map(_account_id).to_numpy()
    amount = tx["amount"].astype("float64").to_numpy()

    events = [
        PixEvent(
            event_id=f"TIDE{i:08d}",
            scenario="mule_chain" if is_fraud[i] else "legit",
            is_fraud=int(is_fraud[i]),
            payer_account=int(payer[i]),
            payee_account=int(payee[i]),
            payee_dict_key=f"evp-tide-{int(payee[i]):08d}",
            amount_brl=float(amount[i]),
            t_init_ms=int(t_init_ms[i]),
            t_settle_ms=int(t_init_ms[i]) + settle_offset_ms,
            med_layer=0,
            device_changed=int(device_changed[i]),
            new_payee=int(new_payee[i]),
            payer_velocity_1h=int(velocity[i]),
            is_remote_session=0,
            coercion_flag=0,
        )
        for i in range(len(tx))
    ]
    frame = events_to_frame(events)
    logger.info(
        "adapted Tide frame: events=%d fraud=%d rate=%.5f",
        len(frame),
        int(frame["is_fraud"].sum()),
        frame["is_fraud"].mean(),
    )
    return frame


# ---------------------------------------------------------------------------
# pix-fraud-br adapter (released set, Hugging Face andremessina/pix-fraud-br)
# ---------------------------------------------------------------------------

# Real released columns of pix-fraud-br (subset consumed; target is ``fraude``).
PIX_FRAUD_BR_COLUMNS: tuple[str, ...] = (
    "id_pagador",
    "id_recebedor",
    "tipo_transacao",
    "valor_brl",
    "saldo_anterior_recebedor",
    "razao_saldo_residual",
    "proporcao_valor_recebedor",
    "hora_dia",
    "horario_noturno",
    "fraude",
)

# Engineered numeric features available in pix-fraud-br beyond the shared
# schema columns. The harness scores tabular detectors on these when present so
# the reproduction of the prior-art baselines uses the same signal they did.
PIX_FRAUD_BR_FEATURES: tuple[str, ...] = (
    "valor_brl",
    "saldo_anterior_recebedor",
    "razao_saldo_residual",
    "proporcao_valor_recebedor",
    "hora_dia",
    "horario_noturno",
)


def adapt_pix_fraud_br(
    df: pd.DataFrame, settle_offset_ms: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Map the released pix-fraud-br frame into the PIX event schema.

    pix-fraud-br is a PaySim-derived, PIX-framed set with a ``fraude`` target
    and engineered balance-ratio features. We project its real Portuguese
    columns onto the PIX schema, derive a relative timeline from
    ``datetime_brasilia`` (or ``hora_dia`` when the datetime is absent), and
    carry its engineered numeric features through so the prior-art tabular
    baselines can be reproduced on identical signal.

    Args:
        df: A frame with the columns of :data:`PIX_FRAUD_BR_COLUMNS`.
        settle_offset_ms: Fixed settlement offset applied to the timeline.
        seed: Seed for the synthesized PIX-native binary signals.

    Returns:
        A frame in the PIX-native schema with the pix-fraud-br engineered
        features appended as extra columns.

    Raises:
        KeyError: If a required pix-fraud-br column is missing.
    """
    missing = [c for c in PIX_FRAUD_BR_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing pix-fraud-br columns: {missing}")

    df = df.reset_index(drop=True)
    is_fraud = df["fraude"].astype(int).to_numpy()
    if "datetime_brasilia" in df.columns:
        ts = pd.to_datetime(df["datetime_brasilia"], errors="coerce")
        t0 = ts.min()
        t_init_ms = (
            (ts - t0).dt.total_seconds().fillna(0.0) * 1000.0
        ).astype("int64").to_numpy()
    else:
        t_init_ms = (df["hora_dia"].astype("int64") * 3_600_000).to_numpy()

    device_changed, new_payee, velocity = _synth_signals(is_fraud, seed)
    payer = df["id_pagador"].map(_account_id).to_numpy()
    payee = df["id_recebedor"].map(_account_id).to_numpy()
    amount = df["valor_brl"].astype("float64").to_numpy()

    events = [
        PixEvent(
            event_id=f"PFB{i:08d}",
            scenario="account_takeover" if is_fraud[i] else "legit",
            is_fraud=int(is_fraud[i]),
            payer_account=int(payer[i]),
            payee_account=int(payee[i]),
            payee_dict_key=str(df["id_recebedor"].iloc[i]),
            amount_brl=float(amount[i]),
            t_init_ms=int(t_init_ms[i]),
            t_settle_ms=int(t_init_ms[i]) + settle_offset_ms,
            med_layer=0,
            device_changed=int(device_changed[i]),
            new_payee=int(new_payee[i]),
            payer_velocity_1h=int(velocity[i]),
            is_remote_session=0,
            coercion_flag=0,
        )
        for i in range(len(df))
    ]
    frame = events_to_frame(events)
    # Carry the engineered numeric features so the prior-art baselines see them.
    for col in PIX_FRAUD_BR_FEATURES:
        frame[col] = df[col].astype("float64").to_numpy()
    logger.info(
        "adapted pix-fraud-br frame: events=%d fraud=%d rate=%.5f",
        len(frame),
        int(frame["is_fraud"].sum()),
        frame["is_fraud"].mean(),
    )
    return frame


# ---------------------------------------------------------------------------
# AMLSim / IBM "Transactions for AML" adapter
# ---------------------------------------------------------------------------

# IBM "Transactions for AML" (AMLSim-based) real columns we consume.
AMLSIM_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "Account",
    "Account.1",
    "Amount Received",
    "Is Laundering",
)


def adapt_amlsim(
    df: pd.DataFrame, settle_offset_ms: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Map an IBM/AMLSim transaction frame into the PIX event schema.

    The IBM "Transactions for AML" set (AMLSim-based) labels each transfer with
    ``Is Laundering``. We project its real columns onto the PIX schema and
    synthesize the PIX-native binary signals, giving a third independently
    authored multi-hop generator on identical detector inputs.

    Args:
        df: A frame with the columns of :data:`AMLSIM_COLUMNS`.
        settle_offset_ms: Fixed settlement offset applied to the timeline.
        seed: Seed for the synthesized PIX-native binary signals.

    Returns:
        A frame in the PIX-native schema.

    Raises:
        KeyError: If a required column is missing.
    """
    missing = [c for c in AMLSIM_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing AMLSim columns: {missing}")

    df = df.reset_index(drop=True)
    is_fraud = df["Is Laundering"].astype(int).to_numpy()
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    t0 = ts.min()
    t_init_ms = (
        (ts - t0).dt.total_seconds().fillna(0.0) * 1000.0
    ).astype("int64").to_numpy()

    device_changed, new_payee, velocity = _synth_signals(is_fraud, seed)
    payer = df["Account"].map(_account_id).to_numpy()
    payee = df["Account.1"].map(_account_id).to_numpy()
    amount = df["Amount Received"].astype("float64").to_numpy()

    events = [
        PixEvent(
            event_id=f"AML{i:08d}",
            scenario="mule_chain" if is_fraud[i] else "legit",
            is_fraud=int(is_fraud[i]),
            payer_account=int(payer[i]),
            payee_account=int(payee[i]),
            payee_dict_key=f"evp-aml-{int(payee[i]):08d}",
            amount_brl=float(amount[i]),
            t_init_ms=int(t_init_ms[i]),
            t_settle_ms=int(t_init_ms[i]) + settle_offset_ms,
            med_layer=0,
            device_changed=int(device_changed[i]),
            new_payee=int(new_payee[i]),
            payer_velocity_1h=int(velocity[i]),
            is_remote_session=0,
            coercion_flag=0,
        )
        for i in range(len(df))
    ]
    frame = events_to_frame(events)
    logger.info(
        "adapted AMLSim frame: events=%d fraud=%d rate=%.5f",
        len(frame),
        int(frame["is_fraud"].sum()),
        frame["is_fraud"].mean(),
    )
    return frame
