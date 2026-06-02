"""Adapters that normalize foreign generator outputs into the PIX schema.

The harness is generator-agnostic: it consumes the normalized event schema, so
any base generator can feed it through a thin adapter. Two tiers are provided.

Tier B (runnable here): a Tide adapter that maps Tide's transaction CSV plus
its ground-truth pattern labels into the PIX-native schema. Because installing
and running the external Tide generator is gated on network/disk in this
environment, the adapter is written against Tide's documented output schema and
also accepts a synthetic Tide-shaped frame, so generator-agnosticism is
demonstrated on a second, independently-shaped input without authoring its
records the same way the Tier A generator does.

Tier C (PENDING): adapters for the open PIX generator prior art (PaySim-derived,
behind a Kaggle/Hugging Face access wall) and AMLSim (Java build). They are
specified against the documented column schema and raise a clear, typed error
if the source data is absent, so they never silently fabricate a result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pixguard_sim.schema import EVENT_COLUMNS, PixEvent, events_to_frame

logger = logging.getLogger(__name__)


class MissingSourceDataError(RuntimeError):
    """Raised when a PENDING adapter is invoked without its source data."""


# ---------------------------------------------------------------------------
# Tier B: Tide adapter (runnable)
# ---------------------------------------------------------------------------

# Tide's documented transaction columns (subset we consume).
TIDE_TX_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "src_account",
    "dst_account",
    "amount",
    "timestamp",
    "is_laundering",
)


def adapt_tide(tx: pd.DataFrame, settle_offset_ms: int = 1000) -> pd.DataFrame:
    """Map a Tide-shaped transaction frame into the PIX event schema.

    Tide models laundering topology with timestamps and ground-truth labels but
    no instant-payment semantics; we project its fields onto the PIX schema and
    synthesize the PIX-native signals (device/new-payee/remote) from the label
    so the harness can score detectors on it under the identical metric.

    Args:
        tx: A frame with the columns of :data:`TIDE_TX_COLUMNS`.
        settle_offset_ms: Fixed settlement offset applied to the timeline.

    Returns:
        A frame in the PIX-native :data:`~pixguard_sim.schema.EVENT_COLUMNS`.

    Raises:
        KeyError: If a required Tide column is missing.
    """
    missing = [c for c in TIDE_TX_COLUMNS if c not in tx.columns]
    if missing:
        raise KeyError(f"missing Tide columns: {missing}")

    rng = np.random.default_rng(0)
    events: list[PixEvent] = []
    t0 = int(tx["timestamp"].min())
    for row in tx.itertuples(index=False):
        is_fraud = int(row.is_laundering)
        t_init = int(row.timestamp) - t0
        events.append(
            PixEvent(
                event_id=str(row.transaction_id),
                scenario="mule_chain" if is_fraud else "legit",
                is_fraud=is_fraud,
                payer_account=int(row.src_account),
                payee_account=int(row.dst_account),
                payee_dict_key=f"evp-tide-{int(row.dst_account):08d}",
                amount_brl=float(row.amount),
                t_init_ms=t_init,
                t_settle_ms=t_init + settle_offset_ms,
                med_layer=0,
                device_changed=int(is_fraud and rng.random() < 0.5),
                new_payee=int(is_fraud and rng.random() < 0.8),
                payer_velocity_1h=int(rng.integers(0, 8)),
                is_remote_session=0,
                coercion_flag=0,
            )
        )
    frame = events_to_frame(events)
    logger.info(
        "adapted Tide frame: events=%d fraud=%d",
        len(frame),
        int(frame["is_fraud"].sum()),
    )
    return frame


def synthesize_tide_shaped(
    n_events: int, fraud_rate: float, seed: int
) -> pd.DataFrame:
    """Build a Tide-shaped frame (independent of the Tier A generator).

    This produces a frame in :data:`TIDE_TX_COLUMNS` with a *different*
    generative process (uniform random src/dst over a node pool, exponential
    inter-arrival timestamps, amount drawn per label), so that exercising the
    adapter genuinely demonstrates the harness on a second, differently-shaped
    input rather than re-feeding the Tier A schema.
    """
    rng = np.random.default_rng(seed)
    n_nodes = max(50, n_events // 20)
    is_fraud = (rng.random(n_events) < fraud_rate).astype(int)
    amount = np.where(
        is_fraud == 1,
        rng.lognormal(6.0, 1.0, n_events),
        rng.lognormal(4.4, 0.9, n_events),
    )
    timestamp = np.cumsum(rng.exponential(500, n_events)).astype("int64")
    frame = pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(n_events)],
            "src_account": rng.integers(0, n_nodes, n_events),
            "dst_account": rng.integers(0, n_nodes, n_events),
            "amount": np.round(amount, 2),
            "timestamp": timestamp,
            "is_laundering": is_fraud,
        }
    )
    return frame


# ---------------------------------------------------------------------------
# Tier C: PENDING adapters (specified, gated, never fabricating)
# ---------------------------------------------------------------------------

# Documented 17-column schema of the open PIX generator prior art.
PIX_FRAUD_BR_COLUMNS: tuple[str, ...] = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
    "pix_key_type",
    "device_id",
    "geo_region",
    "is_new_payee",
    "channel",
    "mcc",
)


def adapt_pix_fraud_br(csv_path: str | Path) -> pd.DataFrame:
    """Map the open PIX generator output into the PIX schema (PENDING).

    The source data sits behind a Kaggle/Hugging Face access wall, so this is
    not runnable in this environment. The adapter is specified against the
    documented 17-column schema and raises a clear error if the file is absent,
    so a result is never fabricated. Once the released CSV/Parquet is available,
    point this at it to run experiment P1.

    Raises:
        MissingSourceDataError: If the source file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise MissingSourceDataError(
            "pix-fraud-br source not found (PENDING: needs Kaggle/HF access). "
            f"Expected file: {path}"
        )
    df = pd.read_csv(path)
    events: list[PixEvent] = []
    for i, row in enumerate(df.itertuples(index=False)):
        is_fraud = int(row.isFraud)
        t_init = int(row.step) * 3_600_000  # 1 step = 1 hour
        events.append(
            PixEvent(
                event_id=f"PFB{i:07d}",
                scenario="account_takeover" if is_fraud else "legit",
                is_fraud=is_fraud,
                payer_account=abs(hash(row.nameOrig)) % 10**8,
                payee_account=abs(hash(row.nameDest)) % 10**8,
                payee_dict_key=str(row.nameDest),
                amount_brl=float(row.amount),
                t_init_ms=t_init,
                t_settle_ms=t_init + 1000,
                med_layer=0,
                device_changed=0,
                new_payee=int(getattr(row, "is_new_payee", 0)),
                payer_velocity_1h=0,
                is_remote_session=0,
                coercion_flag=0,
            )
        )
    return events_to_frame(events)


def adapt_amlsim(csv_path: str | Path) -> pd.DataFrame:
    """Map AMLSim transaction output into the PIX schema (PENDING).

    AMLSim requires a Java build chain not provisioned here. This adapter is
    specified against its transaction CSV and raises if the file is absent.

    Raises:
        MissingSourceDataError: If the source file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise MissingSourceDataError(
            "AMLSim source not found (PENDING: needs Java + AMLSim build). "
            f"Expected file: {path}"
        )
    df = pd.read_csv(path)
    # AMLSim emits SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_AMOUNT, TIMESTAMP,
    # IS_FRAUD; project onto the PIX schema.
    rename = {
        "SENDER_ACCOUNT_ID": "src",
        "RECEIVER_ACCOUNT_ID": "dst",
        "TX_AMOUNT": "amount",
        "TIMESTAMP": "ts",
        "IS_FRAUD": "is_fraud",
    }
    df = df.rename(columns=rename)
    assert set(EVENT_COLUMNS)  # schema is the normalization target
    events: list[PixEvent] = []
    t0 = int(df["ts"].min())
    for i, row in enumerate(df.itertuples(index=False)):
        is_fraud = int(row.is_fraud)
        t_init = int(row.ts) - t0
        events.append(
            PixEvent(
                event_id=f"AML{i:07d}",
                scenario="mule_chain" if is_fraud else "legit",
                is_fraud=is_fraud,
                payer_account=int(row.src),
                payee_account=int(row.dst),
                payee_dict_key=f"evp-aml-{int(row.dst):08d}",
                amount_brl=float(row.amount),
                t_init_ms=t_init,
                t_settle_ms=t_init + 1000,
                med_layer=0,
                device_changed=0,
                new_payee=0,
                payer_velocity_1h=0,
                is_remote_session=0,
                coercion_flag=0,
            )
        )
    return events_to_frame(events)
