"""PIX-native event schema.

This module defines the normalized event record that every generator (the
in-repo synthetic generator and the Tide, pix-fraud-br, and AMLSim adapters)
maps into, and that the harness consumes. The fields mirror authentic
PIX/DICT/MED payload semantics (DICT key resolution, device/geo signals,
multi-hop refund layer index) drawn from the public PIX ecosystem schemas,
without using any real customer data.

Every event carries a *relative* timeline: ``t_init_ms`` is when the transfer
is initiated and ``t_settle_ms`` is when it becomes irrevocable. The harness
measures, per fraud event, whether a detector flags it before a configurable
decision deadline expressed on this same relative timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Ordered column schema for the tabular event stream. Keeping it explicit makes
# adapters from foreign generators auditable and the CSV stable across runs.
EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "scenario",  # one of config.SCENARIOS, or "legit"
    "is_fraud",  # ground-truth label (1 fraud, 0 legitimate)
    "payer_account",
    "payee_account",
    "payee_dict_key",  # resolved DICT key of the payee
    "amount_brl",
    "t_init_ms",  # transfer initiation, relative timeline (ms)
    "t_settle_ms",  # irrevocable settlement, relative timeline (ms)
    "med_layer",  # 0 for the originating transfer, 1..N for refund-chain hops
    "device_changed",  # 1 if the payer device differs from its usual device
    "new_payee",  # 1 if the payee DICT key is new to the payer
    "payer_velocity_1h",  # count of payer transfers in the trailing hour
    "is_remote_session",  # 1 if a remote-access ("ghost hand") session flag set
    "coercion_flag",  # 1 if the event is flagged as duress-initiated
)


@dataclass(frozen=True)
class PixEvent:
    """A single normalized PIX transfer event.

    Field semantics match :data:`EVENT_COLUMNS`. This dataclass is the typed
    surface generators build; the harness operates on the DataFrame form.
    """

    event_id: str
    scenario: str
    is_fraud: int
    payer_account: int
    payee_account: int
    payee_dict_key: str
    amount_brl: float
    t_init_ms: int
    t_settle_ms: int
    med_layer: int
    device_changed: int
    new_payee: int
    payer_velocity_1h: int
    is_remote_session: int
    coercion_flag: int

    def as_row(self) -> tuple[object, ...]:
        """Return the event as a tuple ordered per :data:`EVENT_COLUMNS`."""
        return (
            self.event_id,
            self.scenario,
            self.is_fraud,
            self.payer_account,
            self.payee_account,
            self.payee_dict_key,
            self.amount_brl,
            self.t_init_ms,
            self.t_settle_ms,
            self.med_layer,
            self.device_changed,
            self.new_payee,
            self.payer_velocity_1h,
            self.is_remote_session,
            self.coercion_flag,
        )


def events_to_frame(events: list[PixEvent]) -> pd.DataFrame:
    """Build a typed DataFrame from a list of events.

    The column order is fixed by :data:`EVENT_COLUMNS` so the serialized CSV is
    stable and content-hashable across runs.
    """
    frame = pd.DataFrame(
        [event.as_row() for event in events], columns=list(EVENT_COLUMNS)
    )
    return _enforce_dtypes(frame)


def _enforce_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to stable dtypes for deterministic serialization."""
    int_cols = [
        "is_fraud",
        "payer_account",
        "payee_account",
        "t_init_ms",
        "t_settle_ms",
        "med_layer",
        "device_changed",
        "new_payee",
        "payer_velocity_1h",
        "is_remote_session",
        "coercion_flag",
    ]
    for col in int_cols:
        frame[col] = frame[col].astype("int64")
    frame["amount_brl"] = frame["amount_brl"].astype("float64").round(2)
    return frame


# Numeric feature columns exposed to detectors. The harness never leaks the
# label, scenario, or any identifier; detectors see only these observable
# behavioural signals plus the event timeline.
#
# Three event-schema columns are deliberately *excluded* from this detector
# feature set because they are per-scenario labels in disguise (label leakage):
# ``coercion_flag`` is 1 iff the event is a coercion scenario, ``is_remote_session``
# is 1 iff the event is an account-takeover scenario, and ``med_layer`` is >0 iff
# the event is a multi-hop MED refund hop. Training on those would let a detector
# read the scenario off a single column rather than learn behaviour, so they stay
# in the event schema/data but are kept out of the feature set. Detectors see only
# the four behavioural signals below.
FEATURE_COLUMNS: tuple[str, ...] = (
    "amount_brl",
    "device_changed",
    "new_payee",
    "payer_velocity_1h",
)
