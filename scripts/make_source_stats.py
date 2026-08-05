"""Summarize the three evaluation sources into one committed JSON.

The cross-source figure must not re-read 1.8 GB of released CSV every time it is
drawn, and it must not invent what it cannot measure. This script does the heavy
pass once: it streams each released source, counts its transaction edges, its
illicit rate, its amount quantiles, and the size of the account graph it
induces, and writes ``results/dataset_stats.json``. ``make_dataset_figures.py``
plots that file.

Only the fields each source genuinely carries are recorded. Tide's frame is
filtered to ``edge_type == "transaction"``, matching ``adapters.adapt_tide``, so
the counts describe the same edges the harness scores.

Usage: python scripts/make_source_stats.py [data_dir] [events_csv] [out_json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixguard_sim import data_io  # noqa: E402
from pixguard_sim.config import PipelineConfig  # noqa: E402

CHUNK = 2_000_000
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    """Return the quantiles the figure draws, plus the extremes."""
    qs = np.quantile(values, QUANTILES)
    return {
        "min": float(values.min()),
        **{f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, qs, strict=True)},
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _stats_from_edges(
    amount: np.ndarray,
    is_fraud: np.ndarray,
    src: np.ndarray,
    dest: np.ndarray,
    label: str,
    note: str,
) -> dict[str, object]:
    """Assemble one source's record from its edge arrays.

    Both graph-size measures are recorded, because they disagree in direction
    across these sources and each answers a different question: ``mean_degree``
    is the neighbourhood a graph detector has to aggregate over, while
    ``density`` is the share of possible account pairs that transact.
    """
    accounts = np.union1d(np.unique(src), np.unique(dest))
    n_edges = int(amount.size)
    n_accounts = int(accounts.size)
    possible = n_accounts * (n_accounts - 1) / 2.0
    return {
        "label": label,
        "n_events": n_edges,
        "n_fraud": int(is_fraud.sum()),
        "illicit_rate": float(is_fraud.mean()),
        "amount": _quantile_summary(amount),
        "n_accounts": n_accounts,
        "mean_degree": float(2.0 * n_edges / n_accounts) if n_accounts else 0.0,
        "density": float(n_edges / possible) if possible else 0.0,
        "note": note,
    }


def _ours(events_csv: Path) -> dict[str, object]:
    d = pd.read_csv(events_csv)
    rec = _stats_from_edges(
        d.amount_brl.to_numpy(dtype=float),
        d.is_fraud.to_numpy(dtype=int),
        d.payer_account.to_numpy(),
        d.payee_account.to_numpy(),
        "PixGuard-Sim",
        "generated under configs/default.json",
    )
    # The settlement clock the events carry. The deadline metric is defined
    # against the decision latency alone, so this window is what tells a reader
    # which deadlines still describe a blockable transfer.
    window = (d.t_settle_ms - d.t_init_ms).to_numpy(dtype=float)
    rec["settlement_window_ms"] = _quantile_summary(window)
    return rec


def _scored_slice(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    """The label-stratified subsample the harness actually scores."""
    return data_io.stratified_subsample(df, label_col, n, seed)


def _tide(path: Path, label: str, n_scored: int, seed: int) -> dict[str, object]:
    """Stream one released Tide transaction file, full frame and scored slice."""
    parts: list[pd.DataFrame] = []
    cols = ["src", "dest", "edge_type", "amount", "is_fraudulent"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK):
        parts.append(chunk[chunk.edge_type == "transaction"])
    df = pd.concat(parts, ignore_index=True)

    def rec(frame: pd.DataFrame, note: str) -> dict[str, object]:
        return _stats_from_edges(
            frame.amount.to_numpy(dtype=float),
            frame.is_fraudulent.astype(bool).astype(int).to_numpy(),
            frame.src.to_numpy(), frame.dest.to_numpy(), label, note,
        )

    out = rec(df, "released Zenodo record, transaction edges only")
    out["as_scored"] = rec(
        _scored_slice(df, "is_fraudulent", n_scored, seed),
        f"label-stratified subsample of {n_scored:,} the harness scores",
    )
    return out


def _pix_fraud_br(path: Path, n_scored: int, seed: int) -> dict[str, object]:
    d = pd.read_parquet(
        path, columns=["id_pagador", "id_recebedor", "valor_brl", "fraude"]
    )

    def rec(frame: pd.DataFrame, note: str) -> dict[str, object]:
        return _stats_from_edges(
            frame.valor_brl.to_numpy(dtype=float),
            frame.fraude.astype(bool).astype(int).to_numpy(),
            frame.id_pagador.to_numpy(), frame.id_recebedor.to_numpy(),
            "pix-fraud-br", note,
        )

    out = rec(d, "released Hugging Face set, full frame")
    out["as_scored"] = rec(
        _scored_slice(d, "fraude", n_scored, seed),
        f"label-stratified subsample of {n_scored:,} the harness scores",
    )
    return out


def main(argv: list[str]) -> int:
    data_dir = Path(argv[1]) if len(argv) > 1 else Path("data")
    events_csv = Path(argv[2]) if len(argv) > 2 else Path("data/events.csv")
    out = (
        Path(argv[3]) if len(argv) > 3
        else Path("results/published/dataset_stats.json")
    )

    cfg = PipelineConfig.load(Path("configs/default.json"))
    seed = cfg.generator.seed

    sources = {}
    sources["ours"] = _ours(events_csv)
    print("ours done")
    sources["pix_fraud_br"] = _pix_fraud_br(
        data_dir / "pix_fraud_br.parquet", cfg.data.pix_subsample_n, seed
    )
    print("pix-fraud-br done")
    for key, fname, label in (
        ("tide_hi", "generated_transactions_HI.csv", "Tide-HI"),
        ("tide_li", "generated_transactions_LI.csv", "Tide-LI"),
    ):
        sources[key] = _tide(
            data_dir / "tide" / fname, label, cfg.data.tide_subsample_n, seed
        )
        print(f"{label} done")

    # Feasibility against the events' own clock. The deadline metric compares a
    # detector's latency to a chosen d; this instead compares it to the moment
    # each transfer becomes irrevocable, which removes the free parameter. Only
    # per-detector latency statistics are published, not the per-event vector,
    # so what is computed here is a bound: the share of frauds still blockable
    # if every decision took the detector's fastest observed time, and the share
    # if every decision took its slowest.
    e8_path = out.parent / "e8.json"
    if e8_path.exists():
        d = pd.read_csv(events_csv, usecols=["t_init_ms", "t_settle_ms", "is_fraud"])
        window = (d.t_settle_ms - d.t_init_ms).to_numpy(dtype=float)
        window = window[d.is_fraud.to_numpy() == 1]
        e8 = json.loads(e8_path.read_text(encoding="utf-8"))
        feas = {}
        for det in e8["detectors"]:
            s = det.get("per_event_latency_stats_ms")
            lo = s["min_ms"] if s else float(det["measured_mean_latency_ms"])
            hi = s["max_ms"] if s else float(det["measured_mean_latency_ms"])
            feas[det["detector"]] = {
                "latency_min_ms": lo,
                "latency_max_ms": hi,
                "blockable_at_best_case": float((window > lo).mean()),
                "blockable_at_worst_case": float((window > hi).mean()),
            }
        sources["feasibility_vs_settlement"] = {
            "n_fraud_events": int(window.size),
            "note": "bound from published latency min/max; per-event pairing not published",
            "detectors": feas,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sources, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
