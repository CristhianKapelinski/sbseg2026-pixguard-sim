#!/usr/bin/env python3
"""Emit DOCUMENTATION.md's results tables from the committed result files.

The tables in DOCUMENTATION.md are captured output. Hand-maintained, they go
stale the first time an experiment is rerun, and a reader has no way to tell
which of the two numbers in front of them is the live one. This script prints
the whole block, so refreshing the document is a regeneration rather than a
transcription.

Usage: python scripts/make_documentation_tables.py [results_dir] > block.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _f(x: float | None, nd: int = 3) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def _ci(b: dict, key: str) -> str:
    """Render ``value [lo, hi]`` when the bootstrap bounds are present."""
    lo, hi = b.get(f"{key}_ci_lo"), b.get(f"{key}_ci_hi")
    if lo is None or hi is None:
        return _f(b[key])
    return f"{_f(b[key])} [{_f(lo)},{_f(hi)}]"


def _lat(ms: float) -> str:
    return "< 0.01" if ms < 0.01 else f"{ms:.3f}" if ms < 1 else f"{ms:.0f}"


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("results/published")
    load = lambda n: json.loads((root / f"{n}.json").read_text(encoding="utf-8"))  # noqa: E731

    e1, e2, e3, e5, e6, e8 = (load(n) for n in ("e1", "e2", "e3", "e5", "e6", "e8"))
    e9 = load("e9_hosted")
    out: list[str] = []
    w = out.append

    # -- E1: every tabular detector, batch metrics next to measured latency.
    w("### E1 — the deadline metric under measured latency\n")
    w(f"In-repo stream: {e1['n_events']:,} events "
      f"(fraud={e1['n_fraud']}, rate={e1['fraud_rate']:.4f}); "
      f"eval={e1['n_eval']} (fraud={e1['n_eval_fraud']}).\n")
    w("| detector | meas. latency (ms/event) | F1 [95% CI] | PR-AUC [95% CI] | recall | pre@1000ms |")
    w("|---|---|---|---|---|---|")
    for d in e1["detectors"]:
        b = d["batch"]
        w(f"| {d['detector']} | {_lat(d['measured_latency_ms'])} | {_ci(b, 'f1')} | "
          f"{_ci(b, 'pr_auc')} | {_f(b['recall'])} | "
          f"{_f(d['pre_deadline_fraction']['1000']['value'])} |")

    # -- E8 and E9 share one slice, so they belong in one table.
    w("\n### E8 and E9 — language models on one shared slice\n")
    w(f"Both score the same {e8['n_subsample']:,} events "
      f"({e8['n_fraud_subsample']} frauds), drawn with the same seed. "
      f"*In time* is at the regulator's {e9['authorization_budget_ms']:.0f} ms "
      "authorization budget.\n")
    w("| detector | where it runs | PR-AUC | latency mean / p95 (ms) | output tokens | in time |")
    w("|---|---|---|---|---|---|")
    for d in e8["detectors"]:
        s = d.get("per_event_latency_stats_ms")
        lat = f"{s['mean_ms']:.0f} / {s['p95_ms']:.0f}" if s else "< 0.01"
        w(f"| {d['detector']} | this machine | {_f(d['batch']['pr_auc'])} | {lat} | "
          f"{d.get('new_tokens_median', '--')} | "
          f"{_f(d['pre_deadline_fraction']['1000']['value'])} |")
    for d in e9["detectors"]:
        s = d["per_event_latency_stats_ms"]
        w(f"| {d['detector']} | hosted, over the network | {_f(d['batch']['pr_auc'])} | "
          f"{s['mean_ms']:.0f} / {s['p95_ms']:.0f} | {d['new_tokens_median']} | "
          f"{_f(d['pre_deadline_1500ms']['value'])} |")

    # -- E2: recall per scenario for the single-hop-trained detectors.
    w("\n### E2 — single-hop-trained detectors on the unseen scenarios\n")
    scen = ["account_takeover", "mule_chain", "fake_med_refund", "coercion"]
    w("| detector | " + " | ".join(scen) + " |")
    w("|---" * (len(scen) + 1) + "|")
    for d in e2["detectors"]:
        cells = []
        for s in scen:
            r = d["per_scenario"].get(s, {}).get("recall", {})
            cells.append(f"{_f(r.get('value'))} [{_f(r.get('ci_lo'))},{_f(r.get('ci_hi'))}] N={r.get('n', '--')}")
        w(f"| {d['detector']} | " + " | ".join(cells) + " |")

    # -- E3/E5: the external sources.
    w("\n### E3 and E5 — the externally released sources\n")
    w("| source | detector | F1 | PR-AUC [95% CI] | recall |")
    w("|---|---|---|---|---|")
    for split, obj in e3["splits"].items():
        for d in obj["detectors"]:
            b = d["batch"]
            w(f"| Tide-{split} | {d['detector']} | {_f(b['f1'])} | {_ci(b, 'pr_auc')} | {_f(b['recall'])} |")
    for d in e5["detectors"]:
        b = d["batch"]
        w(f"| pix-fraud-br | {d['detector']} | {_f(b['f1'])} | {_ci(b, 'pr_auc')} | {_f(b['recall'])} |")

    # -- E6: train on one source, test on another.
    w("\n### E6 — training on one source and testing on another\n")
    w("| transfer | n_test | F1 | PR-AUC [95% CI] | pre@1000ms |")
    w("|---|---|---|---|---|")
    for t in e6["transfers"]:
        b = t["batch"]
        w(f"| {t['name']} | {t['n_test']:,} | {_f(b['f1'])} | {_ci(b, 'pr_auc')} | "
          f"{_f(t['pre_deadline_1000ms']['value'])} |")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
