#!/usr/bin/env python3
"""Report the hosted-model latency smoke check against the regulator's budget.

Ten events say nothing about accuracy, so nothing here is gated on it. What ten
events do measure is the per-request round trip, and that is the quantity the
deadline metric reads: the paper's claim is that a hosted reasoning model answers
well and answers too late, and the second half of that is visible immediately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BAR = "═" * 66
SEP = "─" * 66
AUTH_BUDGET_MS = 1500.0


def main() -> int:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print()
    print(BAR)
    print("  Hosted-model latency check (not a paper claim)")
    print(SEP)
    for det in data["detectors"]:
        lat = det.get("per_event_latency_stats_ms") or {}
        if not lat:
            continue
        pre = det.get("pre_deadline_fraction", {}).get("1500")
        share = 100.0 * lat["p95_ms"] / AUTH_BUDGET_MS
        print(f"  {det['detector']}")
        print(f"    {'events scored':<29}: {data['n_subsample']}")
        print(f"    {'mean latency (ms)':<29}: {lat['mean_ms']:.0f}")
        print(f"    {'p95 latency (ms)':<29}: {lat['p95_ms']:.0f}")
        print(f"    {'p95 as share of the 1.5 s budget':<29}: {share:.0f}%")
        if pre:
            print(f"    {'decided inside the budget':<29}: {pre['value']:.3f}")
        if det.get("failed_count"):
            print(f"    {'requests that failed':<29}: {det['failed_count']}")
    print(SEP)
    print("  Latency is a property of this network and this provider, so it is")
    print("  reported and never gated. The paper's E9 scores 1000 events; what these")
    print("  few show is whether a hosted model can answer inside 1.5 s at all.")
    print(BAR)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
