"""Regenerate the paper's results figure from the real experiment outputs.

Reads ``results/e1.json`` and ``results/e2.json`` (produced by the harness) and
writes a compact two-panel PDF: (a) the pre-deadline flag fraction vs deadline
for the E1 detectors, the headline discriminative curve; (b) per-scenario
recall of single-hop-trained detectors from E2, showing the collapse on the
PIX-native scenarios. The figure plots only numbers the harness computed; it
fabricates nothing. This script is a build tool, not part of the package
runtime, so matplotlib is an optional ``figures`` extra rather than a runtime
dependency.

Usage: python scripts/make_figure.py [results_dir] [output_pdf]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEADLINES = [100, 250, 500, 1000, 2000, 5000, 10000]


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    out = Path(argv[2]) if len(argv) > 2 else Path("paper/fig_results.pdf")

    e1 = json.loads((results_dir / "e1.json").read_text(encoding="utf-8"))
    e2 = json.loads((results_dir / "e2.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Panel (a): pre-deadline flag fraction vs deadline.
    markers = ["o", "s", "^", "D"]
    for det, marker in zip(e1["detectors"], markers):
        pdf = det["pre_deadline_fraction"]
        y = [pdf[str(d)] for d in DEADLINES]
        ax1.plot(DEADLINES, y, marker=marker, label=det["detector"], linewidth=1.3)
    ax1.set_xscale("log")
    ax1.set_xlabel("decision deadline (ms)")
    ax1.set_ylabel("pre-deadline flag fraction")
    ax1.set_ylim(-0.03, 1.03)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.4)
    ax1.legend(fontsize=7, loc="center left")
    ax1.set_title("(a) deadline metric is discriminative", fontsize=9)

    # Panel (b): per-scenario recall, single-hop-trained (E2).
    det = e2["detectors"][0]
    scenarios = ["account_takeover", "mule_chain", "fake_med_refund", "coercion"]
    labels = ["ATO", "mule", "MED", "coerc."]
    recalls = [det["per_scenario"].get(s, {}).get("recall", 0.0) for s in scenarios]
    ax2.bar(labels, recalls, color="#444", width=0.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("recall (single-hop trained)")
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax2.set_title("(b) collapse on unseen scenarios", fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
