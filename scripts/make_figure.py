"""Regenerate the paper's evaluation figure from the real experiment outputs.

Reads the harness JSON outputs plus the generated event stream and writes one
wide, short four-panel PDF, because vertical centimetres are the scarce
resource under a page limit:

(a) the settlement window the events carry, read against the swept deadlines;
(b) each detector's measured decision latency against that same window (E8);
(c) per-scenario recall of single-hop-trained detectors (E2), showing the
collapse on the PIX-native scenarios; (d) PR-AUC with 95% bootstrap CIs across
the in-repo, pix-fraud-br, and real Tide HI/LI generators (E3/E5).

Panels (a) and (b) are the two clocks the deadline metric involves. They
replace an earlier pre-deadline-fraction-vs-deadline panel: that curve is a
step function recoverable from a detector's latency span and its recall, so it
said nothing these two do not, and it hid where settlement falls.

The figure plots only numbers the harness computed or the generator emitted; it
fabricates nothing. Where a quantity was not measured, the panel says so rather
than drawing a plausible shape.

Usage: python scripts/make_figure.py [results_dir] [output_pdf] [events_csv]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical CDF (x sorted, y in (0, 1]) of ``values``."""
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    out = Path(argv[2]) if len(argv) > 2 else Path("paper/figs/fig_results.pdf")
    events_csv = Path(argv[3]) if len(argv) > 3 else Path("data/events.csv")

    e1 = json.loads((results_dir / "e1.json").read_text(encoding="utf-8"))
    e2 = json.loads((results_dir / "e2.json").read_text(encoding="utf-8"))
    e3 = json.loads((results_dir / "e3.json").read_text(encoding="utf-8"))
    e5 = json.loads((results_dir / "e5.json").read_text(encoding="utf-8"))
    e8 = json.loads((results_dir / "e8.json").read_text(encoding="utf-8"))
    events = pd.read_csv(events_csv, usecols=["t_init_ms", "t_settle_ms"])

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "figure.dpi": 200,
    })
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(9.6, 2.1), gridspec_kw={"width_ratios": [1, 1.25, 0.85, 0.9]}
    )

    # Panel (a): the settlement clock the events carry. A companion "share
    # still blockable at d" curve would be one minus this one, so it is left
    # out rather than drawn twice.
    window = (events.t_settle_ms - events.t_init_ms).to_numpy(dtype=float)
    lo, hi = window.min(), window.max()
    x, y = _ecdf(window)
    ax1.plot(x, y, color="#0072B2", linewidth=1.4)
    ax1.axvspan(lo, hi, color="#0072B2", alpha=0.08, linewidth=0)
    sweep = sorted(e8.get("deadline_sweep_ms", [200, 1000, 2000, 5000]))
    for i, d in enumerate(sweep):
        if 10 <= d <= 12000:
            ax1.axvline(d, color="#555", linewidth=0.6, linestyle="--", alpha=0.6)
            ax1.annotate(f"{d}", xy=(d, 1.13 if i % 2 else 1.03), xytext=(1, 0),
                         textcoords="offset points", fontsize=5.5, color="#555")
    ax1.annotate(f"{lo:.0f}–{hi:.0f} ms", xy=(np.median(window), 0.45),
                 xytext=(8, -15), textcoords="offset points", fontsize=6,
                 color="#0072B2")
    ax1.set_xscale("log")
    ax1.set_xlim(10, 12000)
    ax1.set_ylim(-0.03, 1.25)
    ax1.set_xlabel("time from initiation (ms, log)")
    ax1.set_ylabel("share settled")
    ax1.grid(True, linestyle=":", linewidth=0.4)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) settlement window vs. deadlines")

    # Panel (b): measured decision latency on the same axis. A detector whose
    # per-event spread was not published gets a labelled point, not a span.
    ax2.axvspan(lo, hi, color="#0072B2", alpha=0.10, linewidth=0,
                label="settlement window")
    rows = [("rf_fast", "RF", "#009E73", "^"),
            ("llm_terse", "Terse LM", "#D55E00", "s"),
            ("llm_reasoning", "Reasoning LM", "#CC79A7", "o")]
    by_name = {d["detector"]: d for d in e8["detectors"]}
    for i, (key, label, colour, marker) in enumerate(rows):
        det = by_name[key]
        stats = det.get("per_event_latency_stats_ms")
        ypos = len(rows) - 1 - i
        if stats:
            ax2.plot([stats["min_ms"], stats["max_ms"]], [ypos, ypos], color=colour,
                     linewidth=4.5, solid_capstyle="butt", alpha=0.45)
            ax2.plot(stats["median_ms"], ypos, marker=marker, color=colour,
                     markersize=4, linestyle="none")
            ax2.annotate(f"{stats['min_ms']:.0f}–{stats['max_ms']:.0f} ms",
                         xy=(stats["max_ms"], ypos), xytext=(5, 2),
                         textcoords="offset points", fontsize=5.5, color=colour)
        else:
            mean = float(det["measured_mean_latency_ms"])
            ax2.plot(max(mean, 1e-3), ypos, marker=marker, color=colour,
                     markersize=4, linestyle="none")
            ax2.annotate(f"mean {mean:.3f} ms (spread not published)",
                         xy=(max(mean, 1e-3), ypos), xytext=(5, 2),
                         textcoords="offset points", fontsize=5.5, color=colour)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([lab for _, lab, _, _ in reversed(rows)], fontsize=6)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 12000)
    ax2.set_ylim(-0.6, len(rows) - 0.35)
    ax2.set_xlabel("measured decision latency (ms, log)")
    ax2.grid(True, axis="x", linestyle=":", linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower left", framealpha=0.9, borderpad=0.3)
    ax2.set_title("(b) decision latency vs. that window")

    # Panel (c): per-scenario recall, single-hop-trained (E2), with Wilson CIs.
    det = e2["detectors"][0]
    scenarios = ["account_takeover", "mule_chain", "fake_med_refund", "coercion"]
    labels = ["takeover", "mule", "MED", "coercion"]
    recalls, los, his = [], [], []
    for s in scenarios:
        r = det["per_scenario"].get(s, {}).get("recall", {})
        v = r.get("value", 0.0)
        recalls.append(v)
        los.append(v - r.get("ci_lo", v))
        his.append(r.get("ci_hi", v) - v)
    ax3.bar(labels, recalls, color="#444", width=0.6, yerr=[los, his], capsize=2.5,
            ecolor="#999")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("recall (single-hop trained)")
    ax3.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax3.set_axisbelow(True)
    ax3.tick_params(axis="x", labelrotation=20, labelsize=6)
    ax3.set_title("(c) recall per scenario")

    # Panel (d): PR-AUC with 95% CI across generators (in-repo, pfb, Tide HI/LI).
    def _best_prauc(detectors: list[dict]) -> tuple[float, float, float]:
        best = max(detectors, key=lambda d: d["batch"]["pr_auc"])
        b = best["batch"]
        return b["pr_auc"], b["pr_auc_ci_lo"], b["pr_auc_ci_hi"]

    gens = ["ours", "pix-fraud-br", "Tide-HI", "Tide-LI"]
    pr_all = [_best_prauc(e1["detectors"]), _best_prauc(e5["detectors"]),
              _best_prauc(e3["splits"]["HI"]["detectors"]),
              _best_prauc(e3["splits"]["LI"]["detectors"])]
    vals = [p[0] for p in pr_all]
    los = [v - p[1] for v, p in zip(vals, pr_all, strict=True)]
    his = [p[2] - v for v, p in zip(vals, pr_all, strict=True)]
    ax4.bar(gens, vals, color="#3b6", width=0.6, yerr=[los, his], capsize=2.5,
            ecolor="#777")
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("best PR-AUC (95% CI)")
    ax4.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax4.set_axisbelow(True)
    ax4.tick_params(axis="x", labelrotation=20, labelsize=6)
    ax4.set_title("(d) PR-AUC per source")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
