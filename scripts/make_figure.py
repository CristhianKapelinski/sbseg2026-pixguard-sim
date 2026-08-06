"""Regenerate the paper's evaluation figure from the real experiment outputs.

Reads the harness JSON outputs plus the generated event stream and writes one
wide, short four-panel PDF, because vertical centimetres are the scarce
resource under a page limit:

(a) the settlement window the events carry, read against the swept deadlines;
(b) each detector's measured decision latency against that same window (E8);
(c) per-scenario recall of single-hop-trained detectors (E2), showing the
collapse on the PIX-native scenarios; (d) PR-AUC with 95% bootstrap CIs across
the in-repo, pix-fraud-br, and real Tide HI/LI generators (E3/E5).

Panels (a) and (b) are the two clocks the deadline metric involves.

The figure plots only numbers the harness computed or the generator emitted; it
fabricates nothing. A detector that scores a whole frame in one vectorized call
has a mean time per event and no per-event spread to draw, so the panel draws a
point and says the latency is a batch mean, rather than inventing a span.

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


def _label(ax, x: float, y: float, text: str, colour: str, dy: int = 2) -> None:
    """Annotate at ``(x, y)`` without letting the text leave the axes.

    A label anchored past the middle of a (log) x axis has no room to its
    right, so it is right-aligned and grows leftwards instead; one anchored in
    the left half grows rightwards as usual.
    """
    x0, x1 = (np.log10(v) for v in ax.get_xlim())
    frac = (np.log10(max(x, 10 ** x0)) - x0) / (x1 - x0)
    right = frac > 0.5
    ax.annotate(
        text, xy=(x, y), xytext=(-4 if right else 4, dy),
        textcoords="offset points", ha="right" if right else "left",
        va="bottom", fontsize=5.5, color=colour,
    )


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    out = Path(argv[2]) if len(argv) > 2 else Path("paper/figs/fig_results.pdf")
    events_csv = Path(argv[3]) if len(argv) > 3 else Path("data/events.csv")

    e1 = json.loads((results_dir / "e1.json").read_text(encoding="utf-8"))
    e2 = json.loads((results_dir / "e2.json").read_text(encoding="utf-8"))
    e3 = json.loads((results_dir / "e3.json").read_text(encoding="utf-8"))
    e5 = json.loads((results_dir / "e5.json").read_text(encoding="utf-8"))
    e8 = json.loads((results_dir / "e8.json").read_text(encoding="utf-8"))
    e9 = json.loads((results_dir / "e9_hosted.json").read_text(encoding="utf-8"))
    events = pd.read_csv(events_csv, usecols=["t_init_ms", "t_settle_ms"])

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "figure.dpi": 200,
    })
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(9.6, 2.05), gridspec_kw={"width_ratios": [0.95, 1.36, 0.80, 0.96]}
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
    for d in sweep:
        if 10 <= d <= 12000:
            ax1.axvline(d, color="#888", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.annotate(f"grey: swept deadlines {sweep[0]}–{sweep[-1]}\u2009ms",
                 xy=(0.03, 0.985), xycoords="axes fraction", fontsize=5.0,
                 color="#777", ha="left", va="top")
    ax1.axvline(1500.0, color="#B00", linewidth=0.9, linestyle="-.", zorder=2)
    # Labelled with the sweep marks rather than along the line, which would run
    # through the curve.
    ax1.annotate("1500\u2009ms:\nthe budget", xy=(1500.0, 1.00), xytext=(-3, 0),
                 textcoords="offset points", fontsize=5.0, color="#B00",
                 ha="right", va="top")
    ax1.set_xscale("log")
    ax1.set_xlim(10, 12000)
    ax1.set_ylim(-0.03, 1.20)
    # After the scale and limits, so the placement rule reads real bounds.
    _label(ax1, 15.0, 0.52, f"window\n{lo:.0f}–{hi:.0f} ms", "#0072B2")  # empty left
    ax1.set_xlabel("time from initiation (ms, log)")
    ax1.set_ylabel("share settled")
    ax1.grid(True, linestyle=":", linewidth=0.4)
    ax1.set_axisbelow(True)
    ax1.annotate(f"n={len(window):,} events", xy=(0.03, 0.90), xycoords="axes fraction",
                 ha="left", va="top", fontsize=5.5, color="#333")
    ax1.set_title("(a) settlement window vs. deadlines")

    # Panel (b): measured decision latency on the same axis. A detector scored
    # frame-at-a-time has only a mean time per event, so it gets a labelled
    # point rather than a span it never produced.
    ax2.axvspan(lo, hi, color="#0072B2", alpha=0.10, linewidth=0,
                label="settlement")
    rows = [("rf_fast", "RF", "#009E73", "^"),
            ("llm_terse", "Local LM, terse", "#D55E00", "s"),
            ("llm_reasoning", "Local LM, reasoning", "#CC79A7", "o"),
            ("deepseek-v4-flash", "Hosted flash", "#0072B2", "D"),
            ("deepseek-v4-pro", "Hosted pro", "#000000", "v")]
    by_name = {d["detector"]: d for d in e8["detectors"] + e9["detectors"]}
    pending: list[tuple[float, float, str, str]] = []
    for i, (key, label, colour, marker) in enumerate(rows):
        det = by_name[key]
        stats = det.get("per_event_latency_stats_ms")
        ypos = len(rows) - 1 - i
        if stats:
            ax2.plot([stats["min_ms"], stats["max_ms"]], [ypos, ypos], color=colour,
                     linewidth=4.5, solid_capstyle="butt", alpha=0.45)
            ax2.plot(stats["median_ms"], ypos, marker=marker, color=colour,
                     markersize=4, linestyle="none")
            pending.append((stats["max_ms"], ypos,
                            f"{stats['min_ms']:.0f}–{stats['max_ms']:.0f} ms", colour))
        else:
            mean = float(det["measured_mean_latency_ms"])
            ax2.plot(max(mean, 1e-3), ypos, marker=marker, color=colour,
                     markersize=4, linestyle="none")
            pending.append((max(mean, 1e-3), ypos,
                            f"{mean:.3f} ms/event", colour))
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([lab for _, lab, _, _ in reversed(rows)], fontsize=5.5)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 60000)
    ax2.set_ylim(-1.75, len(rows) - 0.25)
    ax2.set_xlabel("measured decision latency (ms, log)")
    ax2.grid(True, axis="x", linestyle=":", linewidth=0.4)
    ax2.set_axisbelow(True)
    for lx, ly, text, colour in pending:  # placed once the axis bounds are final
        _label(ax2, lx, ly, text, colour, dy=5)
    ax2.axvline(1500.0, color="#B00", linewidth=0.9, linestyle="-.", zorder=1,
                label="1.5 s budget")
    from matplotlib.lines import Line2D  # noqa: PLC0415
    encoding = Line2D([0], [0], color="#555", linewidth=4.5, alpha=0.45,
                      marker="o", markersize=4, markerfacecolor="#555",
                      markeredgecolor="#555",
                      label="bar: min to max; marker: median")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles + [encoding], labels + [encoding.get_label()],
               loc="lower left", framealpha=0.9, borderpad=0.25,
               fontsize=5, handlelength=1.8, labelspacing=0.3)
    ax2.annotate(f"n={e8['n_subsample']:,} events, {e8['n_fraud_subsample']:,} fraud",
                 xy=(0.985, 0.985), xycoords="axes fraction", ha="right", va="top",
                 fontsize=5.0, color="#333")
    ax2.set_title("(b) decision latency vs. that window")

    # Panel (c): per-scenario recall, single-hop-trained (E2), with Wilson CIs.
    det = next(d for d in e2["detectors"] if d["detector"].startswith("rf_"))
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
    for x, (v, hi) in enumerate(zip(recalls, his, strict=True)):
        ax3.annotate(f"{v:.2f}", xy=(x, v + hi), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=5.5)
    ax3.set_ylim(0, 1.16)
    ax3.set_ylabel("RF recall, trained on takeover only")
    ax3.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax3.set_axisbelow(True)
    ax3.tick_params(axis="x", labelrotation=0, labelsize=4.6, pad=1)
    ns = [det["per_scenario"].get(s, {}).get("recall", {}).get("n", 0) for s in scenarios]
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels([f"{lab}\nn={n}" for lab, n in zip(labels, ns, strict=True)])
    ax3.set_title("(c) recall per scenario")

    # Panel (d): PR-AUC with 95% CI across generators (in-repo, pfb, Tide HI/LI).
    SHORT = {"rf_fast": "RF", "xgb_fast": "XGB", "gb_slow": "GB",
             "lr_fast": "LR", "rule_threshold": "RULE"}

    def _best_prauc(detectors: list[dict]) -> tuple[float, float, float, str]:
        best = max(detectors, key=lambda d: d["batch"]["pr_auc"])
        b = best["batch"]
        return (b["pr_auc"], b["pr_auc_ci_lo"], b["pr_auc_ci_hi"],
                SHORT.get(best["detector"], best["detector"]))

    gens = ["ours", "pix-fraud-br", "Tide-HI", "Tide-LI"]
    pr_all = [_best_prauc(e1["detectors"]), _best_prauc(e5["detectors"]),
              _best_prauc(e3["splits"]["HI"]["detectors"]),
              _best_prauc(e3["splits"]["LI"]["detectors"])]
    vals = [p[0] for p in pr_all]
    los = [v - p[1] for v, p in zip(vals, pr_all, strict=True)]
    his = [p[2] - v for v, p in zip(vals, pr_all, strict=True)]
    ax4.bar(gens, vals, color="#3b6", width=0.6, yerr=[los, his], capsize=2.5,
            ecolor="#777")
    ax4.set_xticks(range(len(gens)))
    ax4.set_xticklabels([f"{g}\n{p[3]}" for g, p in zip(gens, pr_all, strict=True)])
    for x, (v, hi) in enumerate(zip(vals, his, strict=True)):
        ax4.annotate(f"{v:.2f}", xy=(x, v + hi), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=5.5)
    ax4.set_ylim(0, 1.16)
    ax4.set_ylabel("best PR-AUC (95% CI)")
    ax4.tick_params(axis="x", pad=1)
    ax4.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax4.set_axisbelow(True)
    ax4.tick_params(axis="x", labelrotation=0, labelsize=4.2)
    ax4.set_title("(d) PR-AUC per source")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
