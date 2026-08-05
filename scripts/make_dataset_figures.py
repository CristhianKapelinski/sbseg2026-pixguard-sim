"""Regenerate the paper's DATASET figures from the real generated stream.

Where ``make_figure.py`` plots what the detectors *scored*, this script plots
what the detectors were scored *on*: the event stream itself. It writes two
figures, each a wide, short strip, because vertical centimetres are the scarce
resource under a page limit:

``fig_dataset.pdf`` (four panels)
    (a) a sampled neighbourhood of the account graph with one real MED refund
    chain and one real mule chain traced through it, (b) measured signal
    presence per scenario, (c) transfer-amount scale per source, (d) the degree
    distribution of the full account graph.

``fig_clocks.pdf`` (two panels)
    (a) the settlement window the events carry, read against the swept
    deadlines, (b) each detector's measured decision latency against that same
    window. Together they put the metric's two clocks on one axis.

Panels deliberately NOT drawn, because another panel already carries the same
information: a "share still blockable at deadline d" curve (one minus panel
(a) of ``fig_clocks``); an illicit-rate bar chart (the Table of sources already
lists the rate); a mean-degree bar chart (four scalars, and panel (d) shows the
whole distribution); and an events-per-hop-index bar chart (panel (a) numbers
the hops of a real chain).

Every number plotted is read from ``data/events.csv`` (the generator's own
output), from the published result JSONs, or recomputed from the base graph
under the config's seed. Nothing is synthesized for the plot: where a quantity
was not measured, the panel says so rather than drawing a plausible shape.

Usage: python scripts/make_dataset_figures.py [events_csv] [results_dir] [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixguard_sim.base_graph import build_base_graph  # noqa: E402
from pixguard_sim.config import PipelineConfig  # noqa: E402

# Okabe-Ito subset, validated for colour-vision deficiency separation. The four
# hues sit in one lightness band, so they collapse under greyscale printing;
# every panel therefore carries a second, non-colour encoding (line style,
# hatch, marker, or a direct label).
SCENARIO_STYLE: dict[str, tuple[str, str, str]] = {
    # scenario: (colour, line style, hatch)
    "legit": ("#7F7F7F", "-", ""),
    "account_takeover": ("#0072B2", "--", "//"),
    "mule_chain": ("#009E73", "-.", "\\\\"),
    "fake_med_refund": ("#D55E00", ":", "xx"),
    "coercion": ("#CC79A7", (0, (3, 1, 1, 1)), ".."),
}
SCENARIO_LABEL: dict[str, str] = {
    "legit": "legit",
    "account_takeover": "takeover",
    "mule_chain": "mule",
    "fake_med_refund": "MED",
    "coercion": "coercion",
}
FRAUD_SCENARIOS = ["account_takeover", "mule_chain", "fake_med_refund", "coercion"]
SIGNALS = ["device_changed", "new_payee", "is_remote_session", "coercion_flag"]
SIGNAL_LABEL = ["device\nchanged", "new\npayee", "remote\nsession", "coercion\nflag"]

SOURCE_ORDER = [
    ("ours", "ours", "#0072B2"),
    ("pix_fraud_br", "pix-fraud-br", "#009E73"),
    ("tide_hi", "Tide-HI", "#D55E00"),
    ("tide_li", "Tide-LI", "#CC79A7"),
]


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical CDF (x sorted, y in (0, 1]) of ``values``."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "figure.dpi": 200,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
        }
    )


def _chains(events: pd.DataFrame, scenario: str) -> list[list[tuple[int, int]]]:
    """Reconstruct hop paths for a chained scenario.

    A chain starts where ``med_layer`` returns to 0 and continues while the
    payee of one hop is the payer of the next, which is how the generator emits
    them. Returns a list of edge lists, longest first.
    """
    sub = events[events.scenario == scenario].sort_values("event_id")
    out: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for _, row in sub.iterrows():
        edge = (int(row.payer_account), int(row.payee_account))
        if row.med_layer == 0:
            if current:
                out.append(current)
            current = [edge]
        elif current and current[-1][1] == edge[0]:
            current.append(edge)
        else:  # defensive: a hop that does not continue its predecessor
            if current:
                out.append(current)
            current = [edge]
    if current:
        out.append(current)
    return sorted(out, key=len, reverse=True)


# --------------------------------------------------------------------------
# Figure 1: what the stream and its account graph look like
# --------------------------------------------------------------------------
def fig_dataset(
    events: pd.DataFrame, stats_path: Path, config: Path, out: Path
) -> None:
    """Draw the account graph, the signals, the amount scale, and the degrees."""
    cfg = PipelineConfig.load(config)
    # Same call the generator makes, so this is the graph the events ran on.
    base = build_base_graph(cfg.generator.n_accounts, seed=cfg.generator.seed)
    g = base.graph

    fig, axes = plt.subplots(
        1, 4, figsize=(9.6, 2.15), gridspec_kw={"width_ratios": [1.5, 1.0, 1.1, 0.95]}
    )
    ax1, ax2, ax3, ax4 = axes

    # (a) a real MED refund chain and a real mule chain, traced through a
    # sampled neighbourhood of the base account graph.
    med = _chains(events, "fake_med_refund")[0]
    mule = _chains(events, "mule_chain")[0]
    seeds = {n for e in med + mule for n in e}
    nodes = set(seeds)
    for s in seeds:
        if s in g:
            nodes.update(list(g.neighbors(s))[:5])
    sub = g.subgraph(nodes).copy()
    for e in med + mule:  # chain hops are transaction edges, not base edges
        sub.add_edge(*e)

    pos = nx.spring_layout(sub, seed=cfg.generator.seed, k=0.6, iterations=200)
    deg = dict(g.degree())
    nx.draw_networkx_edges(sub, pos, ax=ax1, edge_color="#D5D5D5", width=0.4)
    nx.draw_networkx_nodes(
        sub, pos, ax=ax1, node_size=[8 + 1.6 * deg.get(n, 1) for n in sub.nodes()],
        node_color="#EAEAEA", edgecolors="#9A9A9A", linewidths=0.35,
    )
    for edges, name in ((med, "fake_med_refund"), (mule, "mule_chain")):
        colour = SCENARIO_STYLE[name][0]
        nx.draw_networkx_edges(
            sub, pos, ax=ax1, edgelist=edges, edge_color=colour, width=1.6,
            arrows=True, arrowsize=7, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
        )
        chain_nodes = [edges[0][0]] + [b for _, b in edges]
        nx.draw_networkx_nodes(
            sub, pos, ax=ax1, nodelist=chain_nodes, node_size=[20] * len(chain_nodes),
            node_color=colour, edgecolors="white", linewidths=0.4,
        )
        for i, (a, b) in enumerate(edges, start=1):
            ax1.text(
                (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2, str(i),
                fontsize=5, color=colour, ha="center", va="center",
                bbox={"boxstyle": "circle,pad=0.10", "fc": "white", "ec": colour,
                      "lw": 0.4},
            )
    ax1.legend(
        handles=[
            plt.Line2D([], [], color=SCENARIO_STYLE[n][0], lw=1.6,
                       label=f"{SCENARIO_LABEL[n]}, {len(e)} hops")
            for n, e in (("fake_med_refund", med), ("mule_chain", mule))
        ],
        loc="lower left", framealpha=0.9, fontsize=5.5, handlelength=1.4,
        borderpad=0.3,
    )
    ax1.set_axis_off()
    ax1.set_title("(a) sampled account neighbourhood")

    # (b) measured signal presence per scenario: one hue, light to dark.
    order = ["legit", *FRAUD_SCENARIOS]
    mat = np.array(
        [[events.loc[events.scenario == s, c].mean() for c in SIGNALS] for s in order]
    )
    ax2.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(len(SIGNALS)))
    ax2.set_xticklabels(SIGNAL_LABEL, fontsize=5.5)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([SCENARIO_LABEL[s] for s in order], fontsize=6)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax2.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=5.5,
                     color="white" if mat[r, c] > 0.55 else "#222")
    ax2.grid(False)
    ax2.set_title("(b) signal presence per scenario")

    # (c) the amount scale each source carries: the one real feature that
    # survives the shared schema.
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    recs = [stats[k].get("as_scored", stats[k]) for k, _, _ in SOURCE_ORDER]
    for i, (rec, (_, label, colour)) in enumerate(
        zip(recs, SOURCE_ORDER, strict=True)
    ):
        a = rec["amount"]
        y = len(recs) - 1 - i
        ax3.plot([a["q05"], a["q95"]], [y, y], color=colour, linewidth=3.5,
                 alpha=0.40, solid_capstyle="butt")
        ax3.plot([a["q25"], a["q75"]], [y, y], color=colour, linewidth=3.5,
                 alpha=0.85, solid_capstyle="butt")
        ax3.plot(a["q50"], y, marker="|", color="white", markersize=5,
                 markeredgewidth=1.2, linestyle="none")
        ax3.annotate(f"{a['q50']:,.0f}", xy=(a["q95"], y), xytext=(4, 2),
                     textcoords="offset points", fontsize=5.5, color=colour)
    ax3.set_yticks(range(len(recs)))
    ax3.set_yticklabels([lab for _, lab, _ in reversed(SOURCE_ORDER)], fontsize=6)
    ax3.set_xscale("log")
    ax3.set_xlim(1, 3e8)
    ax3.set_xlabel("amount (log scale); bar q25-q75, median labelled")
    ax3.grid(True, axis="x")
    ax3.grid(False, axis="y")
    ax3.set_title("(c) amount scale per source")

    # (d) degree distribution of the FULL account graph, log-log.
    degrees = np.array([d for _, d in g.degree()])
    vals, counts = np.unique(degrees, return_counts=True)
    ax4.plot(vals, counts / counts.sum(), marker="o", markersize=2.2,
             linestyle="none", color="#0072B2")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("degree (log scale)")
    ax4.set_ylabel("share of accounts (log)")
    ax4.annotate(
        f"{g.number_of_nodes():,} accounts\n{g.number_of_edges():,} links\n"
        f"max degree {degrees.max()}",
        xy=(0.96, 0.96), xycoords="axes fraction", ha="right", va="top",
        fontsize=5.5, color="#333",
    )
    ax4.set_title("(d) account degree distribution")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 2: the two clocks (settlement window vs measured decision latency)
# --------------------------------------------------------------------------
def fig_clocks(events: pd.DataFrame, results: Path, out: Path) -> None:
    """Plot the settlement window against the detectors' measured latency."""
    window = (events.t_settle_ms - events.t_init_ms).to_numpy(dtype=float)
    e8 = json.loads((results / "e8.json").read_text(encoding="utf-8"))
    sweep = sorted(e8.get("deadline_sweep_ms", [200, 1000, 2000, 5000]))
    lo, hi = window.min(), window.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 1.95))

    # (a) the settlement window, read against the swept deadlines. A companion
    # "share still blockable at d" panel would be one minus this curve, so it
    # is left out rather than drawn twice.
    x, y = _ecdf(window)
    ax1.plot(x, y, color="#0072B2", linewidth=1.4)
    ax1.axvspan(lo, hi, color="#0072B2", alpha=0.08, linewidth=0)
    for i, d in enumerate(sweep):
        if 10 <= d <= 12000:
            ax1.axvline(d, color="#555", linewidth=0.6, linestyle="--", alpha=0.6)
            ax1.annotate(f"{d}", xy=(d, 1.13 if i % 2 else 1.03), xytext=(1, 0),
                         textcoords="offset points", fontsize=5.5, color="#555")
    ax1.annotate(f"{lo:.0f}–{hi:.0f} ms", xy=(np.median(window), 0.45),
                 xytext=(9, -16), textcoords="offset points", fontsize=6,
                 color="#0072B2")
    ax1.set_xscale("log")
    ax1.set_xlim(10, 12000)
    ax1.set_ylim(-0.03, 1.25)
    ax1.set_xlabel("time from initiation (ms, log scale)")
    ax1.set_ylabel("share settled")
    ax1.set_title("(a) settlement window vs. swept deadlines")

    # (b) measured decision latency per detector on the same axis. A detector
    # whose per-event spread was not published gets a single labelled point,
    # not an invented span.
    ax2.axvspan(lo, hi, color="#0072B2", alpha=0.10, linewidth=0,
                label="settlement window")
    rows = [
        ("rf_fast", "RF", "#009E73", "^"),
        ("llm_terse", "Terse LM", "#D55E00", "s"),
        ("llm_reasoning", "Reasoning LM", "#CC79A7", "o"),
    ]
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
    ax2.set_yticklabels([label for _, label, _, _ in reversed(rows)], fontsize=6)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 12000)
    ax2.set_ylim(-0.6, len(rows) - 0.35)
    ax2.set_xlabel("measured decision latency (ms, log scale)")
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.legend(loc="lower left", framealpha=0.9, borderpad=0.3)
    ax2.set_title("(b) decision latency vs. that window")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str]) -> int:
    events_csv = Path(argv[1]) if len(argv) > 1 else Path("data/events.csv")
    results = Path(argv[2]) if len(argv) > 2 else Path("results/published")
    out_dir = Path(argv[3]) if len(argv) > 3 else Path("figs")
    config = Path("configs/default.json")

    _style()
    events = pd.read_csv(events_csv)
    fig_dataset(events, results / "dataset_stats.json", config,
                out_dir / "fig_dataset.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
