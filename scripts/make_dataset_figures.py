"""Regenerate the paper's DATASET figures from the real generated stream.

Where ``make_figure.py`` plots what the detectors *scored*, this script plots
what the detectors were scored *on*: the event stream itself. It writes three
vector figures, each a thin horizontal strip so it costs little vertical space:

``fig_deadline_clock.pdf``
    (a) the settlement window carried by the events, (b) each detector's
    measured decision latency against that window, (c) the share of events not
    yet settled at a given deadline. This is the figure that puts the metric's
    two clocks, settlement and decision, on one axis.

``fig_dataset.pdf``
    (a) transfer-amount distribution per scenario, (b) events per hop index for
    the two chained scenarios, (c) measured signal presence per scenario.

``fig_graph.pdf``
    (a) a sampled neighbourhood of the account graph with one real MED refund
    chain and one real mule chain traced through it, (b) the degree
    distribution of the full account graph.

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


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical CDF (x sorted, y in (0, 1]) of ``values``."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "figure.dpi": 200,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
        }
    )


# --------------------------------------------------------------------------
# Figure 1: the two clocks (settlement window vs measured decision latency)
# --------------------------------------------------------------------------
def fig_deadline_clock(events: pd.DataFrame, results: Path, out: Path) -> None:
    """Plot the settlement window against the detectors' measured latency."""
    window = (events.t_settle_ms - events.t_init_ms).to_numpy(dtype=float)
    e8 = json.loads((results / "e8.json").read_text(encoding="utf-8"))
    sweep = e8.get("deadline_sweep_ms", [200, 1000, 2000, 5000])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.35))

    # (a) the settlement window the events actually carry.
    x, y = _ecdf(window)
    ax1.plot(x, y, color="#0072B2", linewidth=1.4)
    lo, hi = window.min(), window.max()
    ax1.axvspan(lo, hi, color="#0072B2", alpha=0.08, linewidth=0)
    ax1.set_xscale("log")
    ax1.set_xlim(10, 12000)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("time from initiation (ms, log scale)")
    ax1.set_ylabel("share of events settled")
    ax1.annotate(
        f"{lo:.0f}–{hi:.0f} ms",
        xy=(np.median(window), 0.5),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=6.5,
        color="#0072B2",
    )
    ax1.set_title("(a) settlement window")

    # (b) measured decision latency per detector on the same axis. Detectors
    # whose per-event spread was not published get a single point, labelled as
    # a mean, rather than an invented span.
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
            ax2.plot([stats["min_ms"], stats["max_ms"]], [ypos, ypos],
                     color=colour, linewidth=5.0, solid_capstyle="butt",
                     alpha=0.45)
            ax2.plot(stats["median_ms"], ypos, marker=marker, color=colour,
                     markersize=5, linestyle="none")
            ax2.plot(stats["p95_ms"], ypos, marker="|", color=colour,
                     markersize=8, linestyle="none")
            ax2.annotate(f"{stats['min_ms']:.0f}–{stats['max_ms']:.0f} ms",
                         xy=(stats["max_ms"], ypos), xytext=(6, 3),
                         textcoords="offset points", fontsize=6, color=colour)
        else:
            mean = float(det["measured_mean_latency_ms"])
            ax2.plot(max(mean, 1e-3), ypos, marker=marker, color=colour,
                     markersize=5, linestyle="none")
            ax2.annotate(f"mean {mean:.3f} ms (spread not published)",
                         xy=(max(mean, 1e-3), ypos), xytext=(6, 3),
                         textcoords="offset points", fontsize=6, color=colour)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([label for _, label, _, _ in reversed(rows)])
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 12000)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlabel("measured decision latency (ms, log scale)")
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.set_title("(b) decision latency vs. that window")

    # (c) how much of the stream is still blockable at a given deadline.
    grid = np.logspace(np.log10(10), np.log10(12000), 300)
    still = [(window > g).mean() for g in grid]
    ax3.plot(grid, still, color="#0072B2", linewidth=1.4)
    # Deadline labels alternate height: neighbouring ticks (1000, 2000) sit too
    # close on a log axis to label on one line.
    for i, d in enumerate(sorted(sweep)):
        if 10 <= d <= 12000:
            ax3.axvline(d, color="#555", linewidth=0.6, linestyle="--", alpha=0.6)
            ax3.annotate(f"{d}", xy=(d, 1.10 if i % 2 else 1.02),
                         xytext=(1, 0), textcoords="offset points",
                         fontsize=6, color="#555")
    ax3.set_ylim(-0.03, 1.20)
    ax3.set_xscale("log")
    ax3.set_xlim(10, 12000)
    ax3.set_xlabel("decision deadline (ms, log scale)")
    ax3.set_ylabel("share not yet settled")
    ax3.set_title("(c) events still blockable at a deadline")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 2: what the stream looks like per scenario
# --------------------------------------------------------------------------
def fig_dataset(events: pd.DataFrame, out: Path) -> None:
    """Plot amount, chain depth, and measured signal presence per scenario."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.35))

    # (a) amount distribution per scenario, as an ECDF (no binning choice).
    for name in ["legit", *FRAUD_SCENARIOS]:
        colour, dash, _ = SCENARIO_STYLE[name]
        sub = events.loc[events.scenario == name, "amount_brl"].to_numpy()
        x, y = _ecdf(sub)
        ax1.plot(x, y, color=colour, linestyle=dash, linewidth=1.3,
                 label=f"{SCENARIO_LABEL[name]} (n={sub.size:,})")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("transfer amount, BRL (log scale)")
    ax1.set_ylabel("cumulative share")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("(a) amount per scenario")

    # (b) events per hop index, for the two scenarios that chain.
    chained = ["fake_med_refund", "mule_chain"]
    depths = sorted(events.loc[events.scenario.isin(chained), "med_layer"].unique())
    width = 0.38
    for i, name in enumerate(chained):
        colour, _, hatch = SCENARIO_STYLE[name]
        counts = [
            int(((events.scenario == name) & (events.med_layer == d)).sum())
            for d in depths
        ]
        pos = np.arange(len(depths)) + (i - 0.5) * width
        ax2.bar(pos, counts, width=width, color=colour, hatch=hatch,
                edgecolor="white", linewidth=0.5, label=SCENARIO_LABEL[name])
        for p, c in zip(pos, counts, strict=True):
            if c:
                ax2.annotate(str(c), xy=(p, c), xytext=(0, 2),
                             textcoords="offset points", ha="center", fontsize=6)
    ax2.set_xticks(np.arange(len(depths)))
    ax2.set_xticklabels([str(int(d)) for d in depths])
    ax2.set_xlabel("hop index carried in med_layer")
    ax2.set_ylabel("events")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(False, axis="x")
    ax2.set_title("(b) events per hop index")

    # (c) measured signal presence per scenario: one hue, light to dark.
    order = ["legit", *FRAUD_SCENARIOS]
    mat = np.array(
        [[events.loc[events.scenario == s, c].mean() for c in SIGNALS] for s in order]
    )
    im = ax3.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(len(SIGNALS)))
    ax3.set_xticklabels(SIGNAL_LABEL, fontsize=6)
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels([SCENARIO_LABEL[s] for s in order])
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax3.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                     fontsize=6,
                     color="white" if mat[r, c] > 0.55 else "#222")
    ax3.grid(False)
    cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("share of events", fontsize=6.5)
    ax3.set_title("(c) signal presence per scenario")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 3: the account graph, with real chains traced through it
# --------------------------------------------------------------------------
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


def fig_graph(events: pd.DataFrame, config: Path, out: Path) -> None:
    """Draw a sampled neighbourhood of the account graph and its degrees."""
    cfg = PipelineConfig.load(config)
    # Same call the generator makes, so this is the graph the events ran on.
    base = build_base_graph(cfg.generator.n_accounts, seed=cfg.generator.seed)
    g = base.graph

    med = _chains(events, "fake_med_refund")[0]
    mule = _chains(events, "mule_chain")[0]
    seeds = {n for e in med + mule for n in e}

    # A readable neighbourhood: the chain accounts plus their base-graph
    # neighbours, capped so the drawing stays legible.
    nodes = set(seeds)
    for s in seeds:
        if s in g:
            nodes.update(list(g.neighbors(s))[:6])
    sub = g.subgraph(nodes).copy()
    for e in med + mule:  # chain hops are transaction edges, not base edges
        sub.add_edge(*e)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    pos = nx.spring_layout(sub, seed=cfg.generator.seed, k=0.55, iterations=200)
    deg = dict(g.degree())
    sizes = [12 + 2.2 * deg.get(n, 1) for n in sub.nodes()]
    nx.draw_networkx_edges(sub, pos, ax=ax1, edge_color="#CCCCCC", width=0.5)
    nx.draw_networkx_nodes(sub, pos, ax=ax1, node_size=sizes,
                           node_color="#E8E8E8", edgecolors="#999999",
                           linewidths=0.4)
    for edges, name in ((med, "fake_med_refund"), (mule, "mule_chain")):
        colour, dash, _ = SCENARIO_STYLE[name]
        nx.draw_networkx_edges(
            sub, pos, ax=ax1, edgelist=edges, edge_color=colour, width=1.9,
            arrows=True, arrowsize=9, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            label=SCENARIO_LABEL[name],
        )
        chain_nodes = [edges[0][0]] + [b for _, b in edges]
        nx.draw_networkx_nodes(sub, pos, ax=ax1, nodelist=chain_nodes,
                               node_size=[26] * len(chain_nodes),
                               node_color=colour, edgecolors="white",
                               linewidths=0.5)
        for i, (a, b) in enumerate(edges, start=1):
            xm = (pos[a][0] + pos[b][0]) / 2
            ym = (pos[a][1] + pos[b][1]) / 2
            ax1.text(xm, ym, str(i), fontsize=6, color=colour, ha="center",
                     va="center",
                     bbox={"boxstyle": "circle,pad=0.12", "fc": "white",
                           "ec": colour, "lw": 0.5})
    handles = [
        plt.Line2D([], [], color=SCENARIO_STYLE[n][0], lw=1.9,
                   label=f"{SCENARIO_LABEL[n]} chain ({len(e)} hops)")
        for n, e in (("fake_med_refund", med), ("mule_chain", mule))
    ]
    handles.append(plt.Line2D([], [], color="#CCCCCC", lw=0.5,
                              label="base account link"))
    ax1.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=6.5)
    ax1.set_axis_off()
    ax1.set_title("(a) sampled account neighbourhood, hops numbered")

    # (b) degree distribution of the FULL graph, log-log.
    degrees = np.array([d for _, d in g.degree()])
    vals, counts = np.unique(degrees, return_counts=True)
    ax2.plot(vals, counts / counts.sum(), marker="o", markersize=3,
             linestyle="none", color="#0072B2")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("account degree (log scale)")
    ax2.set_ylabel("share of accounts (log scale)")
    ax2.annotate(
        f"{g.number_of_nodes():,} accounts, {g.number_of_edges():,} links\n"
        f"max degree {degrees.max()}, median {int(np.median(degrees))}",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.5, color="#333",
    )
    ax2.set_title("(b) account degree distribution")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 4: the three sources side by side, as the harness scores them
# --------------------------------------------------------------------------
SOURCE_ORDER = [
    ("ours", "PixGuard-Sim\n(ours)", "#0072B2", "//"),
    ("pix_fraud_br", "pix-fraud-br", "#009E73", "\\\\"),
    ("tide_hi", "Tide-HI", "#D55E00", "xx"),
    ("tide_li", "Tide-LI", "#CC79A7", ".."),
]


def fig_sources(stats_path: Path, out: Path) -> None:
    """Plot illicit rate, amount scale, and graph shape across the sources."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    # Describe each source as the harness scores it: the external sets enter
    # through a label-stratified subsample, and that slice is what every
    # reported number was computed on.
    recs = [stats[k].get("as_scored", stats[k]) for k, _, _, _ in SOURCE_ORDER]
    labels = [lab for _, lab, _, _ in SOURCE_ORDER]
    colours = [c for _, _, c, _ in SOURCE_ORDER]
    hatches = [h for _, _, _, h in SOURCE_ORDER]
    xs = np.arange(len(recs))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.5))

    # (a) illicit rate, log scale: an order of magnitude separates the sources.
    rates = [r["illicit_rate"] * 100 for r in recs]
    ax1.bar(xs, rates, color=colours, hatch=hatches, edgecolor="white",
            linewidth=0.6, width=0.62)
    for x, v in zip(xs, rates, strict=True):
        ax1.annotate(f"{v:.3f}%", xy=(x, v), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=6)
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_ylabel("illicit events (%, log scale)")
    ax1.grid(False, axis="x")
    ax1.set_title("(a) illicit rate per source")

    # (b) amount scale: the one real feature carried across the shared schema.
    for i, (rec, colour) in enumerate(zip(recs, colours, strict=True)):
        a = rec["amount"]
        y = len(recs) - 1 - i
        ax2.plot([a["q05"], a["q95"]], [y, y], color=colour, linewidth=5.0,
                 alpha=0.45, solid_capstyle="butt")
        ax2.plot([a["q25"], a["q75"]], [y, y], color=colour, linewidth=5.0,
                 alpha=0.85, solid_capstyle="butt")
        ax2.plot(a["q50"], y, marker="|", color="white", markersize=7,
                 markeredgewidth=1.4, linestyle="none")
        ax2.annotate(f"median {a['q50']:,.0f}", xy=(a["q95"], y), xytext=(6, 2),
                     textcoords="offset points", fontsize=6, color=colour)
    ax2.set_yticks(range(len(recs)))
    ax2.set_yticklabels(list(reversed(labels)), fontsize=6)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 1e9)
    ax2.set_xlabel("transfer amount (log scale); bar q25-q75, line q05-q95")
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.set_title("(b) amount scale per source")

    # (c) the neighbourhood a graph detector actually gets to aggregate over.
    degs = [r["mean_degree"] for r in recs]
    ax3.bar(xs, degs, color=colours, hatch=hatches, edgecolor="white",
            linewidth=0.6, width=0.62)
    for x, r in zip(xs, recs, strict=True):
        ax3.annotate(f"{r['mean_degree']:.1f}\n({r['n_accounts']:,} acc.)",
                     xy=(x, r["mean_degree"]), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=5.5)
    ax3.set_yscale("log")
    ax3.set_ylim(1, 200)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(labels, fontsize=6)
    ax3.set_ylabel("mean account degree (log scale)")
    ax3.grid(False, axis="x")
    ax3.set_title("(c) account-graph neighbourhood")

    fig.tight_layout()
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
    fig_deadline_clock(events, results, out_dir / "fig_deadline_clock.pdf")
    fig_dataset(events, out_dir / "fig_dataset.pdf")
    fig_graph(events, config, out_dir / "fig_graph.pdf")
    stats_path = results / "dataset_stats.json"
    if stats_path.exists():
        fig_sources(stats_path, out_dir / "fig_sources.pdf")
    else:
        print(f"skipped fig_sources: run make_source_stats.py first ({stats_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
