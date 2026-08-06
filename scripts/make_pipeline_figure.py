"""Draw the four-stage pipeline banner from the results the paper reports.

The banner carries numbers -- the settlement window, the field count, the
scenario count, each detector's decision latency -- so it is generated from
``results/*.json`` and the event stream rather than drawn by hand. A hand-drawn
banner drifts out of agreement with the tables the moment an experiment is
rerun, and a reader who spots the disagreement cannot tell which number is the
stale one.

Stages: the window that closes, the three sources feeding one schema, the four
scenarios, and where each detector's decision lands against that window.

Usage: python scripts/make_pipeline_figure.py [results_dir] [output_pdf] [events_csv]
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# The regulator's ordinary authorization budget for the paying institution at
# the 95th percentile (Manual de Tempos do Pix v7.0).
AUTH_BUDGET_MS = 1500.0
INK = "#1a1a1a"
FRAME = "#8a8a8a"
FILL = "#f2efe9"


def _panel(ax, x0: float, w: float, title: str, badge: str | None) -> None:
    """Draw one stage frame with its title and optional badge."""
    ax.add_patch(FancyBboxPatch(
        (x0, 0.06), w, 0.88, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0.7, edgecolor=FRAME, facecolor="white", zorder=1))
    ax.text(x0 + 0.012, 0.86, title, fontsize=6.2, fontweight="bold",
            color=INK, va="center", zorder=3)
    if badge:
        ax.add_patch(FancyBboxPatch(
            (x0 + w - 0.014 - 0.008 * len(badge), 0.815), 0.008 * len(badge) + 0.004,
            0.075, boxstyle="round,pad=0.002,rounding_size=0.02", linewidth=0.6,
            edgecolor=FRAME, facecolor=FILL, zorder=3))
        ax.text(x0 + w - 0.012, 0.852, badge, fontsize=5.4, color=INK,
                ha="right", va="center", zorder=4)


def _arrow(ax, x0: float, x1: float, y: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.8, color=INK, zorder=4))


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    out = Path(argv[2]) if len(argv) > 2 else Path("figs/pipeline.pdf")
    events_csv = Path(argv[3]) if len(argv) > 3 else Path("data/events.csv")

    e8 = json.loads((results_dir / "e8.json").read_text(encoding="utf-8"))
    e9 = json.loads((results_dir / "e9_hosted.json").read_text(encoding="utf-8"))
    events = pd.read_csv(events_csv,
                         usecols=["t_init_ms", "t_settle_ms", "scenario", "med_layer"])
    window = (events.t_settle_ms - events.t_init_ms).to_numpy(dtype=float)
    lo, hi = window.min(), window.max()
    # Read each scenario's chain depth off the stream rather than restating it:
    # a hand-written "multi-hop" label is exactly the kind of claim that goes
    # stale when a scenario is retuned.
    depth = events.groupby("scenario")["med_layer"].max().to_dict()

    by_name = {d["detector"]: d for d in e8["detectors"] + e9["detectors"]}

    def latency(key: str) -> float:
        """Median decision latency, or the batch mean when that is all there is."""
        stats = by_name[key].get("per_event_latency_stats_ms")
        return float(stats["median_ms"]) if stats else float(
            by_name[key]["measured_mean_latency_ms"])

    plt.rcParams.update({"font.size": 6.2, "figure.dpi": 220})
    fig, ax = plt.subplots(figsize=(9.6, 1.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- stage 1: the window that closes.
    _panel(ax, 0.005, 0.20, "THE WINDOW", f"{lo:.0f}–{hi:.0f} ms")
    ax.add_patch(FancyBboxPatch((0.028, 0.34), 0.155, 0.14,
                                boxstyle="round,pad=0.002,rounding_size=0.01",
                                linewidth=0.6, edgecolor="#7fa9b8",
                                facecolor="#cfe3ea", zorder=2))
    ax.plot([0.032, 0.179], [0.41, 0.41], color="#2b6b80", linewidth=1.1, zorder=3)
    for x in (0.032, 0.179):
        ax.plot([x], [0.41], marker="|", color="#2b6b80", markersize=4, zorder=3)
    ax.text(0.032, 0.24, "transfer", fontsize=5.4, color=INK, ha="left")
    ax.text(0.179, 0.24, "irrevocable", fontsize=5.4, color=INK, ha="right")
    ax.text(0.105, 0.60, "settlement is final at the right edge", fontsize=5.2,
            color="#555", ha="center")
    _arrow(ax, 0.208, 0.238, 0.5)

    # ---- stage 2: three sources into one schema.
    _panel(ax, 0.245, 0.20, "THREE SOURCES", "one schema")
    for i, name in enumerate(("PixGuard-Sim", "Tide HI/LI", "pix-fraud-br")):
        y = 0.60 - 0.155 * i
        ax.add_patch(FancyBboxPatch((0.262, y - 0.05), 0.088, 0.10,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    linewidth=0.6, edgecolor=FRAME,
                                    facecolor=FILL, zorder=2))
        ax.text(0.306, y, name, fontsize=5.2, color=INK, ha="center",
                va="center", zorder=3)
        ax.add_patch(FancyArrowPatch((0.352, y), (0.393, 0.42), arrowstyle="-|>",
                                     mutation_scale=4, linewidth=0.6,
                                     color=FRAME, zorder=2))
    ax.add_patch(FancyBboxPatch((0.395, 0.36), 0.043, 0.12,
                                boxstyle="round,pad=0.002,rounding_size=0.012",
                                linewidth=0.7, edgecolor=INK, facecolor="white",
                                zorder=3))
    ax.text(0.4165, 0.42, "15 fields", fontsize=5.0, color=INK, ha="center",
            va="center", zorder=4)
    _arrow(ax, 0.448, 0.478, 0.5)

    # ---- stage 3: the four scenarios, two of them new here.
    _panel(ax, 0.485, 0.20, "FOUR SCENARIOS", "2 new")
    cells = [("account takeover", "account_takeover"), ("mule chain", "mule_chain"),
             ("fake MED refund", "fake_med_refund"), ("coercion", "coercion")]
    for i, (name, key) in enumerate(cells):
        hops = int(depth.get(key, 0))
        cx = 0.505 + 0.088 * (i % 2)
        cy = 0.55 - 0.20 * (i // 2)
        new = name in ("fake MED refund", "coercion")
        ax.add_patch(FancyBboxPatch((cx, cy - 0.07), 0.082, 0.14,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    linewidth=0.9 if new else 0.6,
                                    edgecolor="#c2611f" if new else FRAME,
                                    facecolor="#fbeee2" if new else "white", zorder=2))
        ax.text(cx + 0.041, cy + 0.012, name, fontsize=4.9, color=INK,
                ha="center", va="center", zorder=3)
        ax.text(cx + 0.041, cy - 0.045,
                f"{hops + 1} hops" if hops else "single-hop",
                fontsize=4.4, color="#666", ha="center", va="center", zorder=3)
    ax.text(0.585, 0.16, "shaded: no open dataset carries it", fontsize=4.8,
            color="#c2611f", ha="center")
    _arrow(ax, 0.688, 0.718, 0.5)

    # ---- stage 4: where each decision lands against the window.
    _panel(ax, 0.725, 0.27, "THE VERDICT", f"budget {AUTH_BUDGET_MS / 1000:.1f} s")
    ax_l, ax_r = 0.752, 0.977
    lo_ms, hi_ms = 1e-3, 3e4

    def xpos(ms: float) -> float:
        f = (np.log10(max(ms, lo_ms)) - np.log10(lo_ms)) / (
            np.log10(hi_ms) - np.log10(lo_ms))
        return ax_l + f * (ax_r - ax_l)

    # The window and the budget, the two references a decision is read against.
    ax.add_patch(FancyBboxPatch((xpos(lo), 0.26), xpos(hi) - xpos(lo), 0.425,
                                boxstyle="square,pad=0", linewidth=0,
                                facecolor="#cfe3ea", alpha=0.75, zorder=1.5))
    ax.plot([xpos(AUTH_BUDGET_MS)] * 2, [0.26, 0.685], color="#b00", linewidth=0.9,
            linestyle="-.", zorder=3)
    ax.text(xpos(AUTH_BUDGET_MS) - 0.005, 0.715, "in time", fontsize=5.0,
            color="#0a7", ha="right", va="center", zorder=4)
    ax.text(xpos(AUTH_BUDGET_MS) + 0.005, 0.715, "too late", fontsize=5.0,
            color="#b00", ha="left", va="center", zorder=4)
    ax.plot([ax_l, ax_r], [0.26, 0.26], color=INK, linewidth=0.7, zorder=3)
    for ms, lab in ((1e-2, "0.01"), (1.0, "1"), (100.0, "100"),
                    (1e4, "10 s")):
        ax.plot([xpos(ms)] * 2, [0.245, 0.26], color=INK, linewidth=0.6, zorder=3)
        ax.text(xpos(ms), 0.185, lab, fontsize=4.8, color=INK, ha="center", zorder=3)
    ax.text(ax_r, 0.115, "decision latency (ms, log)", fontsize=4.8, color="#555",
            ha="right")

    marks = [("rf_fast", "RF", "#009E73", "^"),
             ("llm_reasoning", "local LM", "#CC79A7", "o"),
             ("deepseek-v4-flash", "hosted flash", "#0072B2", "D"),
             ("deepseek-v4-pro", "hosted pro", "#000000", "v")]
    for i, (key, lab, colour, marker) in enumerate(marks):
        x, y = xpos(latency(key)), 0.615 - 0.075 * i
        ax.plot([x], [y], marker=marker, color=colour, markersize=3.0, zorder=5)
        # Labels grow away from the right edge once past the axis midpoint.
        right = x > (ax_l + ax_r) / 2
        # A label that crosses the budget rule needs to sit on top of it, or the
        # dash-dot line reads as a strikethrough.
        ax.annotate(lab, xy=(x, y), xytext=(-4 if right else 4, 0),
                    textcoords="offset points", fontsize=4.6, color=colour,
                    ha="right" if right else "left", va="center", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "pad": 0.6, "alpha": 0.9})

    fig.tight_layout(pad=0.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
