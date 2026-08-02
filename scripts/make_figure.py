"""Regenerate the paper's results figure from the real experiment outputs.

Reads the harness JSON outputs and writes a compact three-panel PDF:
(a) the pre-deadline flag fraction vs deadline for the E1 detectors, the
headline discriminative curve; (b) per-scenario recall of single-hop-trained
detectors from E2, showing the collapse on the PIX-native scenarios; (c) PR-AUC
with 95% bootstrap CIs across the in-repo, pix-fraud-br, and real Tide HI/LI
generators (E3/E5), showing the sub-perfect, spread-out cross-generator
behaviour. The figure plots only numbers the harness computed; it fabricates
nothing.

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


def _pdf_value(entry: object) -> float:
    """Pre-deadline entries are now ``{value, ci_lo, ...}`` dicts."""
    return entry["value"] if isinstance(entry, dict) else float(entry)


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    out = Path(argv[2]) if len(argv) > 2 else Path("paper/figs/fig_results.pdf")

    e1 = json.loads((results_dir / "e1.json").read_text(encoding="utf-8"))
    e2 = json.loads((results_dir / "e2.json").read_text(encoding="utf-8"))
    e3 = json.loads((results_dir / "e3.json").read_text(encoding="utf-8"))
    e5 = json.loads((results_dir / "e5.json").read_text(encoding="utf-8"))
    e8 = json.loads((results_dir / "e8.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.7))

    # Panel (a): pre-deadline flag fraction vs deadline under REAL measured
    # latency (E8). The reasoning LLM sits at zero until its latency clears the
    # window; the fast detectors plateau at their recall.
    label_map = {"rf_fast": "RF", "llm_terse": "Terse LM",
                 "llm_reasoning": "Reasoning LM"}
    e8_deadlines = e8.get("deadline_sweep_ms", [200, 1000, 2000, 5000])
    markers = {"rf_fast": "^", "llm_terse": "s", "llm_reasoning": "o"}
    for det in e8["detectors"]:
        pdf = det["pre_deadline_fraction"]
        y = [_pdf_value(pdf[str(d)]) for d in e8_deadlines]
        name = label_map.get(det["detector"], det["detector"])
        ax1.plot(e8_deadlines, y, marker=markers.get(det["detector"], "o"),
                 label=name, linewidth=1.3)
    ax1.set_xscale("log")
    # Label only the deadlines actually swept; matplotlib's default decade
    # minor ticks collide at this figure width.
    ax1.set_xticks(e8_deadlines)
    ax1.set_xticklabels([str(d) for d in e8_deadlines], fontsize=7)
    ax1.set_xticks([], minor=True)
    ax1.set_xlabel("decision deadline (ms, log scale)")
    ax1.set_ylabel("pre-deadline flag fraction")
    ax1.set_ylim(-0.03, 1.03)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.4)
    ax1.legend(fontsize=7, loc="center left")
    ax1.set_title("(a) measured latency vs. deadline", fontsize=9)

    # Panel (b): per-scenario recall, single-hop-trained (E2), with Wilson CIs.
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
    ax2.bar(labels, recalls, color="#444", width=0.6,
            yerr=[los, his], capsize=3, ecolor="#999")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("recall (single-hop trained)")
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax2.set_title("(b) recall per scenario", fontsize=9)

    # Panel (c): PR-AUC with 95% CI across generators (in-repo, pfb, Tide HI/LI).
    def _best_prauc(detectors: list[dict]) -> tuple[float, float, float]:
        best = max(detectors, key=lambda d: d["batch"]["pr_auc"])
        b = best["batch"]
        return b["pr_auc"], b["pr_auc_ci_lo"], b["pr_auc_ci_hi"]

    gens = ["ours", "pix-fraud-br", "Tide-HI", "Tide-LI"]
    pr_inrepo = _best_prauc(e1["detectors"])
    pr_pfb = _best_prauc(e5["detectors"])
    pr_hi = _best_prauc(e3["splits"]["HI"]["detectors"])
    pr_li = _best_prauc(e3["splits"]["LI"]["detectors"])
    vals = [pr_inrepo[0], pr_pfb[0], pr_hi[0], pr_li[0]]
    pr_all = [pr_inrepo, pr_pfb, pr_hi, pr_li]
    los = [v - p[1] for v, p in zip(vals, pr_all, strict=True)]
    his = [p[2] - v for v, p in zip(vals, pr_all, strict=True)]
    ax3.bar(gens, vals, color="#3b6", width=0.6,
            yerr=[los, his], capsize=3, ecolor="#777")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("best PR-AUC (95% CI)")
    ax3.grid(True, axis="y", linestyle=":", linewidth=0.4)
    ax3.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax3.set_title("(c) PR-AUC per source", fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
