#!/usr/bin/env python3
"""Emit the LaTeX results-macro block for PixGuard-Sim from results/*.json.

Every empirical number in the paper is a \\newcommand produced here and read
from results/{e1,e2,e3,e5,e6,e7,e8}.json; the prose hardcodes no number, so a
data refresh is a one-command regeneration. Run after the experiments and write
the output to paper/macros.tex (preserving the macro-definitions header).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    def load(name: str) -> dict:
        return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))

    e1, e2, e3 = load("e1"), load("e2"), load("e3")
    e5, e6, e7, e8 = load("e5"), load("e6"), load("e7"), load("e8")

    lines: list[str] = []

    def cmd(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def f3(x) -> str:
        return f"{float(x):.3f}"

    def by_name(items, key, val):
        return next(o for o in items if o.get(key) == val)

    # -- in-house PIX layer counts (E1) --
    cmd("Nevents", f"{e1['n_events']:,}".replace(",", "\\,"))
    cmd("Nfraud", str(e1["n_fraud"]))
    cmd("Frate", f"{100 * e1['n_fraud'] / e1['n_events']:.2f}\\%")
    e1d = {o["detector"]: o for o in e1["detectors"]}
    cmd("Neval", f"{e1d['rf_fast']['batch']['n_eval']:,}".replace(",", "\\,"))
    cmd("Nevalfraud", str(e1d["rf_fast"]["batch"]["n_fraud"]))

    # E1 tabular detectors on the in-house layer (measured latency is sub-us for
    # all, so the pre-deadline fraction equals recall at every deadline).
    for det, pre in (("rule_threshold", "rule"), ("lr_fast", "lr"),
                     ("rf_fast", "rf")):
        b = e1d[det]["batch"]
        cmd(pre + "F", f3(b["f1"]))
        cmd(pre + "PRAUC", f3(b["pr_auc"]))
        cmd(pre + "Rec", f3(b["recall"]))
        cmd(pre + "Prec", f3(b["precision"]))
    cmd("rfRecLo", f3(e1d["rf_fast"]["batch"]["recall_ci_lo"]))
    cmd("rfRecHi", f3(e1d["rf_fast"]["batch"]["recall_ci_hi"]))
    cmd("rfPRAUClo", f3(e1d["rf_fast"]["batch"]["pr_auc_ci_lo"]))
    cmd("rfPRAUChi", f3(e1d["rf_fast"]["batch"]["pr_auc_ci_hi"]))

    # E2 per-scenario recall of a single-hop-trained detector (rf_single_hop).
    sc = by_name(e2["detectors"], "detector", "rf_single_hop")["per_scenario"]
    name_map = {"account_takeover": "eAto", "coercion": "eCoerc",
                "fake_med_refund": "eMed", "mule_chain": "eMule"}
    for scen, pre in name_map.items():
        r = sc[scen]["recall"]
        cmd(pre, f3(r["value"]))
        cmd(pre + "Lo", f3(r["ci_lo"]))
        cmd(pre + "Hi", f3(r["ci_hi"]))
        cmd(pre + "N", str(r["n"]))

    # E8 deadline metric under real measured latency: a reasoning LLM vs a fast
    # tabular RF on the same fraud-enriched subsample.
    e8d = {o["detector"]: o for o in e8["detectors"]}
    cmd("EightN", f"{e8['n_subsample']:,}".replace(",", "\\,"))
    cmd("EightFraud", str(e8["n_fraud_subsample"]))
    cmd("EightBase", f"{100 * e8['operational_base_rate']:.2f}\\%")

    def e8stats(det, pre):
        b = e8d[det]["batch"]
        lt = e8d[det].get("per_event_latency_stats_ms") or {}
        cmd(pre + "PRAUC", f3(b["pr_auc"]))
        cmd(pre + "F", f3(b["f1"]))
        cmd(pre + "Rec", f3(b["recall"]))
        cmd(pre + "Prec", f3(b["precision"]))
        if lt:
            cmd(pre + "Lat", f"{lt['mean_ms']:.0f}")
            cmd(pre + "LatPgo", f"{lt['p95_ms']:.0f}")
            cmd(pre + "LatMin", f"{lt['min_ms']:.0f}")
            cmd(pre + "LatMax", f"{lt['max_ms']:.0f}")

    e8stats("rf_fast", "ErfFast")
    e8stats("llm_terse", "Eterse")
    e8stats("llm_reasoning", "Ereason")
    pr = e8d["llm_reasoning"]["pre_deadline_fraction"]
    # LaTeX command names cannot contain digits, so deadline keys are spelled out.
    word = {"200": "TwoHun", "1000": "OneK", "2000": "TwoK", "5000": "FiveK"}
    for d, w in word.items():
        cmd("EreasonPre" + w, f3(pr[d]["value"]))
    prrf = e8d["rf_fast"]["pre_deadline_fraction"]
    cmd("ErfPre", f3(prrf["1000"]["value"]))

    # E5 pix-fraud-br reproduction (PIX-native external anchor).
    cmd("PfbN", f"{e5['n_events']:,}".replace(",", "\\,"))
    cmd("PfbRate", f"{100 * e5['fraud_rate']:.2f}\\%")
    e5d = {o["detector"]: o for o in e5["detectors"]}
    for det, pre in (("xgb_fast", "PfbXgb"), ("rf_fast", "PfbRf")):
        b = e5d[det]["batch"]
        cmd(pre + "PRAUC", f3(b["pr_auc"]))
        cmd(pre + "PRAUClo", f3(b["pr_auc_ci_lo"]))
        cmd(pre + "PRAUChi", f3(b["pr_auc_ci_hi"]))
        cmd(pre + "F", f3(b["f1"]))
        cmd(pre + "Rec", f3(b["recall"]))
    cmd("PfbPubPRAUC", f3(e5["published_baselines_prauc"]["xgboost"]))
    cmd("PfbRulePRAUC", f3(e5d["rule_threshold"]["batch"]["pr_auc"]))
    cmd("PfbRuleRec", f3(e5d["rule_threshold"]["batch"]["recall"]))

    # E3 Tide HI/LI cross-generator (rare-illicit regime).
    for split, pre in (("HI", "TideHi"), ("LI", "TideLi")):
        dets = {o["detector"]: o for o in e3["splits"][split]["detectors"]}
        b = dets["xgb_fast"]["batch"]
        cmd(pre + "XgbPRAUC", f3(b["pr_auc"]))
        cmd(pre + "XgbPRAUClo", f3(b["pr_auc_ci_lo"]))
        cmd(pre + "XgbPRAUChi", f3(b["pr_auc_ci_hi"]))
        cmd(pre + "XgbR", f3(b["recall"]))
        cmd(pre + "XgbF", f3(b["f1"]))
        cmd(pre + "RulePRAUC", f3(dets["rule_threshold"]["batch"]["pr_auc"]))
        rate = e3["splits"][split].get("fraud_rate")
        if rate is not None:
            cmd(pre + "Rate", f"{100 * rate:.2f}\\%")

    # E6 cross-generator transfer (out-of-distribution collapse).
    t = {o["name"]: o for o in e6["transfers"]}
    cmd("TransInSelf", f3(t["inrepo_to_inrepo"]["batch"]["pr_auc"]))
    cmd("TransPfbSelf", f3(t["pfb_to_pfb"]["batch"]["pr_auc"]))
    cmd("TransInPfb", f3(t["inrepo_to_pfb"]["batch"]["pr_auc"]))
    cmd("TransPfbIn", f3(t["pfb_to_inrepo"]["batch"]["pr_auc"]))

    # E7 GraphSAGE GPU baseline.
    g_in = e7["datasets"]["inrepo"]["detectors"]["gnn_sage"]
    cmd("GnnInPRAUC", f3(g_in["batch"]["pr_auc"]))
    cmd("GnnInRec", f3(g_in["batch"]["recall"]))
    cmd("GnnFitS", f"{g_in['fit_seconds']:.1f}")
    if "tide_hi" in e7["datasets"]:
        g_t = e7["datasets"]["tide_hi"]["detectors"].get("gnn_sage")
        if g_t:
            cmd("GnnTideHiPRAUC", f3(g_t["batch"]["pr_auc"]))

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
