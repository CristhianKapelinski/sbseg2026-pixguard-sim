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



def _thousands(n: int) -> str:
    """Group digits with a thin space from five digits up; four-digit numbers stay solid.

    Mixing "1\\,000" with a solid "1066" inside one paper reads as a typo, so the rule
    is applied here rather than by hand in the prose.
    """
    return f"{n:,}".replace(",", "\\,") if n >= 10000 else str(n)

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
    cmd("Nevents", _thousands(e1['n_events']))
    cmd("Nfraud", str(e1["n_fraud"]))
    cmd("Frate", f"{100 * e1['n_fraud'] / e1['n_events']:.2f}\\%")
    e1d = {o["detector"]: o for o in e1["detectors"]}
    cmd("Neval", _thousands(e1d['rf_fast']['batch']['n_eval']))
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
    cmd("EightN", _thousands(e8['n_subsample']))
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

    # The paper reads every detector against one deadline, the regulator's 1.5 s
    # authorization budget. The sweep does not evaluate 1500 ms directly, but the
    # slowest local decision is well under 1000 ms, so each local detector's
    # fraction at 1500 ms is exactly its fraction at 1000 ms. Assert that rather
    # than assume it: if a rerun ever pushes a local latency past 1000 ms, this
    # stops instead of quietly publishing the wrong deadline.
    for det in ("rf_fast", "llm_terse", "llm_reasoning"):
        slowest = e8d[det].get("per_event_latency_stats_ms")
        slowest = slowest["max_ms"] if slowest else e8d[det]["measured_mean_latency_ms"]
        assert slowest < 1000.0, (
            f"{det} decides in {slowest:.0f} ms; the 1000 ms sweep point no longer "
            "stands in for the 1.5 s authorization budget"
        )
    cmd("ErfPreBudget", f3(prrf["1000"]["value"]))
    cmd("EtersePreBudget", f3(e8d["llm_terse"]["pre_deadline_fraction"]["1000"]["value"]))
    cmd("EreasonPreBudget", f3(pr["1000"]["value"]))

    # E5 pix-fraud-br reproduction (PIX-native external anchor).
    cmd("PfbN", _thousands(e5['n_events']))
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
    cmd("PfbRuleF", f3(e5d["rule_threshold"]["batch"]["f1"]))
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
    cmd("PfbSharedF", f3(t["pfb_to_pfb"]["batch"]["f1"]))
    cmd("PfbSharedRec", f3(t["pfb_to_pfb"]["batch"]["recall"]))
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

    # -- hardware normalization. Raw milliseconds describe one machine; the
    # share of the regulator's authorization budget and the throughput it
    # implies transfer, because the denominator is published rather than ours.
    # (Manual de Tempos do Pix v7.0: 1.5 s at the 95th percentile for the
    # paying institution to authorize at initiation.)
    AUTH_BUDGET_MS = 1500.0

    def _budget(ms: float) -> str:
        return f"{100.0 * ms / AUTH_BUDGET_MS:.1f}\\%"

    def _throughput(ms: float) -> str:
        if ms <= 0:
            return "--"
        v = 1000.0 / ms
        # Below one decision per second, rounding to an integer prints "0".
        return f"{v:.1f}" if v < 10 else _thousands(int(round(v)))

    e8d = {o["detector"]: o for o in e8["detectors"]}
    cmd("RfThroughput", _throughput(max(e8d["rf_fast"]["measured_mean_latency_ms"], 1e-6)))
    for key, pre in (("llm_terse", "Terse"), ("llm_reasoning", "Reason")):
        s = e8d[key]["per_event_latency_stats_ms"]
        cmd(pre + "Budget", _budget(s["p95_ms"]))
        cmd(pre + "Throughput", _throughput(s["p95_ms"]))

    # The budget expressed in the unit that actually governs an LLM's cost.
    # The marginal cost per generated token is read off the two regimes, which
    # differ only in how many tokens they emit, so it isolates generation from
    # the fixed prefill both pay.
    tt = e8d["llm_terse"].get("new_tokens_median")
    tr = e8d["llm_reasoning"].get("new_tokens_median")
    if tt and tr and tr != tt:
        lt = e8d["llm_terse"]["per_event_latency_stats_ms"]["median_ms"]
        lr = e8d["llm_reasoning"]["per_event_latency_stats_ms"]["median_ms"]
        per_token = (lr - lt) / (tr - tt)
        if per_token > 0:
            cmd("MsPerToken", f"{per_token:.0f}")
            cmd("BudgetTokens", str(int(round(AUTH_BUDGET_MS / per_token))))
        cmd("ReasonTokens", str(int(tr)))
        cmd("TerseTokens", str(int(tt)))

    # The same detector on a second device, when that run is present.
    cpu_path = root / "e8_cpu.json"
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
        c = {o["detector"]: o for o in cpu["detectors"]}["llm_reasoning"]
        s = c["per_event_latency_stats_ms"]
        cmd("ReasonCpuLat", str(int(round(s["mean_ms"]))))
        cmd("ReasonCpuLatPgo", str(int(round(s["p95_ms"]))))
        cmd("ReasonCpuLatMin", str(int(round(s["min_ms"]))))
        cmd("ReasonCpuLatMax", str(int(round(s["max_ms"]))))
        cmd("ReasonCpuBudget", _budget(s["p95_ms"]))
        cmd("ReasonCpuThroughput", _throughput(s["p95_ms"]))
        pdf = c["pre_deadline_fraction"]["1000"]
        cmd("ReasonCpuPre", f3(pdf["value"] if isinstance(pdf, dict) else pdf))

    # -- hosted reasoning models (e9_hosted.json). Same slice as E8, scored by
    # models called over the network the way an institution would call them.
    e9_path = root / "e9_hosted.json"
    if e9_path.exists():
        e9 = json.loads(e9_path.read_text(encoding="utf-8"))
        nf = e9["n_fraud_subsample"]
        pre = {"deepseek-v4-flash": "Flash", "deepseek-v4-pro": "Pro"}
        for det in e9["detectors"]:
            k = pre.get(det["detector"])
            if not k:
                continue
            b, s = det["batch"], det["per_event_latency_stats_ms"]
            cmd(k + "PRAUC", f3(b["pr_auc"]))
            cmd(k + "Prec", f3(b["precision"]))
            cmd(k + "Rec", f3(b["recall"]))
            cmd(k + "Lat", _thousands(int(round(s["median_ms"]))))
            cmd(k + "LatPgo", _thousands(int(round(s["p95_ms"]))))
            cmd(k + "Budget", f"{100 * det['budget_share_p95']:.0f}\\%")
            cmd(k + "Tokens", str(det["new_tokens_median"]))
            cmd(k + "Throughput", _throughput(s["p95_ms"]))
            cmd(k + "Pre", f3(det["pre_deadline_1500ms"]["value"]))
            # Counts say what a rate cannot: how many frauds this would catch.
            tp = round(b["recall"] * nf)
            cmd(k + "TP", str(tp))
            cmd(k + "FP", str(round(tp / b["precision"]) - tp if b["precision"] else 0))
        rf = {o["detector"]: o for o in e8["detectors"]}["rf_fast"]["batch"]
        rftp = round(rf["recall"] * nf)
        cmd("RfTP", str(rftp))
        cmd("RfFP", str(round(rftp / rf["precision"]) - rftp if rf["precision"] else 0))
        # The chance level a PR-AUC on this slice must be read against, and the
        # denominator the false alarms are counted out of.
        cmd("EightBaseSlice", f3(nf / e9["n_subsample"]))
        cmd("EightLegit", _thousands(e9["n_subsample"] - nf))

    # -- dataset description (dataset_stats.json, written by
    # make_source_stats.py). These describe the streams the detectors were
    # scored on rather than any detector's output.
    stats_path = root / "dataset_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        w = s["ours"]["settlement_window_ms"]
        cmd("SettleMin", _thousands(int(round(w["min"]))))
        cmd("SettleMax", _thousands(int(round(w["max"]))))
        cmd("SettleMed", _thousands(int(round(w["q50"]))))
        # Median transfer amount per source, on the slice the harness scores.
        # The amount is the one real feature that survives the shared schema,
        # so its scale is part of why a score does not transfer across sources.
        for key, name in (("ours", "OursMedAmt"), ("pix_fraud_br", "PfbMedAmt"),
                          ("tide_hi", "TideHiMedAmt")):
            rec = s[key].get("as_scored", s[key])
            cmd(name, _thousands(int(round(rec["amount"]["q50"]))))
        # Mean account degree, on that same slice: the neighbourhood a graph
        # detector has to aggregate over.
        for key, name in (("ours", "OursDeg"), ("tide_hi", "TideHiDeg")):
            rec = s[key].get("as_scored", s[key])
            cmd(name, f"{rec['mean_degree']:.0f}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
