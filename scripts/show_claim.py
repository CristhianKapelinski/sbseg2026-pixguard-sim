#!/usr/bin/env python3
"""Print one claim as a framed block: measured value beside the paper's, with a verdict.

The paper's side of every comparison is read from ``expected/paper_macros.tex``, the
frozen camera-ready macro block, so a claim can never drift from the paper without
this failing. The measured side is read from whichever results directory the calling
script points at, and the header says which one that was: a number recomputed on the
evaluator's machine and a number replayed from the committed campaign are both
legitimate, but the evaluator must never have to guess which they are looking at.

Wall clock and peak memory are reported and never gated, because they belong to the
machine rather than to the claim. Exits non-zero if any gated value differs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAR = "\u2550" * 66
SEP = "\u2500" * 66
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", re.M)


def paper_values() -> dict[str, str]:
    """The camera-ready's own numbers, as the macro block defines them."""
    text = (ROOT / "expected" / "paper_macros.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in MACRO.finditer(text)}


def f3(x: float) -> str:
    """Format exactly as make_macros.py does, so a comparison is like for like."""
    return f"{float(x):.3f}"


def load(results: Path, name: str) -> dict:
    return json.loads((results / f"{name}.json").read_text(encoding="utf-8"))


def by_detector(block: dict) -> dict:
    return {o["detector"]: o for o in block["detectors"]}


def claim1(results: Path, paper: dict) -> tuple[str, list[tuple]]:
    """E1: the deadline metric is driven by measured latency, not by accuracy."""
    e1 = load(results, "e1")
    d = by_detector(e1)
    rows: list[tuple] = []
    for det, pre, shown in (("rule_threshold", "rule", "rule_threshold"),
                            ("lr_fast", "lr", "lr_fast"),
                            ("rf_fast", "rf", "rf_fast")):
        b = d[det]["batch"]
        rows.append((f"{shown} F1", f3(b["f1"]), paper[pre + "F"], True))
        rows.append((f"{shown} PR-AUC", f3(b["pr_auc"]), paper[pre + "PRAUC"], True))
    rows.append(("rf_fast recall", f3(d["rf_fast"]["batch"]["recall"]), paper["rfRec"], True))

    # The mechanism of the claim, and the part that holds on any hardware: these
    # detectors are fast enough that the deadline never binds, so the pre-deadline
    # fraction collapses onto recall. It is what makes the LLM result meaningful.
    equal = all(
        abs(o["pre_deadline_fraction"]["1000"]["value"] - o["batch"]["recall"]) < 5e-4
        for o in e1["detectors"]
    )
    rows.append(("pre@1000ms equals recall", "yes" if equal else "no", "yes", True))
    slowest = max(o.get("measured_latency_ms", 0.0) for o in e1["detectors"])
    rows.append(("slowest per-event latency (ms)", f"{slowest:.4f}", None, False))
    return "The deadline metric is driven by measured latency", rows


def claim2(results: Path, paper: dict) -> tuple[str, list[tuple]]:
    """E2: a single-hop-trained detector collapses on the Pix-native scenarios."""
    e2 = load(results, "e2")
    sc = by_detector(e2)["rf_single_hop"]["per_scenario"]
    rows = []
    for scen, pre in (("account_takeover", "eAto"), ("mule_chain", "eMule"),
                      ("fake_med_refund", "eMed"), ("coercion", "eCoerc")):
        r = sc[scen]["recall"]
        rows.append((f"recall, {scen}", f3(r["value"]), paper[pre], True))
        rows.append((f"  N, {scen}", str(r["n"]), paper[pre + "N"], True))
    return "Single-hop training collapses on the Pix-native scenarios", rows


def claim3(results: Path, paper: dict) -> tuple[str, list[tuple]]:
    """E3/E5/E6: the harness holds up on independently authored generators."""
    e3, e5, e6 = load(results, "e3"), load(results, "e5"), load(results, "e6")
    e5d = by_detector(e5)
    t = {o["name"]: o for o in e6["transfers"]}
    hi = {o["detector"]: o for o in e3["splits"]["HI"]["detectors"]}
    li = {o["detector"]: o for o in e3["splits"]["LI"]["detectors"]}
    rows = [
        ("pix-fraud-br XGBoost PR-AUC", f3(e5d["xgb_fast"]["batch"]["pr_auc"]),
         paper["PfbXgbPRAUC"], True),
        ("  its published baseline", f3(e5["published_baselines_prauc"]["xgboost"]),
         paper["PfbPubPRAUC"], True),
        ("  on shared columns only", f3(t["pfb_to_pfb"]["batch"]["pr_auc"]),
         paper["TransPfbSelf"], True),
        ("Tide HI XGBoost PR-AUC", f3(hi["xgb_fast"]["batch"]["pr_auc"]),
         paper["TideHiXgbPRAUC"], True),
        ("Tide LI XGBoost PR-AUC", f3(li["xgb_fast"]["batch"]["pr_auc"]),
         paper["TideLiXgbPRAUC"], True),
        ("transfer ours -> pix-fraud-br", f3(t["inrepo_to_pfb"]["batch"]["pr_auc"]),
         paper["TransInPfb"], True),
        ("transfer pix-fraud-br -> ours", f3(t["pfb_to_inrepo"]["batch"]["pr_auc"]),
         paper["TransPfbIn"], True),
        ("in-distribution, ours", f3(t["inrepo_to_inrepo"]["batch"]["pr_auc"]),
         paper["TransInSelf"], True),
    ]
    return "Cross-generator credibility on three generators", rows


CLAIMS = {1: claim1, 2: claim2, 3: claim3}
MAIN = 1


def main() -> int:
    number = int(sys.argv[1])
    results = Path(sys.argv[2])
    title, rows = CLAIMS[number](results, paper_values())

    print()
    print(BAR)
    print(f"  Claim #{number}  {title}" + ("  (MAIN CLAIM)" if number == MAIN else ""))
    print(SEP)
    failed = gated = 0
    for label, got, want, is_gated in rows:
        if not is_gated or want is None:
            print(f"  {label:31s}: {got:<12s}")
            continue
        gated += 1
        ok = got == want
        failed += not ok
        print(f"  {label:31s}: {got:<12s} (paper {want})".ljust(66) + ("OK" if ok else "FAIL"))
    print(SEP)

    src = Path(os.environ.get("PIXGUARD_CLAIM_SRC", str(results)))
    if src.name == "claim_run":
        print(f"  {'source of these numbers':<31}: recomputed on this machine just now")
    else:
        print(f"  {'source of these numbers':<31}: read from the committed campaign")
        print(f"  {'':<31}  ({src}); pass --run to regenerate it here")
    elapsed = os.environ.get("PIXGUARD_CLAIM_ELAPSED")
    peak_kb = os.environ.get("PIXGUARD_CLAIM_PEAK_KB")
    if elapsed:
        print(f"  {'wall clock on this machine':<31}: {elapsed} s")
    if peak_kb:
        print(f"  {'peak memory on this machine':<31}: {int(peak_kb) // 1024} MB")
    print(SEP)
    verdict = "FAIL" if failed else "OK"
    print(f"  RESULT: {verdict}   ({gated - failed}/{gated} gated values match the paper)")
    print(BAR)
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
