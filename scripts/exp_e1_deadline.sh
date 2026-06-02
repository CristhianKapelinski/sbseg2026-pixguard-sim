#!/usr/bin/env bash
# E1 (MAIN CLAIM): the pre-deadline flag fraction separates two detectors that
# batch precision/recall/F1 rank as equivalent. Runs the in-repo synthetic
# pipeline end to end and writes results/e1.json, then prints the headline.
#
# Default: reduced "fast" configuration (a few seconds). The rank inversion is
# identical to the full run. Pass --full for the paper's full configuration
# (seed 20260202, 40k legit events, 1000 bootstrap resamples; ~25 s).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/fast.json"
if [[ "${1:-}" == "--full" ]]; then
  CONFIG="configs/default.json"
fi

uv run pixguard-sim --config "$CONFIG" run --experiments E1

echo
echo "=== E1: F1 vs pre-deadline flag fraction (rule_threshold / lr / rf / gb_slow) ==="
uv run python - <<'PY'
import json
d = json.load(open("results/e1.json"))
print(f"stream: {d['n_events']} events, {d['n_fraud']} frauds")
print(f"{'detector':16s} {'F1':>6s} {'PR-AUC':>7s} {'pre@1000':>9s} {'pre@5000':>9s}")
for det in d["detectors"]:
    b = det["batch"]
    p1 = det["pre_deadline_fraction"]["1000"]["value"]
    p5 = det["pre_deadline_fraction"]["5000"]["value"]
    print(f"{det['detector']:16s} {b['f1']:6.3f} {b['pr_auc']:7.3f} {p1:9.3f} {p5:9.3f}")
PY
