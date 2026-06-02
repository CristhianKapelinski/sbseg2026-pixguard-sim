#!/usr/bin/env bash
# Minimal end-to-end test: runs the designated main-claim experiment (E1) on a
# freshly generated synthetic PIX stream and prints the headline result. This
# exercises the real pipeline (generate -> fit detectors -> latency-aware
# scoring -> metrics with CIs), not a --help stub. It needs no external data.
#
# Uses the reduced "fast" configuration so the whole pipeline finishes in a few
# seconds; the discriminative result (the rank inversion) is identical to the
# full run. Pass --full to use the paper's full configuration instead.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/fast.json"
if [[ "${1:-}" == "--full" ]]; then
  CONFIG="configs/default.json"
fi

uv run pixguard-sim --config "$CONFIG" reproduce

echo
echo "=== E1 headline: pre-deadline flag fraction separates equal-batch detectors ==="
uv run python - <<'PY'
import json
d = json.load(open("results/e1.json"))
print(f"stream: {d['n_events']} events, {d['n_fraud']} frauds")
print(f"{'detector':16s} {'F1':>6s} {'pre@1000ms':>11s} {'pre@5000ms':>11s}")
for det in d["detectors"]:
    name = det["detector"]
    f1 = det["batch"]["f1"]
    p1 = det["pre_deadline_fraction"]["1000"]["value"]
    p5 = det["pre_deadline_fraction"]["5000"]["value"]
    print(f"{name:16s} {f1:6.3f} {p1:11.3f} {p5:11.3f}")
print()
print("Expected: rf_fast and gb_slow have near-identical batch F1, yet rf_fast")
print("flags a high fraction of frauds at the 1000 ms deadline while gb_slow")
print("flags 0.000 until the deadline reaches its 5000 ms inference budget.")
PY
