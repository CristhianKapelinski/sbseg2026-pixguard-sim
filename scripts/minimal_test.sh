#!/usr/bin/env bash
# Minimal end-to-end test: runs the designated main-claim experiment (E1) on a
# freshly generated synthetic PIX stream and prints the headline result. This
# exercises the real pipeline (generate -> fit detectors -> latency-aware
# scoring -> metrics with CIs), not a --help stub. It needs no external data.
#
# Uses the reduced "fast" configuration so the whole pipeline finishes in a few
# seconds. Pass --full to use the paper's full configuration instead.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/fast.json"
if [[ "${1:-}" == "--full" ]]; then
  CONFIG="configs/default.json"
fi

uv run pixguard-sim --config "$CONFIG" reproduce

echo
echo "=== E1: F1 vs pre-deadline flag fraction over the tabular baseline detectors ==="
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
print("Expected: all four tabular detectors score each event in well under 1 ms,")
print("so every detector's pre-deadline flag fraction equals its recall (none is")
print("slow enough to miss the deadline). The deadline metric separates a detector")
print("from accuracy only once it is genuinely slow; see the LLM-latency study (E8)")
print("in the README.")
PY

echo
echo "=== Unit tests ==="
uv run --extra dev pytest -q
