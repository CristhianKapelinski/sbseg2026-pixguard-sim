#!/usr/bin/env bash
# E2: a detector trained only on the single-hop case (legitimate traffic plus
# account-takeover, the modeling scope of the open Pix generator) keeps full
# recall there but collapses on the two Pix-native scenarios no open artifact
# covers: coercion and multi-hop MED-2.0 refunds. Runs in-repo, no external
# data; writes results/e2.json and prints per-scenario recall.
#
# Default: reduced "fast" configuration. Pass --full for the paper's config.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/fast.json"
if [[ "${1:-}" == "--full" ]]; then
  CONFIG="configs/default.json"
fi

uv run pixguard-sim --config "$CONFIG" run --experiments E2

echo
echo "=== E2: per-scenario recall of single-hop-trained detectors ==="
uv run python - <<'PY'
import json
d = json.load(open("results/e2.json"))
for det in d["detectors"]:
    print(det["detector"])
    for sc, v in sorted(det["per_scenario"].items()):
        r = v["recall"]
        print(f"  {sc:18s} recall={r['value']:.3f}  N={r['n']}")
print()
print("Expected: full recall on account_takeover, ~0.000 on coercion and on")
print("fake_med_refund (multi-hop MED-2.0), partial recall on mule_chain.")
PY
