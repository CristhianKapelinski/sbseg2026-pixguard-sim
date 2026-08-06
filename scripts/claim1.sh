#!/usr/bin/env bash
# Claim #1 (MAIN): the deadline metric is driven by each detector's measured
# latency, so it collapses onto recall for the sub-millisecond tabular detectors
# and only separates accuracy from deployability once a detector deliberates.
# Recomputes E1 here with the paper's configuration and gates every value it
# prints against the frozen camera-ready macros.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_claim_common.sh"

recompute E1
report 1 "$LIVE_DIR"
