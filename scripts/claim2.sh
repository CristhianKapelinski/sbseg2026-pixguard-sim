#!/usr/bin/env bash
# Claim #2: a detector trained only on the single-hop case keeps its recall there
# and collapses on the two scenarios no open Pix artifact covers, coercion and
# multi-hop MED-2.0 refunds. Recomputes E2 here and gates against the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_claim_common.sh"

recompute E2
report 2 "$LIVE_DIR"
