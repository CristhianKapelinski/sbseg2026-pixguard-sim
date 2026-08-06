#!/usr/bin/env bash
# Cheap latency check against a hosted model, for an evaluator who wants to see the
# deadline metric bind on a real network round trip without paying for the paper's
# full run.
#
# The paper's E9 scores 1000 events through two reasoning models. This scores TEN,
# through the cheaper flash model, which is enough to measure per-request latency
# against the regulator's 1.5 s budget: latency is a property of the round trip, not
# of how many events you send. Accuracy on ten events is not, so this reports the
# timing and deliberately does not gate on PR-AUC.
#
# It needs a credential, which the evaluator supplies and this repository never
# stores. Either point PIXGUARD_LLM_ENDPOINT at a provider and set
# PIXGUARD_LLM_API_KEY, or leave both unset and run a local forwarder on
# 127.0.0.1:8080 that injects your key.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_claim_common.sh"

MODEL="${PIXGUARD_LLM_MODEL:-deepseek-v4-flash}"
N="${PIXGUARD_LLM_SMOKE_N:-10}"

if [[ -z "${PIXGUARD_LLM_API_KEY:-}" && -z "${PIXGUARD_LLM_ENDPOINT:-}" ]]; then
    echo "note: no PIXGUARD_LLM_API_KEY and no PIXGUARD_LLM_ENDPOINT set, so this will"
    echo "      call the default local forwarder at 127.0.0.1:8080. Set the two"
    echo "      variables to call a provider directly with your own key."
    echo
fi

mkdir -p "$LIVE_DIR"
echo "==> Scoring $N events through $MODEL (cheap latency check, not the paper's E9)"
_run_measured uv run python scripts/run_hosted_llm.py \
    --models "$MODEL" --n "$N" --fraud 3 --out "$LIVE_DIR/e9_smoke.json"

uv run python scripts/show_llm_smoke.py "$LIVE_DIR/e9_smoke.json"
