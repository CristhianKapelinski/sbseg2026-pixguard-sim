#!/usr/bin/env bash
# Shared by claim1.sh, claim2.sh and claim3.sh. Sourced, never executed.
#
# The clock starts HERE, when the calling script sources this file, so the reported
# wall clock covers the experiment and not just the printing that follows it.
_CLAIM_T0=$(date +%s)

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The live run writes here, never into results/published/. That directory is the
# paper's committed campaign and is what every claim compares against; if a claim
# wrote into it, the comparison would be the file against itself and would pass on
# any machine, measuring nothing.
LIVE_DIR="results/claim_run"
PUBLISHED_DIR="results/published"

# Peak RSS of the whole claim, when /usr/bin/time is available. It is reported, never
# gated: it belongs to this machine. Absent, the line is simply omitted.
_run_measured() {
    local peak_file="$LIVE_DIR/.peak"
    if command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -f "%M" -o "$peak_file" "$@"
        PIXGUARD_CLAIM_PEAK_KB=$(cat "$peak_file" 2>/dev/null || true)
        rm -f -- "$peak_file"
    else
        "$@"
    fi
}

# Runs the requested experiments into the live directory. Reviewers get the paper's
# own configuration by default: it costs seconds here, and a reduced config would
# produce numbers that legitimately differ from the paper, which is exactly the
# comparison the framed block is making.
recompute() {
    mkdir -p "$LIVE_DIR"
    echo "==> Recomputing on this machine (writes $LIVE_DIR)"
    _run_measured uv run pixguard-sim --config configs/default.json \
        --results-dir "$LIVE_DIR" run --experiments "$@"
    echo
}

report() {
    local number="$1" src="$2"
    PIXGUARD_CLAIM_SRC="$src" \
    PIXGUARD_CLAIM_ELAPSED="$(( $(date +%s) - _CLAIM_T0 ))" \
    PIXGUARD_CLAIM_PEAK_KB="${PIXGUARD_CLAIM_PEAK_KB:-}" \
        uv run python scripts/show_claim.py "$number" "$src"
}
