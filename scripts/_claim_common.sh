#!/usr/bin/env bash
# Shared by claim1.sh, claim2.sh and claim3.sh. Sourced, never executed.
#
# The clock starts HERE, when the calling script sources this file, so the reported
# wall clock covers the experiment and not just the printing that follows it.
_CLAIM_T0=$(date +%s)

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# uv drives every command below. Without this check the first `uv run` fails with the
# shell's own "command not found", which names no installer -- and the usual cause is
# not absence but PATH: the installer drops uv in ~/.local/bin, which the shell that
# ran it does not pick up until it is re-entered.
if ! command -v uv >/dev/null 2>&1; then
    {
        echo "missing required tool: uv"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo '  then, in THIS shell: export PATH="$HOME/.local/bin:$PATH"'
    } >&2
    exit 1
fi

# The live run writes here, never into results/published/. That directory is the
# paper's committed campaign and is what every claim compares against; if a claim
# wrote into it, the comparison would be the file against itself and would pass on
# any machine, measuring nothing.
LIVE_DIR="results/claim_run"
PUBLISHED_DIR="results/published"

# Peak RSS of the whole claim. It is reported, never gated: it belongs to this
# machine. Absent, the line is simply omitted.
#
# "%M" and "-o" are GNU time's spelling. The /usr/bin/time that ships with macOS
# and the BSDs is a different program that rejects -f outright and writes its own
# format to stderr, so the presence of the path says nothing about the flags: they
# are probed once, against a no-op, and the result decides. gtime is GNU time under
# the name Homebrew installs it as. If neither answers the probe, the claim still
# runs; only the memory line goes away.
_CLAIM_TIME_BIN=""
for _claim_time_candidate in /usr/bin/time gtime; do
    if command -v "$_claim_time_candidate" >/dev/null 2>&1 &&
        "$_claim_time_candidate" -f "%M" -o /dev/null true >/dev/null 2>&1; then
        _CLAIM_TIME_BIN="$_claim_time_candidate"
        break
    fi
done
unset _claim_time_candidate

_run_measured() {
    local peak_file="$LIVE_DIR/.peak"
    if [ -n "$_CLAIM_TIME_BIN" ]; then
        "$_CLAIM_TIME_BIN" -f "%M" -o "$peak_file" "$@"
        PIXGUARD_CLAIM_PEAK_KB=$(cat "$peak_file" 2>/dev/null || true)
        # show_claim.py prints this as a number; anything else is dropped rather
        # than passed on to fail there.
        case "$PIXGUARD_CLAIM_PEAK_KB" in
            "" | *[!0-9]*) PIXGUARD_CLAIM_PEAK_KB="" ;;
        esac
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
