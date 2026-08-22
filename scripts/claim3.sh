#!/usr/bin/env bash
# Claim #3: the harness holds up on two independently authored generators.
#
# By default this reads the committed campaign, because regenerating it downloads
# about 2 GB of third-party data and takes roughly 13 minutes; the output says so
# rather than implying it was measured here. Pass --run to fetch the data and
# recompute E3/E5/E6 on this machine.
#
# Already have the two generators on disk? Point PIXGUARD_DATA_CACHE at them and
# --run downloads nothing. See the "local cache" note next to the fetch below.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_claim_common.sh"

if [[ "${1:-}" == "--run" ]]; then
    DATA="${PIXGUARD_DATA_DIR:-$PWD/data}"
    CACHE="${PIXGUARD_DATA_CACHE:-}"
    mkdir -p "$DATA/tide" "$LIVE_DIR"

    # Absolute path of an existing file, without realpath(1) or readlink -f:
    # neither is portable, and the symlink below has to survive being read from
    # a different directory.
    abspath() {
        printf '%s/%s\n' "$(cd -- "$(dirname -- "$1")" && pwd)" "$(basename -- "$1")"
    }

    # A local cache of the two generators, for a reviewer who already has them
    # (an earlier run, a shared scratch disk, a colleague's copy) and does not
    # want to pull 2 GB again. PIXGUARD_DATA_CACHE points at those files; this
    # script only ever reads them, linking them into the run's data directory, so
    # cleanup.sh removes the links and never the originals. Nothing is trusted on
    # the strength of a filename: the manifest step below checksums whatever was
    # staged, and a stale or truncated cached file fails there exactly as a bad
    # download would. Expected layout is the data directory's own
    # ("$PIXGUARD_DATA_CACHE/tide/generated_transactions_HI.csv",
    # "$PIXGUARD_DATA_CACHE/pix_fraud_br.parquet"); a flat directory holding the
    # same basenames is also accepted.
    stage_from_cache() {   # stage_from_cache <path relative to the data dir>
        local rel="$1" src
        [[ -n "$CACHE" ]] || return 1
        for src in "$CACHE/$rel" "$CACHE/$(basename "$rel")"; do
            if [[ -f "$src" ]]; then
                ln -sf -- "$(abspath "$src")" "$DATA/$rel" 2>/dev/null ||
                    cp -- "$src" "$DATA/$rel"
                echo "    $rel (from PIXGUARD_DATA_CACHE, not downloaded)"
                return 0
            fi
        done
        return 1
    }

    echo "==> Obtaining the third-party generators (~2 GB, unless already local)"
    for f in generated_transactions_HI.csv generated_transactions_LI.csv \
             generated_nodes_HI.csv generated_nodes_LI.csv; do
        if [[ ! -f "$DATA/tide/$f" ]] && ! stage_from_cache "tide/$f"; then
            echo "    tide/$f"
            curl -fSL -o "$DATA/tide/$f" \
                "https://zenodo.org/records/18804069/files/$f?download=1"
        fi
    done
    if [[ ! -f "$DATA/pix_fraud_br.parquet" ]] && ! stage_from_cache "pix_fraud_br.parquet"; then
        echo "    pix-fraud-br"
        uv run --extra datasets python scripts/fetch_pix_fraud_br.py "$DATA/pix_fraud_br.parquet"
    fi
    uv run --extra datasets pixguard-sim --config configs/default.json \
        --data-dir "$DATA" manifest
    _run_measured uv run --extra datasets pixguard-sim --config configs/default.json \
        --data-dir "$DATA" --results-dir "$LIVE_DIR" run --experiments E3 E5 E6
    echo
    report 3 "$LIVE_DIR"
else
    report 3 "$PUBLISHED_DIR"
fi
