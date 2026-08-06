#!/usr/bin/env bash
# Claim #3: the harness holds up on two independently authored generators.
#
# By default this reads the committed campaign, because regenerating it downloads
# about 2 GB of third-party data and takes roughly 13 minutes; the output says so
# rather than implying it was measured here. Pass --run to fetch the data and
# recompute E3/E5/E6 on this machine.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_claim_common.sh"

if [[ "${1:-}" == "--run" ]]; then
    DATA="${PIXGUARD_DATA_DIR:-$PWD/data}"
    mkdir -p "$DATA/tide" "$LIVE_DIR"
    echo "==> Fetching the third-party generators (~2 GB on first run)"
    for f in generated_transactions_HI.csv generated_transactions_LI.csv \
             generated_nodes_HI.csv generated_nodes_LI.csv; do
        if [[ ! -f "$DATA/tide/$f" ]]; then
            echo "    tide/$f"
            curl -fSL -o "$DATA/tide/$f" \
                "https://zenodo.org/records/18804069/files/$f?download=1"
        fi
    done
    if [[ ! -f "$DATA/pix_fraud_br.parquet" ]]; then
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
