#!/usr/bin/env bash
# E3 / E5 / E6: cross-generator credibility on the real, released third-party
# datasets. Fetches the public Tide HI/LI (Zenodo, CC BY 4.0) and pix-fraud-br
# (Hugging Face, ODC-BY) generators into a scratch data directory, verifies and
# pins them by checksum, then runs:
#   E3  rare-illicit transfer on Tide HI/LI
#   E5  reproduce the pix-fraud-br prior-art baseline + add the deadline metric
#   E6  cross-generator transfer (train on one generator, test on another)
#
# This is the heavy, gated path. It needs the "datasets" extra and ~2 GB of free
# disk for the downloads. The data directory defaults to ./data and can be set
# with PIXGUARD_DATA_DIR. Pre-computed results for these experiments already live
# in results/e3.json, results/e5.json, results/e6.json and may be inspected
# instead of re-running.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${PIXGUARD_DATA_DIR:-$PWD/data}"
mkdir -p "$DATA/tide"

# 1. Fetch Tide HI/LI (public, no auth).
for f in generated_transactions_HI.csv generated_transactions_LI.csv \
         generated_nodes_HI.csv generated_nodes_LI.csv; do
  if [[ ! -f "$DATA/tide/$f" ]]; then
    echo "fetching tide/$f ..."
    curl -fSL -o "$DATA/tide/$f" \
      "https://zenodo.org/records/18804069/files/$f?download=1"
  fi
done

# 2. Fetch pix-fraud-br and materialize it as a local Parquet snapshot.
if [[ ! -f "$DATA/pix_fraud_br.parquet" ]]; then
  echo "fetching pix-fraud-br ..."
  uv run --extra datasets python - "$DATA/pix_fraud_br.parquet" <<'PY'
import sys
from datasets import load_dataset
out = sys.argv[1]
load_dataset("andremessina/pix-fraud-br", split="train").to_pandas().to_parquet(out)
print("wrote", out)
PY
fi

# 3. Verify + pin the datasets, then run the cross-generator experiments.
uv run --extra datasets pixguard-sim --config configs/default.json --data-dir "$DATA" manifest
uv run --extra datasets pixguard-sim --config configs/default.json --data-dir "$DATA" run --experiments E3 E5 E6
echo
echo "=== wrote results/e3.json results/e5.json results/e6.json ==="
