# PixGuard-Sim

A deadline-aware evaluation harness for PIX fraud detectors, with multi-hop
MED-2.0 refund and coercion scenarios. This repository is the artifact backing
the paper *PixGuard-Sim: A Deadline-Aware Evaluation Harness for PIX Fraud
Detectors with Multi-Hop MED-2.0 and Coercion Scenarios*.

PixGuard-Sim is **not** a PIX dataset generator. It is a thin,
detector-agnostic and generator-agnostic evaluation harness that scores any PIX
fraud detector by the fraction of frauds it flags before a configurable
pre-transaction decision deadline (the *pre-deadline flag fraction*), in
addition to conventional precision/recall/F1/PR-AUC reported with 95%
confidence intervals. On top of the harness it ships reference definitions of
two PIX-native scenarios absent from open prior artifacts: multi-hop MED-2.0
refund tracing (up to five layers) and coercion. The harness is validated
across three independently-authored generators (an in-repo synthetic source,
the released Tide HI/LI AML datasets, and the released pix-fraud-br set) through
thin, checksum-verified adapters.

> All data used here is **SYNTHETIC** and ground-truth labeled, whether produced
> by the in-repo generator or by a released third-party generator; no number is
> a measurement of a real-world fraud rate. Real PIX transaction data is
> private. The reported results are detector-vs-harness measurements that are
> honestly imperfect and spread across generators by design.

## Repository structure

```
src/pixguard_sim/
  config.py            resolved configuration (frozen dataclasses)
  logging_setup.py     console + file logging, content hashing
  schema.py            PIX-native event schema and feature columns
  base_graph.py        lightweight scale-free base interaction graph
  generator.py         orchestration of the synthetic event stream
  scenarios/           one module per scenario (mule_chain, account_takeover,
                       fake_med_refund, coercion, legit)
  detectors/           detector plugin interface + rule + sklearn baselines
  metrics.py           batch metrics + the pre-deadline flag fraction + CIs
  harness.py           latency-aware evaluation harness (the core)
  adapters.py          real-schema adapters: Tide, pix-fraud-br, AMLSim
  data_io.py           checksum-verified dataset loading + manifest
  detectors/gnn.py     GraphSAGE GPU baseline (optional torch extra)
  experiments.py       experiment drivers E1-E7
  cli.py               command-line interface
configs/default.json   versioned default configuration
tests/                 unit tests (no network, no containers)
paper/                 LaTeX source of the paper
Dockerfile             pinned image
```

## Badges claimed

- **Available**: public, open source (MIT), self-contained, no external data
  needed to run the main claim.
- **Functional**: a single command runs the full pipeline (generate, evaluate,
  report) and writes real outputs to `results/` and `logs/`.
- **Sustainable**: `src/` layout, pinned dependencies, typed and documented
  modules, unit tests, lint-clean (`ruff`).
- **Reproducible**: fixed seeds, content-hashed inputs/outputs, deterministic
  generation (experiment E4 checks byte-stability); the main claim (E1)
  reproduces exactly, and the external datasets are pinned by checksum in a
  `data_manifest.json`.

## Basic information

- OS: Linux (developed and run on Ubuntu); macOS/Windows expected to work.
- Runtime: Python >= 3.10 (validated on 3.12). The tabular pipeline runs on CPU;
  the optional GraphSAGE baseline (E7) uses a CUDA GPU when one is present and
  falls back to CPU otherwise.
- Hardware: a modern multi-core machine. The in-repo experiments (E1, E2, E4)
  use < 1 GB RAM and finish in under a minute; the cross-generator experiments
  (E3, E5, E6) on a 400k-row subsample finish in a few minutes each.
- Disk: the external datasets (Tide HI/LI ~1.8 GB, pix-fraud-br ~134 MB) are
  fetched to a user-chosen `--data-dir` outside the repository and gitignored.

## Dependencies

Pinned in `pyproject.toml`:

- Core: `numpy>=1.26,<2.2`, `pandas>=2.1,<2.3`, `scikit-learn>=1.4,<1.6`,
  `networkx>=3.2,<3.5`.
- Cross-generator extra (`datasets`): `datasets`, `pyarrow`, `xgboost` to
  obtain and score the released third-party datasets.
- GPU baseline extra (`gnn`): `torch` for the GraphSAGE detector.
- Dev extras: `pytest`, `ruff`.

## Security concerns

The tool runs only its own code over synthetic data; it executes no untrusted
code. The dataset loaders fetch only the named public datasets over HTTPS and
verify each file against the provider-published checksum before use; a missing
or mismatched file raises a typed error rather than fabricating a result. No
credentials are stored in the repository, and the data root is read from the
`--data-dir` flag or the `PIXGUARD_DATA_DIR` environment variable, never
hardcoded.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,datasets]"      # add ",gnn" for the GPU GraphSAGE baseline
```

## Quickstart (copy-paste)

Reproduce the main claim with no external data (~15 s):

```bash
pixguard-sim --config configs/default.json reproduce && cat results/e1.json
```

Reproduce the cross-generator results on the real released datasets. Fetch the
data once to a directory of your choice, then run (a few minutes):

```bash
export DATA=/path/to/data            # any directory with ~2 GB free
# Tide HI/LI (Zenodo) and pix-fraud-br (Hugging Face) are public, no auth:
mkdir -p "$DATA/tide"
for f in generated_transactions_HI.csv generated_transactions_LI.csv \
         generated_nodes_HI.csv generated_nodes_LI.csv; do
  curl -sSL -o "$DATA/tide/$f" \
    "https://zenodo.org/records/18804069/files/$f?download=1"
done
python -c "from datasets import load_dataset; \
  load_dataset('andremessina/pix-fraud-br', split='train') \
  .to_pandas().to_parquet('$DATA/pix_fraud_br.parquet')"
pixguard-sim --data-dir "$DATA" manifest      # verifies checksums, pins them
pixguard-sim --data-dir "$DATA" run --experiments E3 E5 E6
```

## Minimal test (real pipeline, end to end)

Runs the designated main-claim experiment (E1) on freshly generated synthetic
data and writes a real JSON result. Expected time: under 15 seconds on a
laptop.

```bash
pixguard-sim --config configs/default.json reproduce
cat results/e1.json
```

Expected output: a JSON report with batch metrics for four baseline detectors
and a pre-deadline-flag-fraction curve over the deadline sweep. The headline
observation is that two detectors with near-identical batch metrics (the fast
random forest and the slow gradient-boosting model) are separated by the
pre-deadline flag fraction: at a 1000 ms deadline the fast detector flags a
high fraction of frauds in time while the slow one flags none until the
deadline reaches its 5000 ms inference budget.

## Experiments

The in-repo experiments need no external data; E3/E5/E6/E7 read the datasets
under `--data-dir`. Each writes `results/e<n>.json` and a timestamped log.

| ID | Claim | Output | Data | Main? |
|----|-------|--------|------|-------|
| E1 | The pre-deadline flag fraction is discriminative: detectors with near-identical batch P/R are separated by it. | `results/e1.json` | in-repo | yes |
| E2 | Detectors trained only on the single-hop case collapse on multi-hop MED-2.0 and coercion. | `results/e2.json` | in-repo | |
| E3 | Cross-generator credibility on the real released Tide HI/LI datasets (honest, sub-perfect, rare-illicit). | `results/e3.json` | Tide | |
| E4 | Determinism: two generations are byte-stable. | `results/e4.json` | in-repo | |
| E5 | Reproduce the prior-art baselines on the released pix-fraud-br set within tolerance, then add the deadline metric. | `results/e5.json` | pix-fraud-br | |
| E6 | Cross-generator transfer (train on one generator, test on another) collapses, the non-circularity check. | `results/e6.json` | in-repo + pix-fraud-br | |
| E7 | GraphSAGE GPU baseline across generators, with measured fit/score time. | `results/e7.json` | in-repo (+ Tide) | |

Designated reproduction target: **E1** (deterministic, no external data). The
cross-generator numbers (E3, E5, E6) reproduce within bootstrap CI; the prior-art
PR-AUC on pix-fraud-br lands within tolerance of the dataset's published value.

## Reproducing in Docker

```bash
docker build -t pixguard-sim .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
    pixguard-sim run --experiments E1 E2 E4
```

## License

MIT. See `LICENSE`.
