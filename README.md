# PixGuard-Sim

A deadline-aware evaluation harness for PIX fraud detectors, with multi-hop
MED-2.0 refund and coercion scenarios. This repository is the artifact backing
the paper *PixGuard-Sim: A Deadline-Aware Evaluation Harness for PIX Fraud
Detectors with Multi-Hop MED-2.0 and Coercion Scenarios*.

PixGuard-Sim is **not** a PIX dataset generator. It is a thin,
detector-agnostic and generator-agnostic evaluation harness that scores any PIX
fraud detector by the fraction of frauds it flags before a configurable
pre-transaction decision deadline (the *pre-deadline flag fraction*), in
addition to conventional precision/recall/F1/PR-AUC. On top of the harness it
ships reference definitions of two PIX-native scenarios absent from open prior
artifacts: multi-hop MED-2.0 refund tracing (up to five layers) and coercion.

> All data in this repository is **SYNTHETIC**, produced by an explicit, seeded
> generative model anchored to public statistics and authentic PIX/DICT/MED
> event-schema fields. It is never a measurement of real-world fraud rates. The
> reported results are detector-vs-harness measurements on synthetic but
> ground-truth-labeled data.

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
  metrics.py           batch metrics + the pre-deadline flag fraction
  harness.py           latency-aware evaluation harness (the core)
  adapters.py          Tide adapter (Tier B) + PENDING Tier C adapters
  experiments.py       experiment drivers E1-E4
  cli.py               command-line interface
configs/default.json   versioned default configuration
tests/                 unit tests (no network, no containers)
paper/                 LaTeX source of the paper
Dockerfile             pinned CPU-only image
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
  reproduces exactly.

## Basic information

- OS: Linux (developed and run on Ubuntu); macOS/Windows expected to work.
- Runtime: Python >= 3.10 (validated on 3.12). CPU only, no GPU.
- Hardware: any modern laptop. The full E1-E4 run uses < 1 GB RAM and finishes
  in well under a minute on a single core.
- Disk: the package plus its dependency wheels fit comfortably under 0.5 GB.

## Dependencies

Pinned in `pyproject.toml`:

- `numpy>=1.26,<2.2`, `pandas>=2.1,<2.3`, `scikit-learn>=1.4,<1.6`,
  `networkx>=3.2,<3.5`.
- Dev extras: `pytest`, `ruff`.

No heavy ML stack (no torch). All dependencies are installed from PyPI.

## Security concerns

The tool runs only its own code over synthetic data; it executes no untrusted
code and accesses no network or real customer data. The PENDING Tier C adapters
read CSV files only if the user supplies them; absent the file, they raise a
typed error rather than fabricating a result. No credentials are stored in the
repository; the PENDING paths read any data from a user-provided location.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
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

Run all four with one command (writes `results/e{1,2,3,4}.json` and a
timestamped log under `logs/`):

```bash
pixguard-sim --config configs/default.json run --experiments E1 E2 E3 E4
```

| ID | Claim | Output | Main? |
|----|-------|--------|-------|
| E1 | The pre-deadline flag fraction is discriminative: detectors with near-identical batch P/R are separated by it. | `results/e1.json` | yes |
| E2 | Detectors trained only on the single-hop case collapse on multi-hop MED-2.0 and coercion. | `results/e2.json` | |
| E3 | Generator-agnosticism: the identical harness and metric run on an independent (Tide-shaped) generator. | `results/e3.json` | |
| E4 | Determinism: two generations are byte-stable. | `results/e4.json` | |

PENDING experiments (harness shipped ready, not runnable here): P1 reproduce
the open PIX generator's baselines on its released 2M-row set then score them
with the deadline metric (needs Kaggle/Hugging Face access; adapter in
`adapters.adapt_pix_fraud_br`); P2 AMLSim as a third base generator (needs a
Java build; adapter in `adapters.adapt_amlsim`); P3 a GNN baseline on the
multi-hop graph (needs a GPU for a full run); P4 a positional comparison
against the gated FCA APP benchmark. See `DOCUMENTATION.md`.

## Reproducing in Docker

```bash
docker build -t pixguard-sim .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
    pixguard-sim run --experiments E1 E2 E3 E4
```

## License

MIT. See `LICENSE`.
