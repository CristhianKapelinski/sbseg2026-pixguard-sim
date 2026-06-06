# PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors

PixGuard-Sim is an open, detector- and generator-agnostic **evaluation harness** for fraud detectors targeting Pix, Brazil's instant-payment system. Because Pix settlement is irrevocable and the MED-2.0 regulation (mandatory since 2 February 2026) requires institutions to *block fraud before money settles*, accuracy alone is not enough: a detector must also decide **in time**. PixGuard-Sim scores any detector by the **pre-deadline flag fraction** — the share of frauds flagged within a configurable decision deadline, using each detector's **measured** per-event inference latency — alongside precision, recall, F1, and PR-AUC with 95% confidence intervals. It also ships reference definitions of two Pix-native scenarios absent from open prior work (multi-hop MED-2.0 refund tracing and coercion) and runs the same harness across three independently-authored generators through thin, checksum-pinned adapters. The headline finding: among sub-millisecond tabular detectors the deadline metric reduces to recall, but once a detector is **genuinely slow** the metric separates it from accuracy — a small reasoning LLM (measured mean latency **1463 ms**) flags **0.000** of frauds within a 1000 ms deadline and only recovers to **1.000** at 2000 ms, while a sub-millisecond random forest meets every deadline at strictly higher accuracy (PR-AUC **0.941** vs **0.171**). All inputs are synthetic and ground-truth labeled; no number is a real-world fraud rate.

> **Paper:** *PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors* (SBSeg 2026).

> **For SBSeg 2026 artifact reviewers (SeloD/F/S/R).** This README is the single, self-contained guide for the evaluation: follow it end to end and you reach all four seals. The other Markdown file in the repository (`DOCUMENTATION.md`) is complementary technical documentation with the full captured experiment output and is **not required** for the artifact review.

---

## README Structure

| Section | Description |
|---|---|
| [Considered Seals](#considered-seals) | The four SBSeg quality seals targeted by this artifact |
| [Basic Information](#basic-information) | Hardware, OS, and software environment |
| [Dependencies](#dependencies) | Key pinned packages, managed by `uv` |
| [Security Concerns](#security-concerns) | Risks and mitigations for evaluators |
| [Installation](#installation) | Clone, install `uv`, `uv sync` |
| [Minimal Test](#minimal-test) | One command that exercises the real pipeline end to end |
| [Experiments](#experiments) | Reproduction of the paper's claims, one designated main claim |
| [License](#license) | Licensing information |

---

## Considered Seals

- **SeloD — Available.** Public, open source (MIT), self-contained in this repository; the runnable claims need no external data.
- **SeloF — Functional.** A single command (the [Minimal Test](#minimal-test)) runs the full pipeline — generate, fit detectors, latency-aware scoring, metrics with CIs — and writes a real `results/e1.json`.
- **SeloS — Sustainable.** A `src/` layout with one module per concern (`generator.py`, `harness.py`, `metrics.py`, `detectors/`, `scenarios/`, `adapters.py`, `data_io.py`), frozen-dataclass configs, typed and docstringed code, 25 unit tests (`uv run --extra dev pytest`), and a lint-clean `ruff` configuration. Every dependency is pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).
- **SeloR — Reproducible.** Fixed seeds and content-hashed inputs/outputs give byte-stable generation (experiment E4 checks frame-hash equality). The in-repo claims (E1, E2) reproduce deterministically; the cross-generator runs reproduce within bootstrap CIs from datasets pinned by checksum in `results/data_manifest.json`.

---

## Basic Information

| | |
|---|---|
| **OS** | Linux (developed on Ubuntu 24.04). macOS/Windows expected to work for the in-repo experiments. |
| **Python** | 3.10+ (validated on 3.13 via `uv`). |
| **RAM** | < 1 GB for the in-repo experiments (E1/E2/E4); ~6 GB peak for the cross-generator runs (E3/E5/E6) on a 400k-row subsample. |
| **Disk** | `.venv` after `uv sync`: ~1.1 GB. Optional third-party datasets (cross-generator claim only): ~1.8 GB (Tide) + ~130 MB (pix-fraud-br), fetched outside the repo. |
| **GPU** | **Not needed.** The whole tabular pipeline runs on CPU. A CUDA GPU only accelerates the optional GraphSAGE (E7) and LLM-latency (E8) baselines, both of which fall back to CPU. |
| **Reference machine** | 6-core/12-thread x86-64 CPU · 30 GB RAM · single CUDA GPU · Ubuntu 24.04 · Python 3.13. All times in this README were measured here. |

---

## Dependencies

All packages are pinned in [`pyproject.toml`](pyproject.toml) with a committed [`uv.lock`](uv.lock); the reviewer installs everything with **`uv sync`** and runs every command with **`uv run`**. No manual `pip` step is needed.

- **Core (installed by `uv sync`):** `numpy`, `pandas`, `scikit-learn`, `networkx` — sufficient for the [Minimal Test](#minimal-test) and the in-repo experiments (E1, E2, E4).
- **Dev (`--extra dev`):** `pytest`, `ruff` — for the unit tests and the linter.
- **Cross-generator (`--extra datasets`):** `datasets`, `pyarrow`, `xgboost` — only for the cross-generator claim (E3/E5/E6) that scores real third-party data.
- **Optional baselines:** `--extra gnn` (`torch`) for the GraphSAGE baseline (E7); `--extra llm` (`torch`, `transformers`, `accelerate`) for the LLM-latency study (E8).

**Third-party inputs are auto-fetched.** The two public generators are downloaded on demand by [`scripts/exp_cross_generator.sh`](scripts/exp_cross_generator.sh): Tide HI/LI from Zenodo (`10.5281/zenodo.18804069`, CC BY 4.0) and pix-fraud-br from Hugging Face (`andremessina/pix-fraud-br`, ODC-BY), both over HTTPS. Each file is verified against the provider-published checksum and pinned in `results/data_manifest.json`; a missing or mismatched file raises a typed error rather than fabricating a result. No dataset bytes are vendored in the repository.

---

## Security Concerns

- The tool runs **entirely locally** over synthetic data; it runs no untrusted code and ships no container required for the review.
- **Network** is used only by the optional cross-generator claim, which fetches the two named public datasets over HTTPS (Zenodo, Hugging Face). The minimal test and the main claim need no network.
- **No credentials** are stored in the repository. The external data root is never hardcoded — it is read from the `--data-dir` flag or the `PIXGUARD_DATA_DIR` environment variable, and downloads land in a scratch directory outside the repo (gitignored).

---

## Installation

```bash
# 1. Clone the repository (anonymous review: clone the anonymized mirror linked in the paper,
#    https://anonymous.4open.science/r/pixguard-sim, or download its ZIP)
git clone <REPOSITORY-URL> pixguard-sim && cd pixguard-sim

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install the pinned environment
uv sync
```

`uv sync` is the only install step. On the reference machine it completes in **under a second** with a warm cache (the first-ever run downloads the wheels — a few tens of seconds depending on network). Every command below is run with `uv run`.

---

## Minimal Test

One command. It runs the in-repo deadline experiment (E1) on a freshly generated synthetic Pix stream and prints the headline table, exercising the full pipeline (generate → fit detectors → **measured-latency** scoring → metrics with CIs). No external data, no network.

```bash
./scripts/minimal_test.sh
```

- **Expected time:** ~4 s on the reference machine (measured: 3.5 s).
- **Expected resources:** < 1 GB RAM (measured peak ~0.18 GB), negligible disk.
- **Expected output:** a table over four baseline detectors. All four score each event in well under one millisecond, so for every one the pre-deadline flag fraction equals its recall (no detector is slow enough to miss a 1000 ms deadline). Exact values on the reduced *fast* config:

```
detector             F1  pre@1000ms  pre@5000ms
rule_threshold    0.511       0.421       0.421
lr_fast           0.729       0.614       0.614
rf_fast           0.772       0.684       0.684
gb_slow           0.713       0.632       0.632
```

This confirms the harness runs end to end and writes a real, inspectable `results/e1.json`. The deadline metric only *separates* detectors once one is genuinely slow — that is the LLM-latency study in [Experiments](#experiments) (Claim #1). (Run `uv run --extra dev pytest` for the 25 unit tests; ~1.5 s.)

---

## Experiments

Each claim below is **one command**. The in-repo claims default to a **fast** variant (a few seconds, reduced event count and bootstrap resamples); pass `--full` for the paper's full configuration. The cross-generator claim is gated behind a separate script and downloads ~2 GB of third-party data — reviewers short on time or disk may instead inspect the pre-computed `results/e3.json`, `results/e5.json`, `results/e6.json`.

| Claim | Paper experiment(s) | Data | Main? |
|---|---|---|---|
| #1 Under real latency, the deadline metric separates a slow detector from accuracy | E1 (latency property) + E8 (LLM realization) | in-repo | **yes** |
| #2 Single-hop-trained detectors collapse on the two new Pix-native scenarios | E2 | in-repo | |
| #3 Cross-generator credibility on real released datasets | E3, E5, E6 | Tide + pix-fraud-br | |

> Two supporting experiments are not separate reviewer steps: **E4** (determinism) is a one-liner — `uv run pixguard-sim --config configs/default.json run --experiments E4` — that writes `results/e4.json` with `"deterministic": true`; **E7** (the optional GraphSAGE baseline) requires the `gnn` extra and its numbers are in `results/e7.json`.

### Claim #1 (MAIN) — The deadline metric separates a genuinely slow detector from accuracy

**Description.** The deadline metric is driven by each detector's *measured* per-event latency. For sub-millisecond tabular detectors it equals recall (E1, runnable below). It becomes discriminating only when a detector is slow: a small instruct LLM (Qwen2.5-1.5B) scored one event at a time reaches **1463 ms** mean latency in its reasoning regime, so it flags **0.000** of frauds within a 1000 ms deadline and only recovers to **1.000** at 2000 ms — while a random forest on the same events meets every deadline at strictly higher accuracy (PR-AUC **0.941** vs the LLM's **0.171**). This is the spine claim (C1): a heavyweight detector can be both slower than the decision window and no more accurate, and only the latency-aware metric exposes it.

```bash
./scripts/exp_e1_deadline.sh            # add --full for the paper's full config
```

- **Expected time:** ~4 s fast (measured: 3.6 s); ~19 s full (measured: 18.98 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (full config, in `results/e1.json`):** the four tabular detectors are all sub-millisecond, so each detector's pre@1000ms equals its recall (e.g. `rf_fast` F1 = 0.797, PR-AUC = 0.844, pre@1000ms = 0.728). The metric therefore adds nothing over recall *here* — by design. The LLM realization (E8, reasoning LLM pre@1000ms = **0.000** → pre@2000ms = **1.000** at PR-AUC 0.171 vs the random forest's 0.941) requires the `llm` extra and a model download; its captured output ships in `results/e8.json` and `DOCUMENTATION.md`.

### Claim #2 — Single-hop-trained detectors collapse on the two new Pix-native scenarios

**Description.** A detector trained only on the single-hop case (legitimate traffic plus account-takeover — the modeling scope of the open Pix generator) keeps high recall there but collapses on the two scenarios no open artifact covers: coercion (deceptively normal, the victim transacts from their own device) and multi-hop MED-2.0 refunds (C2).

```bash
./scripts/exp_e2_scenarios.sh           # add --full for the paper's full config
```

- **Expected time:** ~3 s fast (measured: 2.8 s); ~13 s full (measured: 13.0 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (full config, `rf_single_hop`, in `results/e2.json`):** recall 0.909 on `account_takeover`, **0.000** on `coercion` (the structural collapse), 0.526 on `fake_med_refund` (multi-hop MED-2.0), and 0.846 on `mule_chain`. The fast config is noisier on the small per-scenario counts; run `--full` to reproduce the paper values.

### Claim #3 — Cross-generator credibility on three independently-authored generators

**Description.** The harness transfers to two real released generators with no code change, and the numbers stay honestly sub-perfect and spread with difficulty — never a uniform 1.00 (C3). E5 reproduces the pix-fraud-br prior-art XGBoost baseline within tolerance; E3 shows detectors dropping sharply on rare-illicit Tide; E6 shows cross-generator transfer collapsing (the non-circularity check).

```bash
./scripts/exp_cross_generator.sh        # fetches ~2 GB of public data on first run
```

- **Expected time:** ~13 min once the data is local (measured: 773 s for the three experiments); the first run also downloads ~2 GB. Requires the `datasets` extra (the script invokes `uv run --extra datasets`).
- **Expected resources:** ~6 GB RAM peak, ~2 GB disk for the downloads (outside the repo).
- **Expected result (in `results/e3.json`, `results/e5.json`, `results/e6.json`):** on pix-fraud-br, XGBoost reaches PR-AUC 0.935 (matching the dataset's published ~0.865 baseline within tolerance, different splits/features); on rare-illicit Tide the best PR-AUC drops to ~0.52; cross-generator transfer collapses (in-repo → pix-fraud-br PR-AUC 0.016, pix-fraud-br → in-repo 0.071) against 0.84–0.99 in-distribution.

> Reviewers may skip the run and inspect the pre-computed `results/*.json` plus `DOCUMENTATION.md`, which records every command and its real captured output.

---

## License

MIT. See [`LICENSE`](LICENSE).
