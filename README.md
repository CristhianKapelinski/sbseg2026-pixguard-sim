# PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors

PixGuard-Sim is an open, detector- and generator-agnostic **evaluation harness** for fraud detectors targeting Pix, Brazil's instant-payment system. Because Pix settlement is irrevocable and the new MED-2.0 regulation (mandatory since 2 February 2026) requires institutions to *block fraud before money settles*, accuracy alone is not enough: a detector must also decide **in time**. PixGuard-Sim scores any detector by the **pre-deadline flag fraction** — the share of frauds flagged within a configurable pre-transaction decision deadline — alongside precision, recall, F1, and PR-AUC with 95% confidence intervals. It also ships reference definitions of two Pix-native scenarios absent from open prior work (multi-hop MED-2.0 refund tracing and coercion) and validates everything across three independently-authored generators through thin, checksum-pinned adapters. The headline finding: two detectors with statistically indistinguishable batch F1 are separated decisively by the deadline metric — at a 1000 ms deadline one flags **0.974** of frauds in time while the other flags **0.000** until its 5000 ms budget elapses — and the same inversion reappears on the real released pix-fraud-br set (**0.844** vs **0.000**). All inputs are synthetic and ground-truth labeled; no number is a real-world fraud rate.

> **Paper:** *PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors* — SBSeg 2026 Salão de Ferramentas.

> **For SBSeg 2026 artifact reviewers (SeloD/F/S/R).** This README is the single, self-contained guide for the evaluation: follow it end to end and you reach all four seals. The other Markdown file in the repository (`DOCUMENTATION.md`) is complementary technical documentation with the full captured experiment output and is **not required** for the artifact review.

---

## README Structure

| Section | Description |
|---|---|
| [Considered Seals](#considered-seals) | The four SBSeg quality seals targeted by this artifact |
| [Basic Information](#basic-information) | Hardware, OS, and software environment |
| [Dependencies](#dependencies) | Key pinned packages and how third-party inputs are fetched |
| [Security Concerns](#security-concerns) | Risks and mitigations for evaluators |
| [Installation](#installation) | Clone, install `uv`, `uv sync` |
| [Minimal Test](#minimal-test) | One command that exercises the real pipeline end to end |
| [Experiments](#experiments) | Reproduction of the paper's claims, one designated main claim |
| [License](#license) | Licensing information |

---

## Considered Seals

The seals considered are: **Available (SeloD)**, **Functional (SeloF)**, **Sustainable (SeloS)**, and **Reproducible (SeloR)**.

- **SeloD — Available.** The artifact is public and open source (MIT), self-contained in this repository, and the main claim runs with no external data.
- **SeloF — Functional.** A single command (the [Minimal Test](#minimal-test)) runs the full pipeline — generate, fit detectors, latency-aware scoring, metrics with CIs — and writes a real `results/e1.json` with the headline result.
- **SeloS — Sustainable.** A `src/` layout with one module per concern (`generator.py`, `harness.py`, `metrics.py`, `detectors/`, `scenarios/`, `adapters.py`, `data_io.py`), frozen-dataclass configs, typed and docstringed code, unit tests (`tests/`), and a lint-clean `ruff` configuration. Every dependency is pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).
- **SeloR — Reproducible.** Fixed seeds, content-hashed inputs/outputs, byte-stable generation (experiment E4 checks frame-hash equality). The main claim (E1) reproduces exactly; the cross-generator runs reproduce within bootstrap CIs from datasets pinned by checksum in `results/data_manifest.json`.

---

## Basic Information

| | |
|---|---|
| **OS** | Linux (developed and run on Ubuntu 24.04). macOS/Windows expected to work for the in-repo experiments. |
| **Python** | 3.10+ (validated on 3.13 via `uv`). |
| **RAM** | < 1 GB for the in-repo experiments (E1/E2/E4); ~6 GB peak for the cross-generator runs (E3/E5/E6) on a 400k-row subsample. |
| **Disk** | `.venv` after `uv sync`: ~1.1 GB. Optional third-party datasets (only for the cross-generator claim): ~1.8 GB (Tide) + ~130 MB (pix-fraud-br), fetched to a scratch dir outside the repo. |
| **GPU** | Not required. The whole tabular pipeline runs on CPU. A CUDA GPU is used only by the optional GraphSAGE baseline (E7) and falls back to CPU otherwise. |
| **Reference machine** | AMD Ryzen 5 8600G (6c/12t) · 30 GB RAM · NVIDIA RTX 5060 Ti 16 GB · Ubuntu 24.04 · Python 3.13 — all measured times in this README were taken here. |

---

## Dependencies

All packages are pinned in [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock) and installed by `uv sync`; no manual `pip` step is needed.

- **Core (installed by `uv sync`):** `numpy`, `pandas`, `scikit-learn`, `networkx`. Sufficient for the [Minimal Test](#minimal-test) and the in-repo experiments (E1, E2, E4).
- **Cross-generator extra (`--extra datasets`):** `datasets`, `pyarrow`, `xgboost` — only needed for the cross-generator claim (E3/E5/E6) that scores real third-party data.
- **GPU baseline extra (`--extra gnn`):** `torch` — only for the optional GraphSAGE baseline (E7).

**Third-party inputs are auto-fetched.** The two public generators are downloaded on demand by [`scripts/exp_cross_generator.sh`](scripts/exp_cross_generator.sh): Tide HI/LI from Zenodo (`10.5281/zenodo.18804069`, CC BY 4.0) over HTTPS, and pix-fraud-br from Hugging Face (`andremessina/pix-fraud-br`, ODC-BY). Each file is verified against the provider-published checksum and pinned in `results/data_manifest.json` before use; a missing or mismatched file raises a typed error rather than fabricating a result. No dataset bytes are vendored in the repository.

---

## Security Concerns

- The tool runs **entirely locally** and executes only its own code over synthetic data; it runs no untrusted code and ships no containers required for the review.
- **Network** is used only by the optional cross-generator claim, which fetches the two named public datasets over HTTPS (Zenodo, Hugging Face). The minimal test and the main claim need no network.
- **No credentials** are stored in the repository. The external data root is never hardcoded — it is read from the `--data-dir` flag or the `PIXGUARD_DATA_DIR` environment variable, and downloads land in a scratch directory outside the repo (gitignored).

---

## Installation

```bash
# 1. Clone the repository (anonymous review: download the ZIP from the artifact link in the paper)
git clone <REPOSITORY-URL> pixguard-sim
cd pixguard-sim

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install the pinned environment
uv sync
```

`uv sync` is the only install step. On the reference machine it completes in **~4 s** with a warm cache (first-ever run downloads the wheels, a few tens of seconds depending on network). Every command below is run with `uv run`.

---

## Minimal Test

One command. It runs the designated main-claim experiment (E1) on a freshly generated synthetic Pix stream and prints the headline result, exercising the full pipeline (generate → fit detectors → latency-aware scoring → metrics with CIs). No external data, no network.

```bash
./scripts/minimal_test.sh
```

**Expected time:** ~6 s on the reference machine (measured: 6.21 s).
**Expected resources:** < 1 GB RAM, negligible disk.
**Expected output:** a table over four baseline detectors. The key observation is that `rf_fast` and `gb_slow` have near-identical batch F1, yet at the 1000 ms deadline `rf_fast` flags a high fraction of frauds in time while `gb_slow` flags **0.000** until the deadline reaches its 5000 ms inference budget:

```
detector             F1  pre@1000ms  pre@5000ms
rule_threshold    0.444       0.351       0.351
lr_fast           0.851       0.754       0.754
rf_fast           0.982       0.965       0.965
gb_slow           0.982       0.000       0.965
```

This uses a reduced "fast" configuration so the pipeline finishes in seconds; the rank inversion is identical to the full run. A written `results/e1.json` is the real, inspectable output.

---

## Experiments

Each claim below is **one command**. Every claim defaults to a **fast** variant (a few seconds, reduced event count and bootstrap resamples); pass `--full` to run the paper's full configuration. The fast variant reproduces the same qualitative result. The cross-generator claim is gated behind a separate script and downloads ~2 GB of third-party data — reviewers short on time or disk may instead inspect the pre-computed `results/e3.json`, `results/e5.json`, `results/e6.json`, which are the real full-run outputs.

| Claim | Paper experiment(s) | Data | Main? |
|---|---|---|---|
| #1 The deadline metric separates equal-batch detectors | E1 | in-repo | **yes** |
| #2 Single-hop-trained detectors collapse on the new scenarios | E2 | in-repo | |
| #3 Cross-generator credibility on real released datasets | E3, E5, E6 | Tide + pix-fraud-br | |

> Two supporting experiments from the paper are not separate reviewer steps: **E4** (determinism) is a one-liner — `uv run pixguard-sim --config configs/default.json run --experiments E4` — that writes `results/e4.json` with `"deterministic": true` (frame hash `a1064a3436566c25` on both runs); **E7** (the optional GPU GraphSAGE baseline) requires the `gnn` extra and its measured numbers are in `results/e7.json` and `DOCUMENTATION.md`.

### Claim #1 (MAIN) — The pre-deadline flag fraction separates detectors that batch metrics cannot

**Description.** Two detectors with statistically indistinguishable batch F1 (a fast random forest and a slow gradient-boosting model) are ranked differently by the deadline metric: the slow model cannot decide before its inference budget elapses, so it flags 0 frauds within a 1000 ms deadline while the fast one flags nearly all. This is the spine claim (C1) and the designated reproduction target.

```bash
./scripts/exp_e1_deadline.sh            # add --full for the paper's full config
```

- **Expected time:** ~9 s fast (measured: 9.44 s); ~23 s full (measured: 22.79 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (full config, in `results/e1.json`):** `rf_fast` F1 = 0.982, pre@1000ms = 0.974; `gb_slow` F1 = 0.984, pre@1000ms = **0.000**, recovering to 0.969 at the 5000 ms deadline. On the fast config the inversion is identical (0.965 vs 0.000 → 0.965). The two are indistinguishable in batch F1 but completely separated by the deadline metric.

### Claim #2 — Single-hop-trained detectors collapse on the two new Pix-native scenarios

**Description.** A detector trained only on the single-hop case (legitimate traffic plus account-takeover — the modeling scope of the open Pix generator) keeps full recall there but collapses on the two scenarios no open artifact covers: coercion and multi-hop MED-2.0 refunds (C2).

```bash
./scripts/exp_e2_scenarios.sh           # add --full for the paper's full config
```

- **Expected time:** ~5 s fast (measured: 5.14 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (in `results/e2.json`):** recall ~1.000 on `account_takeover`, **0.000** on both `coercion` and `fake_med_refund` (zero at every MED layer), and partial recall (~0.40) on `mule_chain`.

### Claim #3 — Cross-generator credibility on three independently-authored generators

**Description.** The harness and the deadline metric transfer to two real released generators with no code change, and the numbers stay honestly sub-perfect and spread with difficulty — never a uniform 1.00 (C3). E5 reproduces the pix-fraud-br prior-art XGBoost baseline within tolerance; E3 shows detectors dropping sharply on rare-illicit Tide; E6 shows cross-generator transfer collapsing (the non-circularity check).

```bash
./scripts/exp_cross_generator.sh        # fetches ~2 GB of public data on first run
```

- **Expected time:** ~13 min once the data is local (measured: 773 s for the three experiments); the first run also downloads ~2 GB, adding several minutes depending on network. Requires the `datasets` extra (the script invokes `uv run --extra datasets`).
- **Expected resources:** ~6 GB RAM peak, ~2 GB disk for the downloads (outside the repo).
- **Expected result (in `results/e3.json`, `results/e5.json`, `results/e6.json`):** on pix-fraud-br, XGBoost reaches PR-AUC 0.938 (matching the dataset's published ~0.865 baseline within tolerance) and the deadline inversion reappears (xgb_fast pre@1000ms = 0.844 vs gb_slow = 0.000); on rare-illicit Tide the best PR-AUC drops to ~0.52; cross-generator transfer collapses to PR-AUC 0.016 (in-repo → pix-fraud-br) and 0.071 (pix-fraud-br → in-repo) against 0.49–0.99 in-distribution.

> Reviewers may skip the run and inspect the pre-computed `results/*.json` plus `DOCUMENTATION.md`, which records every command and its real captured output.

---

## License

MIT. See [`LICENSE`](LICENSE).
