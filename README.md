# PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors

PixGuard-Sim is an open, detector- and generator-agnostic **evaluation harness** for fraud detectors targeting Pix, Brazil's instant-payment system. Because Pix settlement is irrevocable and the MED-2.0 regulation (mandatory since 2 February 2026) requires institutions to *block fraud before money settles*, accuracy alone is not enough: a detector must also decide **in time**. PixGuard-Sim scores any detector by the **pre-deadline flag fraction** — the share of frauds flagged within a configurable decision deadline, using each detector's **measured** per-event inference latency — alongside precision, recall, F1, and PR-AUC with 95% confidence intervals. It also ships reference definitions of two Pix-native scenarios absent from open prior work (multi-hop MED-2.0 refund tracing and coercion) and runs the same harness across three independently-authored generators through thin, checksum-pinned adapters. The headline finding: among sub-millisecond tabular detectors the deadline metric reduces to recall, but once a detector deliberates the metric separates accuracy from deployability. Two hosted reasoning models score PR-AUC **0.846** and **0.849** on 1000 events without ever seeing the data, and the stronger of them finds **112** of 150 frauds where a random forest trained on 16 193 labelled events finds **93** — yet at the 95th percentile they spend **5025 ms** and **10 329 ms**, so the share of frauds they both flag and decide inside the regulator's 1.5 s authorization budget is **0.000**. Ranking by accuracy alone selects the two detectors that cannot be deployed when the decision is due. All inputs are synthetic and ground-truth labeled; no number is a real-world fraud rate.

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
- **SeloS — Sustainable.** A `src/` layout with one module per concern (`generator.py`, `harness.py`, `metrics.py`, `detectors/`, `scenarios/`, `adapters.py`, `data_io.py`), frozen-dataclass configs, typed and docstringed code, 30 unit tests (`uv run --extra dev pytest`), and a lint-clean `ruff` configuration. Every dependency is pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).
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
| **Reference machine** | 6-core/12-thread x86-64 CPU · 32 GB RAM · single CUDA GPU · Ubuntu 24.04 · Python 3.13. All times in this README were measured here. |

---

## Dependencies

All packages are pinned in [`pyproject.toml`](pyproject.toml) with a committed [`uv.lock`](uv.lock); the reviewer installs everything with **`uv sync`** and runs every command with **`uv run`**. No manual `pip` step is needed.

- **Core (installed by `uv sync`):** `numpy`, `pandas`, `scikit-learn`, `networkx` — sufficient for the [Minimal Test](#minimal-test) and the in-repo experiments (E1, E2, E4).
- **Dev (`--extra dev`):** `pytest`, `ruff` — for the unit tests and the linter.
- **Cross-generator (`--extra datasets`):** `datasets`, `pyarrow`, `xgboost` — only for the cross-generator claim (E3/E5/E6) that scores real third-party data.
- **Optional baselines:** `--extra gnn` (`torch`) for the GraphSAGE baseline (E7); `--extra llm` (`torch`, `transformers`, `accelerate`) for the on-machine language-model study (E8). The hosted-model study (E9) needs neither, only network.
- **Figures (`--extra figures`):** `matplotlib` — for the figure scripts under `scripts/`.

**Third-party inputs are auto-fetched.** The two public generators are downloaded on demand by [`scripts/exp_cross_generator.sh`](scripts/exp_cross_generator.sh): Tide HI/LI from Zenodo (`10.5281/zenodo.18804069`, CC BY 4.0) and pix-fraud-br from Hugging Face (`andremessina/pix-fraud-br`, ODC-BY), both over HTTPS. Each file is verified against the provider-published checksum and pinned in `results/data_manifest.json`; a missing or mismatched file raises a typed error rather than fabricating a result. No dataset bytes are vendored in the repository.

---

## Security Concerns

- The tool runs **entirely locally** over synthetic data; it runs no untrusted code and ships no container required for the review.
- **Network** is used by the optional cross-generator claim, which fetches the two named public datasets over HTTPS (Zenodo, Hugging Face), and by the hosted-model experiment (E9, `scripts/run_hosted_llm.py`), which calls a reasoning model one transfer per request. E9 sends its requests to a local forwarding endpoint (`http://127.0.0.1:8080/v1/messages`) that holds the provider credential, so no key is read or stored by this repository. The minimal test and the in-repo experiments need no network.
- **No credentials** are stored in the repository. The external data root is never hardcoded — it is read from the `--data-dir` flag or the `PIXGUARD_DATA_DIR` environment variable, and downloads land in a scratch directory outside the repo (gitignored).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/CristhianKapelinski/sbseg2026-pixguard-sim.git pixguard-sim
cd pixguard-sim

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
rule_threshold    0.409       0.333       0.333
lr_fast           0.667       0.544       0.544
rf_fast           0.667       0.632       0.632
gb_slow           0.696       0.684       0.684
```

The *fast* config uses a small stream, so these values move by a few points with the BLAS build and the scikit-learn version even under the fixed seed; what should reproduce exactly is the shape, four sub-millisecond detectors whose pre-deadline fraction equals their recall. The full-config numbers the paper reports are in `results/published/` and do reproduce byte for byte.

This confirms the harness runs end to end and writes a real, inspectable `results/e1.json`. The deadline metric only *separates* detectors once one is genuinely slow — that is the LLM-latency study in [Experiments](#experiments) (Claim #1). (Run `uv run --extra dev pytest` for the 30 unit tests; ~9 s.)

---

## Experiments

Each claim below is **one command**. The in-repo claims default to a **fast** variant (a few seconds, reduced event count and bootstrap resamples); pass `--full` for the paper's full configuration. The cross-generator claim is gated behind a separate script and downloads ~2 GB of third-party data — reviewers short on time or disk may instead inspect the committed `results/published/*.json`, which are the exact outputs behind every number in the paper.

| Claim | Paper experiment(s) | Data | Main? |
|---|---|---|---|
| #1 Under real latency, the deadline metric separates a slow detector from accuracy | E1 (latency property) + E8 (LLM realization) | in-repo | **yes** |
| #2 Single-hop-trained detectors collapse on the two new Pix-native scenarios | E2 | in-repo | |
| #3 Cross-generator credibility on real released datasets | E3, E5, E6 | Tide + pix-fraud-br | |

> Two supporting experiments are not separate reviewer steps: **E4** (determinism) is a one-liner — `uv run pixguard-sim --config configs/default.json run --experiments E4` — that writes `results/e4.json` with `"deterministic": true`; **E7** (the optional GraphSAGE baseline) requires the `gnn` extra and its numbers are in `results/e7.json`.

### Claim #1 (MAIN) — The deadline metric separates a genuinely slow detector from accuracy

**Description.** The deadline metric is driven by each detector's *measured* per-event latency. For sub-millisecond tabular detectors it equals recall (E1, runnable below). It becomes discriminating only when a detector deliberates. A small instruct LLM (Qwen2.5-1.5B) scored one event at a time answers in **109 ms** on this machine and reaches PR-AUC **0.160**, at chance: fast and useless. Two hosted reasoning models on the same 1000 events reach PR-AUC **0.846** and **0.849**, above the random forest's recall on the fraud side, but spend **5025 ms** and **10 329 ms** at the 95th percentile, so their pre-deadline fraction at the regulator's 1.5 s budget is **0.000**. This is the spine claim (C1): the most accurate detector can be the one that cannot answer in time, and only the latency-aware metric exposes it.

```bash
./scripts/exp_e1_deadline.sh            # add --full for the paper's full config
```

- **Expected time:** ~4 s fast (measured: 3.6 s); ~19 s full (measured: 18.98 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (full config, in `results/e1.json`):** the four tabular detectors are all sub-millisecond, so each detector's pre@1000ms equals its recall (e.g. `rf_fast` F1 = 0.684, PR-AUC = 0.743, pre@1000ms = 0.622). The metric therefore adds nothing over recall *here* — by design. It starts to bind in E8 and E9, where the language models are slow enough for latency to matter; the captured outputs ship in `results/published/e8.json` and `results/published/e9_hosted.json`.

### Reproducing the language-model experiments (E8 and E9)

E8 measures a small instruct model on the machine that runs the harness; E9 scores the **same 1000 events, drawn with the same seed**, through hosted reasoning models. Both are optional: the first needs a model download, the second needs network.

```bash
uv run --extra llm pixguard-sim --config configs/default.json run --experiments E8
uv run python scripts/run_hosted_llm.py --models deepseek-v4-flash,deepseek-v4-pro
```

`run_hosted_llm.py` sends every request to `http://127.0.0.1:8080/v1/messages`, a local endpoint you point at your provider; it holds the credential, so nothing here reads or stores a key. It writes `results/e9_hosted.json` incrementally, one detector at a time, and records a concurrency check (the same requests timed serially and in parallel) so the bounded parallelism is shown not to distort the per-request latency the deadline metric reads.

- **Expected result (E8, `results/published/e8.json`):** `llm_terse` 60 ms mean at PR-AUC 0.201, `llm_reasoning` 109 ms at 0.160, `rf_fast` sub-millisecond at 0.931. All three clear a 1000 ms deadline.
- **Expected result (E9, `results/published/e9_hosted.json`):** PR-AUC 0.846 and 0.849, p95 latency 5025 ms and 10 329 ms, `pre_deadline_1500ms` 0.000 for both.

### Regenerating the figures and the paper macros

The figure scripts read the committed results and the generated event stream, so generate the stream first:

```bash
uv run pixguard-sim generate --output data/events.csv
uv run --extra figures python scripts/make_figure.py results/published figs/fig_results.pdf data/events.csv
uv run --extra figures python scripts/make_dataset_figures.py
uv run --extra figures python scripts/make_pipeline_figure.py results/published figs/pipeline.pdf data/events.csv
uv run python scripts/make_macros.py results/published        # LaTeX macros, printed to stdout
```

Every number in the paper is emitted by `make_macros.py` from `results/published/*.json`; the prose hardcodes none of them.

### Claim #2 — Single-hop-trained detectors collapse on the two new Pix-native scenarios

**Description.** A detector trained only on the single-hop case (legitimate traffic plus account-takeover — the modeling scope of the open Pix generator) keeps high recall there but collapses on the two scenarios no open artifact covers: coercion (deceptively normal, the victim transacts from their own device) and multi-hop MED-2.0 refunds (C2).

```bash
./scripts/exp_e2_scenarios.sh           # add --full for the paper's full config
```

- **Expected time:** ~3 s fast (measured: 2.8 s); ~13 s full (measured: 13.0 s).
- **Expected resources:** < 1 GB RAM, negligible disk.
- **Expected result (full config, `rf_single_hop`, in `results/e2.json`):** recall 0.736 on `account_takeover`, 0.433 on `mule_chain`, 0.282 on `fake_med_refund` (multi-hop MED-2.0), and 0.146 on `coercion`. Accuracy falls monotonically as the scenario moves away from the single case the detector saw in training. The fast config is noisier on the small per-scenario counts; run `--full` to reproduce the paper values.

### Claim #3 — Cross-generator credibility on three independently-authored generators

**Description.** The same harness runs on two independently released generators through adapters. E5 reproduces the pix-fraud-br prior-art XGBoost baseline within tolerance; E3 measures the decline on rare-illicit Tide; and E6 quantifies the loss under cross-generator transfer.

```bash
./scripts/exp_cross_generator.sh        # fetches ~2 GB of public data on first run
```

- **Expected time:** ~13 min once the data is local (measured: 773 s for the three experiments); the first run also downloads ~2 GB. Requires the `datasets` extra (the script invokes `uv run --extra datasets`).
- **Expected resources:** ~6 GB RAM peak, ~2 GB disk for the downloads (outside the repo).
- **Expected result (in `results/published/e3.json`, `e5.json`, `e6.json`):** on pix-fraud-br scored with its own features, XGBoost reaches PR-AUC 0.920; the dataset's own published baseline reports 0.865 on different splits and features, so the two are not directly comparable. Reduced to the four columns every source shares, the same dataset falls to 0.137. On rare-illicit Tide the best PR-AUC is 0.250 (HI) and 0.280 (LI). Cross-source transfer collapses: ours → pix-fraud-br 0.021 and pix-fraud-br → ours 0.281, against 0.743 in-distribution on our own data.

> Reviewers may skip the run and inspect the committed `results/published/*.json` plus `DOCUMENTATION.md`, which records every command and its real captured output.

---

## License

MIT. See [`LICENSE`](LICENSE).
