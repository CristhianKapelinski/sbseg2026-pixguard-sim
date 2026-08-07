# PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors

PixGuard-Sim is an open, detector- and generator-agnostic **evaluation harness** for fraud detectors targeting Pix, Brazil's instant-payment system. Because Pix settlement is irrevocable and the MED-2.0 regulation (mandatory since 2 February 2026) requires institutions to *block fraud before money settles*, accuracy alone is not enough: a detector must also decide **in time**. PixGuard-Sim scores any detector by the **pre-deadline flag fraction**, the share of frauds flagged within a configurable decision deadline, using each detector's **measured** per-event inference latency, alongside precision, recall, F1, and PR-AUC with 95% confidence intervals. It also ships reference definitions of two Pix-native scenarios absent from open prior work (multi-hop MED-2.0 refund tracing and coercion) and runs the same harness across three independently-authored generators through thin, checksum-pinned adapters. The headline finding: among sub-millisecond tabular detectors the deadline metric reduces to recall, but once a detector deliberates the metric separates accuracy from deployability. Two hosted reasoning models score PR-AUC **0.846** and **0.849** on 1000 events without ever seeing the data, and the stronger of them finds **112** of 150 frauds where a random forest trained on 16 193 labelled events finds **93**, yet at the 95th percentile they spend **5025 ms** and **10 329 ms**, so the share of frauds they both flag and decide inside the regulator's 1.5 s authorization budget is **0.000**. Ranking by accuracy alone selects the two detectors that cannot be deployed when the decision is due. All inputs are synthetic and ground-truth labeled; no number is a real-world fraud rate.

> **Paper:** *PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud Detectors* (SBSeg 2026).

> **For SBSeg 2026 artifact reviewers (SeloD/F/S/R).** [`DOCUMENTATION.md`](DOCUMENTATION.md) holds the full captured experiment output and is not needed for the review.

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
| [Cleaning up](#cleaning-up) | One command removes what a run created |
| [License](#license) | Licensing information |
| [Optional commands](OPTIONAL_COMMANDS.md) | Commands outside the reviewer path: E4, E7, E8, E9, per-figure scripts |
| [How to cite](#how-to-cite) | Paper reference and machine-readable `CITATION.cff` |

---

## Considered Seals

- **Available (SeloD).** Public, open source (MIT), self-contained in this repository; the runnable claims need no external data.
- **Functional (SeloF).** A single command (the [Minimal Test](#minimal-test)) runs the full pipeline (generate, fit detectors, latency-aware scoring, metrics with CIs) and writes a real `results/e1.json`.
- **Sustainable (SeloS).** A `src/` layout with one module per concern (`generator.py`, `harness.py`, `metrics.py`, `detectors/`, `scenarios/`, `adapters.py`, `data_io.py`), frozen-dataclass configs, typed and docstringed code, 30 unit tests (`uv run --extra dev pytest`), and a lint-clean `ruff` configuration. Every dependency is pinned in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).
- **Reproducible (SeloR).** Fixed seeds and content-hashed inputs/outputs give byte-stable generation (experiment E4 checks frame-hash equality). The in-repo claims (E1, E2) reproduce deterministically; the cross-generator runs reproduce within bootstrap CIs from datasets pinned by checksum in `results/data_manifest.json`.

---

## Basic Information

| | |
|---|---|
| **OS** | Linux (developed on Ubuntu 24.04). macOS/Windows expected to work for the in-repo experiments. |
| **Python** | 3.11 or newer (validated on 3.13 via `uv`). The floor is set by matplotlib 3.11, the version that produced the figures in the paper; below it the figure scripts still run but lay the data out slightly differently. |
| **RAM** | < 1 GB for the in-repo experiments (E1/E2/E4); ~6 GB peak for the cross-generator runs (E3/E5/E6) on a 400k-row subsample. |
| **Disk** | `.venv` after `uv sync`: ~1.1 GB. Optional third-party datasets (cross-generator claim only): ~1.8 GB (Tide) + ~130 MB (pix-fraud-br), fetched outside the repo. |
| **GPU** | **Not needed.** The whole tabular pipeline runs on CPU. A CUDA GPU only accelerates the optional GraphSAGE (E7) and LLM-latency (E8) baselines, both of which fall back to CPU. |
| **Reference machine** | 6-core/12-thread x86-64 CPU · 32 GB RAM · single CUDA GPU · Ubuntu 24.04 · Python 3.13. All times in this README were measured here. |

---

## Dependencies

All packages are pinned in [`pyproject.toml`](pyproject.toml) with a committed [`uv.lock`](uv.lock); the reviewer installs everything with **`uv sync`** and runs every command with **`uv run`**. No manual `pip` step is needed.

- **Core (installed by `uv sync`):** `numpy`, `pandas`, `scikit-learn`, `networkx`, sufficient for the [Minimal Test](#minimal-test) and the in-repo experiments (E1, E2, E4).
- **Dev (`--extra dev`):** `pytest`, `ruff`, for the unit tests and the linter.
- **Cross-generator (`--extra datasets`):** `datasets`, `pyarrow`, `xgboost`, only for the cross-generator claim (E3/E5/E6) that scores real third-party data.
- **Optional baselines:** `--extra gnn` (`torch`) for the GraphSAGE baseline (E7); `--extra llm` (`torch`, `transformers`, `accelerate`) for the on-machine language-model study (E8). The hosted-model study (E9) needs neither, only network.
- **Figures (`--extra figures`):** `matplotlib`, pinned to **3.11.x**, the minor that produced the committed figures. A looser range resolves to 3.10, which regenerates figures that do not match the paper.

**Third-party inputs are auto-fetched.** The two public generators are downloaded on demand by [`scripts/claim3.sh --run`](scripts/claim3.sh): Tide HI/LI from Zenodo (`10.5281/zenodo.18804069`, CC BY 4.0) and pix-fraud-br from Hugging Face (`andremessina/pix-fraud-br`, ODC-BY), both over HTTPS. Each file is verified against the provider-published checksum and pinned in `results/data_manifest.json`; a missing or mismatched file raises a typed error rather than fabricating a result. No dataset bytes are vendored in the repository.

---

## Security Concerns

- The tool runs **entirely locally** over synthetic data; it runs no untrusted code and ships no container required for the review.
- **Network** is used by the optional cross-generator claim, which fetches the two named public datasets over HTTPS (Zenodo, Hugging Face), and by the hosted-model experiment (E9, `scripts/run_hosted_llm.py`), which calls a reasoning model one transfer per request. E9 sends its requests wherever `PIXGUARD_LLM_ENDPOINT` points, defaulting to a local forwarding endpoint (`http://127.0.0.1:8080/v1/messages`) that holds the provider credential. An evaluator who prefers to call a provider directly sets that variable and `PIXGUARD_LLM_API_KEY`; the key is read from the environment, sent as a header, and never written to a results file or stored in this repository. The minimal test and the in-repo experiments need no network.
- **No credentials** are stored in the repository. The external data root is never hardcoded: it is read from the `--data-dir` flag or the `PIXGUARD_DATA_DIR` environment variable, and downloads land in a scratch directory outside the repo (gitignored).

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

`uv sync` is the only install step. On the reference machine it completes in **under a second** with a warm cache (the first-ever run downloads the wheels, a few tens of seconds depending on network). Every command below is run with `uv run`.

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

This confirms the harness runs end to end and writes a real, inspectable `results/e1.json`. The deadline metric only *separates* detectors once one is genuinely slow, which is the LLM-latency study in [Experiments](#experiments) (Claim #1). (Run `uv run --extra dev pytest` for the 30 unit tests; ~9 s.)

---

## Experiments

Each claim is **one command** that recomputes the experiment here with the paper's own
configuration and prints the measured value beside the published one, with an `OK`/`FAIL`
per line and a non-zero exit on any mismatch. The paper's side of every comparison is read
from [`expected/paper_macros.tex`](expected/paper_macros.tex), the frozen camera-ready macro
block, so a claim cannot silently drift from the paper.

Nothing is written into [`results/published/`](results/published): a live run goes to
`results/claim_run/`, and the framed block states which of the two it read. Claim #3 is the
one exception, and says so in its own output.

| Claim | Paper experiment(s) | Data | Recomputed here? | Main? |
|---|---|---|---|---|
| #1 The deadline metric is driven by measured latency, not by accuracy | E1 (+ E8/E9 for the slow-detector realization) | in-repo | yes, ~21 s | **yes** |
| #2 Single-hop training collapses on the two Pix-native scenarios | E2 | in-repo | yes, ~14 s | |
| #3 Cross-generator credibility on three independently authored generators | E3, E5, E6 | Tide + pix-fraud-br | only with `--run` (~2 GB, ~13 min) | |

### Claim #1 (MAIN): the deadline metric is driven by measured latency, so accuracy alone selects detectors that cannot answer in time

**Paper reference:** Section *The deadline metric*, Table 1 and Table 2.

**What this claim asserts, and where it is weakest.** The pre-deadline fraction is computed
from each detector's *measured* per-event latency. The four tabular detectors decide in
microseconds, so none of them ever misses the deadline and the metric collapses onto recall:
run on its own, this experiment shows the metric adding **nothing**, which is the honest
result and the reason the claim needs its second half. The metric only bites once a detector
deliberates: a local instruct model answers in 109 ms at PR-AUC 0.160, while two hosted
reasoning models on the same 1000 events reach PR-AUC 0.846 and 0.849 and spend 5025 ms and
10 329 ms at the 95th percentile, so their pre-deadline fraction at the regulator's 1.5 s
budget is 0.000. That second half needs a model download or a network credential, so the
command below verifies the first half and the published outputs carry the second.

```bash
./scripts/claim1.sh
```

- **Expected time:** 21 s on the reference machine, 12 s on a 32-core server.
- **Expected resources:** ~0.8 GB peak RAM, negligible disk. No GPU, no network.
- **Expected result:**

```text
══════════════════════════════════════════════════════════════════
  Claim #1  The deadline metric is driven by measured latency  (MAIN CLAIM)
──────────────────────────────────────────────────────────────────
  rule_threshold F1              : 0.443        (paper 0.443)           OK
  rule_threshold PR-AUC          : 0.431        (paper 0.431)           OK
  lr_fast F1                     : 0.639        (paper 0.639)           OK
  lr_fast PR-AUC                 : 0.706        (paper 0.706)           OK
  rf_fast F1                     : 0.684        (paper 0.684)           OK
  rf_fast PR-AUC                 : 0.743        (paper 0.743)           OK
  rf_fast recall                 : 0.622        (paper 0.622)           OK
  pre@1000ms equals recall       : yes          (paper yes)             OK
  slowest per-event latency (ms) : 0.0030      
──────────────────────────────────────────────────────────────────
  source of these numbers        : recomputed on this machine just now
  wall clock on this machine     : XXX s
  peak memory on this machine    : XXX MB
──────────────────────────────────────────────────────────────────
  RESULT: OK   (8/8 gated values match the paper)
══════════════════════════════════════════════════════════════════
```

`XXX` marks the two lines that are yours rather than the paper's: wall clock and peak memory
depend on the machine, are reported for your information, and are never gated. Measured on two
hosts: 21 s / 791 MB on a Ryzen 5 8600G and 12 s / 214 MB on a 32-core server.

### Claim #2: a detector trained only on the single-hop case collapses on coercion and multi-hop MED-2.0 refunds

**Paper reference:** Section *Scenarios open data omits*, Table 3.

**What this claim asserts.** A random forest trained only on legitimate traffic plus account
takeover, the modeling scope of the open Pix generator, keeps its recall there and falls
away as the scenario moves further from what it saw: mule chains, then multi-hop MED-2.0
refunds, then coercion, where the victim transacts from their own device and the transfer
looks ordinary by construction.

```bash
./scripts/claim2.sh
```

- **Expected time:** 14 s on the reference machine, 8 s on a 32-core server.
- **Expected resources:** ~0.8 GB peak RAM, negligible disk. No GPU, no network.
- **Expected result:**

```text
══════════════════════════════════════════════════════════════════
  Claim #2  Single-hop training collapses on the Pix-native scenarios
──────────────────────────────────────────────────────────────────
  recall, account_takeover       : 0.736        (paper 0.736)           OK
    N, account_takeover          : 53           (paper 53)              OK
  recall, mule_chain             : 0.433        (paper 0.433)           OK
    N, mule_chain                : 60           (paper 60)              OK
  recall, fake_med_refund        : 0.282        (paper 0.282)           OK
    N, fake_med_refund           : 39           (paper 39)              OK
  recall, coercion               : 0.146        (paper 0.146)           OK
    N, coercion                  : 41           (paper 41)              OK
──────────────────────────────────────────────────────────────────
  source of these numbers        : recomputed on this machine just now
  wall clock on this machine     : XXX s
  peak memory on this machine    : XXX MB
──────────────────────────────────────────────────────────────────
  RESULT: OK   (8/8 gated values match the paper)
══════════════════════════════════════════════════════════════════
```

### Claim #3: the harness holds up on two independently authored generators

**Paper reference:** Section *Cross-generator credibility*, Tables 4 and 5.

**What this claim asserts.** The same harness runs, through adapters, on two generators we
did not write. It reproduces the pix-fraud-br prior-art XGBoost baseline, measures the decline
on rare-illicit Tide, and quantifies how far accuracy falls when a detector is trained on one
generator and tested on another.

```bash
./scripts/claim3.sh                 # reads the committed campaign, instant
./scripts/claim3.sh --run           # refetches ~2 GB and recomputes here
```

- **Flags:** `--run` downloads Tide HI/LI from Zenodo and pix-fraud-br from Hugging Face into
  `$PIXGUARD_DATA_DIR` (default `./data`, gitignored), verifies them against the manifest and
  recomputes E3/E5/E6. Without it the script reads
  [`results/published/`](results/published) and **says so in its output** rather than implying
  it measured anything.
- **Expected time:** instant to read. With `--run`, **9 to 12 min on a 32-core host and 37 min on a
  6-core/12-thread Ryzen 5 8600G**, once the data is local; the first run also downloads
  ~1.8 GB of Tide and ~130 MB of pix-fraud-br.
- **Expected resources:** ~6 GB peak RAM and ~2 GB disk with `--run`, both outside the repo.
- **Expected result:**

```text
══════════════════════════════════════════════════════════════════
  Claim #3  Cross-generator credibility on three generators
──────────────────────────────────────────────────────────────────
  pix-fraud-br XGBoost PR-AUC    : 0.920        (paper 0.920 +/-0.01)   OK
    its published baseline       : 0.865        (paper 0.865)           OK
    on shared columns only       : 0.137        (paper 0.137)           OK
  Tide HI XGBoost PR-AUC         : 0.250        (paper 0.250 +/-0.01)   OK
  Tide LI XGBoost PR-AUC         : 0.280        (paper 0.280 +/-0.01)   OK
  transfer ours -> pix-fraud-br  : 0.021        (paper 0.021)           OK
  transfer pix-fraud-br -> ours  : 0.281        (paper 0.281)           OK
  in-distribution, ours          : 0.743        (paper 0.743)           OK
──────────────────────────────────────────────────────────────────
  source of these numbers        : read from the committed campaign
                                   (results/published); pass --run to regenerate it here
  wall clock on this machine     : 0 s
──────────────────────────────────────────────────────────────────
  RESULT: OK   (8/8 gated values match the paper)
══════════════════════════════════════════════════════════════════
```

**On `--run` the three XGBoost rows are compared with a declared tolerance of ±0.01, and the
block shows it.** Two independent hosts, both installing xgboost 3.2.0 from the committed
lock, produce 0.913, 0.258 and 0.276: they agree with each other to the third decimal and sit
at most 0.008 from the published campaign, which predates the current pin. Every other row on
this claim comes from a deterministic detector and is compared exactly, so a real regression
still fails the gate. Reading the committed campaign (the command without `--run`) matches
the paper exactly on all eight.

The 0.920 and the dataset's published 0.865 are **not** comparable: they use different splits
and different features. What the drop to 0.137 shows is the same dataset scored on the four
columns every source shares.

### Every number in the paper, and the figures

The paper hardcodes no empirical number: each one is a `\newcommand` generated from
[`results/published/`](results/published). One command regenerates the three figures and
checks all 158 of them against the frozen camera-ready block.

```bash
./scripts/make_figures.sh
```

- **Expected time:** ~25 s. **Expected resources:** < 1 GB RAM.
- **Expected result:** `figs/fig_results.pdf`, `figs/fig_dataset.pdf` and `figs/pipeline.pdf`
  are rewritten, followed by `PAPER VALUES: 158 PASS / 0 FAIL`. The regenerated figures render
  pixel-identical to the ones printed in the paper, so `git status` staying clean is itself the
  check. This needs matplotlib 3.11, which is why the project requires Python 3.11 or newer;
  on 3.10 the pinned range resolves to matplotlib 3.10 and the figures come out subtly different.

### Commands that are not part of the reviewer path

The language-model experiments (E8, E9), the determinism check (E4), the GraphSAGE
baseline (E7), the cheap hosted-model latency check, and the per-figure scripts are in
[`OPTIONAL_COMMANDS.md`](OPTIONAL_COMMANDS.md). None of them is needed for a claim or a
seal; each costs a model download, a GPU, a credential, or a large fetch.

---

## Cleaning up

One command removes everything a run created, the environment, the live claim outputs and the fetched generators. It never touches anything tracked by git.

```bash
./cleanup.sh
```

Pass `--dry-run` to list what would go without removing it (about ~1.2 GB).

## License

MIT. See [`LICENSE`](LICENSE).

---

## How to cite

If you use this artifact, please cite the paper:

> Kapelinski, C. and Kreutz, D. (2026). PixGuard-Sim: A Deadline-Aware Testbed for Pix Fraud
> Detectors. In *Anais do XXVI Simpósio Brasileiro de Segurança da Informação e de Sistemas
> Computacionais (SBSeg 2026)*, Workshop de Trabalhos de Iniciação Científica e de Graduação
> (WTICG). Sociedade Brasileira de Computação (SBC).

```bibtex
@inproceedings{kapelinski2026pixguardsim,
  title     = {{PixGuard-Sim}: A Deadline-Aware Testbed for Pix Fraud Detectors},
  author    = {Kapelinski, Cristhian and Kreutz, Diego},
  booktitle = {Anais do XXVI Simp\'osio Brasileiro de Seguran\c{c}a da Informa\c{c}\~ao e de
               Sistemas Computacionais (SBSeg 2026), Workshop de Trabalhos de Inicia\c{c}\~ao
               Cient\'ifica e de Gradua\c{c}\~ao (WTICG)},
  year      = {2026},
  publisher = {Sociedade Brasileira de Computa\c{c}\~ao (SBC)},
}
```

[`CITATION.cff`](CITATION.cff) carries the same reference in machine-readable form, which is
what GitHub's "Cite this repository" button and Zenodo read.
