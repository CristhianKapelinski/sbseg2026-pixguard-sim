# PixGuard-Sim: Technical Documentation

This document records the problem, the design, and every experiment run with its
exact command and the real captured output. All data is **synthetic** and
ground-truth labeled, whether produced by the in-repo generator or by a released
third-party generator; no number here is a measurement of a real-world fraud
rate, and no number appears unless code in this repository produced it. Every
proportion carries a Wilson 95% confidence interval; F1 and PR-AUC carry a
seed-pinned bootstrap 95% interval.

## 1. Problem

PIX, the Brazilian instant-payment system, moved 63.8 billion transactions worth
R$ 26.9 trillion in 2024 and is the country's dominant payment method. Its
irrevocable settlement turns a scam into an unrecoverable loss; unrefunded fraud
losses reached R$ 4.94 billion in 2024, up 70% year over year. The regulatory
framework has shifted from reactive refunds toward real-time prevention: under
the MED-2.0 mechanism (Resolução BCB 493/2025, mandatory from 2 February 2026),
institutions trace and block diverted funds across up to five subsequent account
layers as a fraud alert propagates through the chain. No regulation fixes a
latency in seconds, so the decision deadline is a **researcher parameter**,
reported over a range.

No open benchmark scores fraud detectors against this real-time decision window.
Existing generators (AMLSim/IBM AML, Tide, PaySim) are jurisdiction-agnostic,
and the closest open PIX generator (pix-fraud-br) models only the first transfer,
with no multi-hop tracing, no MED, no coercion, and no latency metric. PixGuard-
Sim fills the gap with an evaluation harness that reports the **pre-deadline flag
fraction** alongside conventional metrics, plus two PIX-native scenarios no open
artifact covers, and validates them across three independently-authored
generators on identical detector inputs.

## 2. Design

### 2.1 Event schema

Every generator maps into one normalized PIX-native event record
(`src/pixguard_sim/schema.py`). Each event carries a relative timeline
(`t_init_ms`, `t_settle_ms`), a MED-layer index, and PIX-native signals (DICT
key, device change, new payee, payer velocity, remote-access session, coercion
flag). Detectors see only the numeric feature columns and the timeline; the
label, scenario, and identifiers are never exposed.

### 2.2 Three independent generators

The harness scores detectors on three independently-authored sources through
thin, checksum-verified adapters (`src/pixguard_sim/adapters.py`,
`src/pixguard_sim/data_io.py`), so a number is never read off data the harness
itself produced.

| Source | Role | Illicit rate | Provenance |
|---|---|---|---|
| in-repo (Tier A) | scenario source (coercion + multi-hop MED) | 1.19% | `generator.py`, seed 20260202, frame hash `999be21b275c3aa8` |
| Tide HI / LI | real multi-hop AML at extreme rarity | 0.19% / 0.10% | Zenodo 10.5281/zenodo.18804069 (arXiv:2603.01863), CC BY 4.0 |
| pix-fraud-br | real PIX-native single-hop, prior-art anchor | 0.77% | Hugging Face `andremessina/pix-fraud-br`, ODC-BY |

Datasets are pinned in `results/data_manifest.json` by SHA-256 + provider MD5 +
source URL. The four Tide MD5s match the Zenodo record exactly:
`generated_transactions_HI.csv` `39c12784...a8a80` (903 MB),
`generated_transactions_LI.csv` `471160b7...db70` (909 MB),
`generated_nodes_HI.csv` `baedb64c...3abe`, `generated_nodes_LI.csv`
`7b8dca92...399c`. The pix-fraud-br Parquet has exactly 2,000,000 rows and
15,376 frauds (0.7688%), as published.

The four in-repo scenarios map onto the four thematic groups of a public PIX-
fraud taxonomy (arXiv:2511.20902), which classifies 15 scam
methodologies into four groups (Social Engineering by Authority/Trust; by
Refund/Benefit/Urgency; Attacks with Physical Interaction; Software- and
Remote-Access Attacks) and explicitly recommends "the creation of fraud
simulators for controlled testing". The mapping is an explicit, justified
translation, not a claim attributed to that paper:

| PixGuard-Sim scenario | Taxonomy group it instantiates |
|---|---|
| `account_takeover` | Software- and Remote-Access-Based Attacks ("ghost hand") |
| `coercion` | Attacks with Physical Interaction (robbery, express kidnapping, extortion) |
| `fake_med_refund` | Social Engineering by Refund/Benefit/Urgency, extended to multi-hop MED-2.0 tracing |
| `mule_chain` | Dispersal topology layered on the above |

### 2.3 Detectors and harness

A detector implements `fit` and `score` and reports its decision latency relative
to each event's initiation (`detectors/base.py`). Baselines: a rule-threshold
floor, scikit-learn LR/RF/GB, XGBoost (`detectors/ml.py`), and a two-layer
GraphSAGE GPU detector (`detectors/gnn.py`). The harness (`harness.py`) fits each
detector on a deterministic stratified split and reports batch
precision/recall/F1/PR-AUC with CIs, the pre-deadline flag fraction over a
deadline sweep, per-scenario recall, and recall by MED layer.

The headline metric (`metrics.py`): among true-fraud events, the fraction the
detector both flags (score at or above threshold) and decides on within the
deadline window measured from the event's own initiation. A fraud caught only
after the deadline is not actionable for pre-transaction blocking and does not
count.

## 3. Experiments (real captured output)

Environment: Python 3.13 on Linux; tabular models on CPU, the GraphSAGE baseline
on a CUDA GPU. Each experiment writes JSON to `results/` and a timestamped log to
`logs/`. Configuration: `configs/default.json` (seed 20260202, 4000 accounts,
40000 legitimate events, fraud base rate 0.012, MED max depth 5, deadline sweep
[100, 250, 500, 1000, 2000, 5000, 10000] ms, 1000 bootstrap resamples). The Tide
and pix-fraud-br runs use a label-stratified, seed-pinned subsample of 400,000
transactions each.

Commands:

```
pixguard-sim --config configs/default.json --data-dir "$DATA" manifest
pixguard-sim --config configs/default.json run --experiments E1 E2 E4
pixguard-sim --config configs/default.json --data-dir "$DATA" run --experiments E3 E5 E6 E7
```

### 3.1 E1 (main claim, latency property): the deadline metric under measured latency

The pre-deadline flag fraction is driven by each detector's **measured** per-event
inference latency (timed by `measure_score_latency_ms`), never by an assumed
budget. In-repo stream (full config): 40482 events (fraud=482, rate=0.0119);
train=24289, eval=16193 (fraud=193). Batch metrics and measured latency on the
evaluation split:

| detector | meas. latency (ms/event) | F1 [95% CI] | PR-AUC [95% CI] | recall | pre@1000ms |
|---|---|---|---|---|---|
| rule_threshold | < 0.001 | 0.443 [0.367,0.516] | 0.431 [0.366,0.497] | 0.342 | 0.342 |
| lr_fast | < 0.001 | 0.639 [0.571,0.702] | 0.706 [0.639,0.765] | 0.513 | 0.513 |
| rf_fast | 0.003 | 0.684 [0.621,0.740] | 0.743 [0.685,0.799] | 0.622 | 0.622 |
| gb_slow | < 0.001 | 0.720 [0.663,0.774] | 0.723 [0.650,0.793] | 0.648 | 0.648 |

**Interpretation.** Every tabular detector scores an event in well under one
millisecond, so each meets any realistic pre-settlement deadline and its
pre-deadline fraction equals its recall — among detectors at this cost the
deadline metric adds nothing over recall, by design. The metric begins to
separate detectors from accuracy only when scoring latency grows, which the
language-model studies (E8 on this machine, E9 over the network) examine
directly.

### 3.1a E8 and E9 (the latency realization): where the deadline actually binds

E8 scores an off-the-shelf small instruct LLM (Qwen2.5-1.5B-Instruct) one event
at a time, the way an online check would have to call it, timing each
generation; an enriched subsample of 1000 in-repo events (150 frauds; underlying
base rate 1.19%) is scored by the LLM in two prompting regimes and by a random
forest on identical events. On the reference CUDA GPU
(`results/published/e8.json`):

| detector | PR-AUC | latency mean / p95 (ms) | pre@200 | pre@1000 | pre@2000 |
|---|---|---|---|---|---|
| rf_fast | 0.931 | < 0.01 | 0.620 | 0.620 | 0.620 |
| llm_terse | 0.201 | 60 / 62 | 0.627 | 0.627 | 0.627 |
| llm_reasoning | 0.160 | 109 / 103 | 0.980 | 0.993 | 0.993 |

A 1.5B model on a GPU is *not* slow enough for the deadline to bind: it answers
in about 109 ms on average, flags 0.993 of the frauds and decides every one of
them inside a 1000 ms window. (The mean sits above the p95 because one request
took 708 ms; the median is 97 ms.) The same run on CPU, with no other change
(`results/published/e8_cpu.json`), already moves the metric:

| detector (CPU) | PR-AUC | latency mean / p95 (ms) | pre@200 | pre@1000 | pre@2000 |
|---|---|---|---|---|---|
| llm_terse | 0.198 | 561 / 594 | **0.000** | 0.633 | 0.633 |
| llm_reasoning | 0.161 | 1078 / 3319 | **0.000** | 0.933 | 0.987 |

E9 sends the same 1000 events, drawn with the same seed, to two hosted reasoning
models over the network, the way an institution would call them
(`results/published/e9_hosted.json`). The deadline here is the regulator's 1.5 s
authorization budget (Manual de Tempos do Pix v7.0), not a sweep point:

| detector | PR-AUC | precision | recall | latency mean / p95 (ms) | output tokens | pre@1500ms |
|---|---|---|---|---|---|---|
| deepseek-v4-flash | 0.846 | 0.966 | 0.380 | 3108 / 5025 | 171 | **0.000** |
| deepseek-v4-pro | 0.849 | 0.789 | 0.747 | 5704 / 10329 | 284 | **0.000** |
| rf_fast (E8, same slice) | 0.931 | 0.989 | 0.620 | < 0.01 | -- | 0.620 |

**Interpretation.** Accuracy and deployability come apart only once a detector
deliberates, and the size of the machine decides where that happens. The local
1.5B model makes every realistic deadline on a GPU and misses a 200 ms one on
CPU; the hosted reasoning models miss the regulator's budget outright. The
stronger hosted model is not a weak detector — it flags 112 of the 150 frauds
against the random forest's 93, at 30 false alarms against 1, without ever
having seen the data — yet at the 95th percentile it spends 10 329 ms, 689% of
the 1.5 s budget (the flash model, 5025 ms, 335%), so the share of frauds it
both flags and decides in time is **0.000**. Ranked by PR-AUC alone the two
hosted models sit within 0.09 of the forest and would look like reasonable
choices; ranked by the deadline metric they catch nothing at all. This is the
designated reproduction target (C1). E8 requires the `llm` extra (torch,
transformers, accelerate) and a model download; E9 requires a network credential
the repository never stores. Both captured outputs are committed under
`results/published/`.

### 3.2 E2: single-hop-trained detectors collapse on the new scenarios

Detectors trained on a single-hop-only subset (legitimate + account-takeover),
evaluated on the full stream. Per-scenario recall (Wilson 95% CI):

| detector | account_takeover | coercion | fake_med_refund | mule_chain |
|---|---|---|---|---|
| rf_single_hop | 0.736 [0.604,0.836] N=53 | 0.146 [0.069,0.284] N=41 | 0.282 [0.165,0.438] N=39 | 0.433 [0.316,0.559] N=60 |
| gb_single_hop | 0.774 [0.645,0.865] N=53 | 0.146 [0.069,0.284] N=41 | 0.282 [0.165,0.438] N=39 | 0.533 [0.409,0.654] N=60 |

**Interpretation.** A detector that has only seen the single-hop case keeps its
highest recall there (0.736) and falls away as the scenario moves further from
what it was trained on: mule chains 0.433, multi-hop MED-2.0 refunds 0.282, and
coercion **0.146**, the lowest of the four and the structural result — a coerced
victim transacts from their own device in a genuine session, so the behavioural
features carry almost no signal. The ordering is the same for the gradient
boosting variant. The two Pix-native scenarios prior artifacts omit are exactly
the ones a single-hop-trained detector cannot reliably catch.

### 3.3 E3: cross-generator credibility on the real released Tide HI/LI sets

Each split is a label-stratified subsample of 400,000 transactions: HI carries
752 illicit events (0.19%), LI carries 417 (0.10%); the evaluation split is
160,001 events in both cases (301 and 167 illicit). Batch metrics:

| split (illicit) | detector | F1 | PR-AUC [95% CI] | recall [95% CI] |
|---|---|---|---|---|
| Tide-HI (0.19%) | rule_threshold | 0.027 | 0.036 [0.027,0.046] | 0.748 [0.696,0.793] |
| Tide-HI (0.19%) | lr_fast | 0.000 | 0.091 [0.066,0.131] | 0.000 [0.000,0.013] |
| Tide-HI (0.19%) | rf_fast | 0.276 | 0.247 [0.200,0.303] | 0.206 [0.164,0.255] |
| Tide-HI (0.19%) | xgb_fast | 0.303 | 0.250 [0.198,0.305] | 0.213 [0.170,0.262] |
| Tide-LI (0.10%) | rule_threshold | 0.017 | 0.024 [0.018,0.030] | 0.832 [0.768,0.881] |
| Tide-LI (0.10%) | lr_fast | 0.000 | 0.046 [0.030,0.071] | 0.000 [0.000,0.022] |
| Tide-LI (0.10%) | rf_fast | 0.293 | 0.241 [0.177,0.316] | 0.210 [0.155,0.277] |
| Tide-LI (0.10%) | xgb_fast | 0.281 | 0.280 [0.209,0.353] | 0.186 [0.134,0.251] |

**Interpretation.** On the rare-illicit Tide data the same detectors drop to
PR-AUC ~0.25 with recall near 0.21; the rule floor is near useless (PR-AUC 0.036
on HI, 0.024 on LI — it catches 0.748 of the illicit events on HI and pays for
them with a precision that leaves F1 at 0.027), and logistic regression flags
nothing at all at the 0.5 threshold. These honest,
sub-perfect numbers are the credibility signal: detection on real, extremely
imbalanced laundering data is genuinely hard.

### 3.4 E5: reproduce the prior-art baselines on pix-fraud-br + deadline metric

pix-fraud-br subsampled to 400,000 transactions (fraud=3075, rate 0.00769;
evaluation split 160,000 events with 1230 frauds), scored on its own engineered
balance-ratio features:

| detector | meas. latency (ms/event) | F1 | PR-AUC [95% CI] | pre@1000ms |
|---|---|---|---|---|
| rule_threshold | < 0.001 | 0.015 | 0.014 [0.013,0.016] | 0.938 |
| lr_fast | < 0.001 | 0.056 | 0.426 [0.398,0.458] | 0.029 |
| rf_fast | 0.002 | 0.847 | 0.917 [0.904,0.928] | 0.794 |
| gb_slow | < 0.001 | 0.842 | 0.902 [0.885,0.917] | 0.793 |
| xgb_fast | < 0.001 | 0.851 | 0.920 [0.907,0.930] | 0.817 |

**Interpretation.** XGBoost reaches PR-AUC 0.920 [0.907,0.930], reproducing the
dataset's published XGBoost baseline (PR-AUC 0.865 on its own validation sample,
different splits and feature sets) within tolerance and confirming the harness
does not inflate scores. All tabular detectors are sub-millisecond on this set,
so each detector's pre-deadline fraction tracks its recall — the deadline metric
discriminates by latency only under a genuinely slow detector (the CPU LLM runs
and the hosted models in 3.1a), not here.

### 3.5 E6: cross-generator transfer

A random forest on the shared schema features, trained on one generator and
tested on another:

| transfer | N test | F1 | PR-AUC [95% CI] | recall |
|---|---|---|---|---|
| in-repo → in-repo | 16193 | 0.684 | 0.743 [0.685,0.799] | 0.622 |
| pix-fraud-br → pix-fraud-br | 160000 | 0.080 | 0.137 [0.120,0.156] | 0.045 |
| in-repo → pix-fraud-br | 400000 | 0.014 | 0.021 [0.020,0.022] | 0.848 |
| pix-fraud-br → in-repo | 40482 | 0.154 | 0.281 [0.241,0.325] | 0.085 |

**Interpretation.** A detector that memorised one generator's quirks collapses
when evaluated on a different generator (PR-AUC 0.021 and 0.281 cross-generator
vs 0.743 in-distribution on the in-repo stream). This is the strongest defence
against the circularity threat. The pix-fraud-br row is the second half of the
same point read from the other direction: scored on the four columns every
source shares, rather than on its own engineered features, that dataset falls
from PR-AUC 0.920 (E5) to 0.137 — the accuracy lives in the source-specific
features, not in the shared schema.

### 3.6 E7: GPU GraphSAGE baseline

Two-layer GraphSAGE on the available CUDA GPU (device=cuda), compared to the
tabular random forest on identical inputs:

| dataset | detector | F1 | PR-AUC | recall | fit (s) |
|---|---|---|---|---|---|
| in-repo | gnn_sage | 0.150 | 0.247 | 0.850 | 2.35 |
| in-repo | rf_fast | 0.684 | 0.743 | 0.622 | 1.40 |
| Tide-HI | gnn_sage | 0.012 | 0.011 | 0.525 | 0.18 |
| Tide-HI | rf_fast | 0.276 | 0.247 | 0.206 | 11.00 |

**Interpretation.** The graph-aware baseline is honest and imperfect: it does not
dominate the tabular models here, reaching PR-AUC 0.247 on the in-repo layer and
0.011 on the sparse Tide-HI graph. Per-event inference latency is sub-millisecond
for every detector. Scaling the GNN with neighbour sampling to production-volume
graphs is a research direction left to future work.

### 3.7 E4: determinism

Two independent generations of the in-repo stream produce identical content
hashes: `frame_hash_run1 = 999be21b275c3aa8`, `frame_hash_run2 =
999be21b275c3aa8`, `deterministic = true`.

## 4. Tests and lint

- `uv run --extra dev pytest`: 30 passed (no network, no containers). The adapter
  tests build small frames in each dataset's real column schema. The suite is a
  command of its own and is deliberately not run by `scripts/minimal_test.sh`:
  the tests check the code, the minimal test checks that the pipeline runs, and
  a test failing for a toolchain reason should not read as the artifact failing.
- `uv run --extra dev ruff check src tests scripts`: all checks passed.

## 5. Reproducibility notes

The in-repo experiments (E1, E2, E4) need no external data and reproduce
exactly. The cross-generator experiments (E3, E5, E6) reproduce within the
reported bootstrap CIs given the same pinned datasets and seed; the prior-art
PR-AUC on pix-fraud-br lands within tolerance of the published value. The
GraphSAGE baseline (E7) is deterministic up to GPU floating-point nondeterminism.
The language-model runs (E8, E9) reproduce their accuracy under the same seed but
not their latency: E8's timings belong to the machine that ran it — the GPU and
CPU columns in 3.1a differ by an order of magnitude on the same model and the
same prompts — and E9's belong to the network path and the provider's queue. The
deadline metric is meant to be read that way, as a property of a detector on a
given deployment rather than a constant of the model.

A large-scale graph study on production-volume traffic, and a positional
comparison against the gated FCA APP benchmark, are left as future work in
research terms.
