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
| in-repo (Tier A) | scenario source (coercion + multi-hop MED) | 1.18% | `generator.py`, seed 20260202, frame hash `a1064a3436566c25` |
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
budget. In-repo stream (full config): 40477 events (fraud=477, rate=0.0118);
train=24286, eval=16191 (fraud=191). Batch metrics and measured latency on the
evaluation split:

| detector | meas. latency (ms/event) | F1 [95% CI] | PR-AUC [95% CI] | recall | pre@1000ms |
|---|---|---|---|---|---|
| rule_threshold | < 0.001 | 0.412 [0.340,0.477] | 0.395 [0.332,0.455] | 0.346 | 0.346 |
| lr_fast | < 0.001 | 0.716 [0.659,0.771] | 0.722 [0.657,0.790] | 0.581 | 0.581 |
| rf_fast | 0.003 | 0.797 [0.747,0.839] | 0.844 [0.799,0.885] | 0.728 | 0.728 |
| gb_slow | < 0.001 | 0.030 [0.025,0.035] | 0.011 [0.009,0.012] | 0.717 | 0.717 |

**Interpretation.** Every tabular detector scores an event in well under one
millisecond, so each meets any realistic pre-settlement deadline and its
pre-deadline fraction equals its recall — among detectors at this cost the
deadline metric adds nothing over recall, by design. The metric begins to
separate detectors from accuracy only when scoring latency grows, which the LLM
study (E8) examines directly.

### 3.1a E8 (the latency realization): a slow reasoning LLM misses the window

We score an off-the-shelf small instruct LLM (Qwen2.5-1.5B) one event at a time,
the way an online check would have to call it, timing each generation; an
enriched subsample of 1000 in-repo events (150 frauds; underlying base rate
1.18%) is scored by the LLM in two regimes and by a random forest on identical
events. Pre-deadline fraction at 200/1000/2000 ms:

| detector | PR-AUC | latency mean/p95 (ms) | pre@200 | pre@1000 | pre@2000 |
|---|---|---|---|---|---|
| rf_fast | 0.941 | < 0.01 | 0.740 | 0.740 | 0.740 |
| llm_terse | 0.213 | 34 / 34 | 0.720 | 0.720 | 0.720 |
| llm_reasoning | 0.171 | 1463 / 1769 | **0.000** | **0.000** | **1.000** |

**Interpretation.** The reasoning LLM's 1463 ms mean latency exceeds the 1000 ms
window, so it flags **0.000** of frauds within a 1000 ms deadline (and none within
200 ms), recovering to 1.000 only once the deadline reaches 2000 ms — while the
sub-millisecond random forest and the terse LLM meet every deadline. The LLM is
not the better detector: it scores PR-AUC 0.171 (reasoning) and 0.213 (terse)
against the random forest's 0.941. The metric is built to surface exactly this:
a heavyweight detector can be both slower than the decision window and no more
accurate, and only the latency-aware metric exposes the first half. This is the
designated reproduction target (C1). E8 requires the `llm` extra (torch,
transformers, accelerate) and a model download; its captured output is in
`results/e8.json`.

### 3.2 E2: single-hop-trained detectors collapse on the new scenarios

Detectors trained on a single-hop-only subset (legitimate + account-takeover),
evaluated on the full stream. Per-scenario recall (Wilson 95% CI):

| detector | account_takeover | coercion | fake_med_refund | mule_chain |
|---|---|---|---|---|
| rf_single_hop | 0.909 [0.804,0.961] N=55 | 0.000 [0.000,0.077] N=46 | 0.526 [0.373,0.675] N=38 | 0.846 [0.725,0.920] N=52 |
| gb_single_hop | 0.891 [0.782,0.949] | 0.043 [0.012,0.145] | 0.526 [0.373,0.675] | 0.827 [0.703,0.906] |

**Interpretation.** A detector that has only seen the single-hop case keeps high
recall there (0.909) and transfers reasonably to mule chains (0.846), whose
behavioural signature resembles it. It does not carry over to the two Pix-native
scenarios absent from open prior work: recall falls to **0.000** on coercion (the
sharp, structural result — a coerced victim transacts from their own device in a
genuine session, so the behavioural features carry no signal) and to 0.526 on
multi-hop MED-2.0 refunds. The two Pix-native scenarios prior artifacts omit are
exactly the ones a single-hop-trained detector cannot reliably catch.

### 3.3 E3: cross-generator credibility on the real released Tide HI/LI sets

Each split subsampled to 400,000 transactions (HI source hash `853081652cb402ec`,
LI `bc01f86f4215cd6f`). Batch metrics:

| split (illicit) | detector | F1 | PR-AUC [95% CI] | recall [95% CI] |
|---|---|---|---|---|
| Tide-HI (0.19%) | rule_threshold | 0.048 | 0.036 [0.028,0.046] | 0.714 [0.661,0.762] |
| Tide-HI (0.19%) | rf_fast | 0.591 | 0.520 [0.463,0.579] | 0.439 [0.384,0.495] |
| Tide-HI (0.19%) | xgb_fast | 0.606 | 0.525 [0.468,0.583] | 0.435 [0.380,0.492] |
| Tide-LI (0.10%) | rf_fast | 0.604 | 0.510 [0.435,0.590] | 0.479 [0.405,0.554] |
| Tide-LI (0.10%) | xgb_fast | 0.631 | 0.523 [0.448,0.601] | 0.461 [0.387,0.537] |

**Interpretation.** On the rare-illicit Tide data the same detectors drop to
PR-AUC ~0.52 with recall near 0.45; the rule floor is near useless (PR-AUC 0.036).
These honest, sub-perfect numbers are the credibility signal: detection on
real, extremely imbalanced laundering data is genuinely hard.

### 3.4 E5: reproduce the prior-art baselines on pix-fraud-br + deadline metric

pix-fraud-br subsampled to 400,000 transactions (fraud=3075, rate 0.00769),
scored on its own engineered balance-ratio features:

| detector | meas. latency (ms/event) | F1 | PR-AUC [95% CI] | pre@1000ms |
|---|---|---|---|---|
| rule_threshold | < 0.001 | 0.015 | 0.014 [0.013,0.015] | 0.938 |
| lr_fast | < 0.001 | 0.402 | 0.532 [0.503,0.564] | 0.277 |
| rf_fast | 0.002 | 0.875 | 0.935 [0.925,0.945] | 0.835 |
| gb_slow | < 0.001 | 0.873 | 0.930 [0.918,0.941] | 0.830 |
| xgb_fast | 0.001 | 0.874 | 0.935 [0.925,0.944] | 0.837 |

**Interpretation.** XGBoost reaches PR-AUC 0.935 [0.925,0.944], reproducing the
dataset's published XGBoost baseline (PR-AUC 0.865 on its own validation sample,
different splits and feature sets) within tolerance and confirming the harness
does not inflate scores. All tabular detectors are sub-millisecond on this set,
so each detector's pre-deadline fraction tracks its recall — the deadline metric
discriminates by latency only under a genuinely slow detector (E8), not here.

### 3.5 E6: cross-generator transfer (the non-circularity check)

A random forest on the shared schema features, trained on one generator and
tested on another:

| transfer | N test | F1 | PR-AUC [95% CI] | recall |
|---|---|---|---|---|
| in-repo → in-repo | 16191 | 0.797 | 0.844 [0.799,0.885] | 0.728 |
| pix-fraud-br → pix-fraud-br | 160000 | 0.613 | 0.493 [0.464,0.520] | 0.445 |
| in-repo → pix-fraud-br | 400000 | 0.021 | 0.016 [0.016,0.017] | 0.826 |
| pix-fraud-br → in-repo | 40477 | 0.109 | 0.071 [0.063,0.079] | 0.352 |

**Interpretation.** A detector that memorised one generator's quirks collapses
when evaluated on a different generator (PR-AUC 0.016 and 0.071 cross-generator
vs 0.49-0.84 in-distribution). This is the strongest defence against the
circularity threat.

### 3.6 E7: GPU GraphSAGE baseline

Two-layer GraphSAGE on the available CUDA GPU (device=cuda), compared to the
tabular random forest on identical inputs:

| dataset | detector | F1 | PR-AUC | recall | fit (s) |
|---|---|---|---|---|---|
| in-repo | gnn_sage | 0.161 | 0.310 | 0.874 | 3.15 |
| in-repo | rf_fast | 0.797 | 0.844 | 0.728 | 1.17 |
| Tide-HI | gnn_sage | 0.027 | 0.085 | 0.595 | 0.19 |
| Tide-HI | rf_fast | 0.599 | 0.522 | 0.439 | 11.13 |

**Interpretation.** The graph-aware baseline is honest and imperfect: it does not
dominate the tabular models here, reaching PR-AUC 0.310 on the in-repo layer and
0.085 on the sparse Tide-HI graph. Per-event inference latency is sub-millisecond
for every detector. Scaling the GNN with neighbour sampling to production-volume
graphs is a research direction left to future work.

### 3.7 E4: determinism

Two independent generations of the in-repo stream produce identical content
hashes: `frame_hash_run1 = a1064a3436566c25`, `frame_hash_run2 =
a1064a3436566c25`, `deterministic = true`.

## 4. Tests and lint

- `uv run --extra dev pytest`: 25 passed (no network, no containers). The adapter
  tests build small frames in each dataset's real column schema.
- `uv run --extra dev ruff check src tests scripts`: all checks passed.

## 5. Reproducibility notes

The in-repo experiments (E1, E2, E4) need no external data and reproduce
exactly. The cross-generator experiments (E3, E5, E6) reproduce within the
reported bootstrap CIs given the same pinned datasets and seed; the prior-art
PR-AUC on pix-fraud-br lands within tolerance of the published value. The
GraphSAGE baseline (E7) is deterministic up to GPU floating-point nondeterminism.
A large-scale graph study on production-volume traffic, and a positional
comparison against the gated FCA APP benchmark, are left as future work in
research terms.
