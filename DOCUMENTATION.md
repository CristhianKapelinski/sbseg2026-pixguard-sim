# PixGuard-Sim: Technical Documentation

This document records the problem, the design, and every experiment run in this
environment with its exact command and the real captured output, followed by an
honest list of PENDING work. All data is **synthetic** and ground-truth
labeled; no number here is a measurement of real-world fraud rates, and no
number appears unless code in this repository produced it.

## 1. Problem

PIX, the Brazilian instant-payment system, moves tens of billions of
transactions per year, and fraud has followed the scale. The Brazilian
regulatory framework has shifted from reactive refunds toward real-time
prevention: the mandate is to block a transfer "as soon as fraud is
identified", and the MED-2.0 refund mechanism traces funds across up to five
subsequent account layers. No regulation fixes a latency in seconds, so the
decision deadline is a **researcher parameter**, reported over a range.

No open benchmark scores fraud detectors against this real-time decision
window. Existing generators (AMLSim, Tide, PaySim) are jurisdiction-agnostic,
and the closest open PIX generator models only the first transfer, with no
multi-hop tracing, no MED, no coercion, and no latency metric. The gap
PixGuard-Sim fills is an evaluation harness that reports the **pre-deadline flag
fraction** alongside conventional metrics, plus two PIX-native scenarios that no
open artifact covers.

## 2. Design

### 2.1 Event schema

Every generator maps into one normalized PIX-native event record
(`src/pixguard_sim/schema.py`). Each event carries a relative timeline
(`t_init_ms`, `t_settle_ms`), a MED-layer index, and PIX-native signals (DICT
key, device change, new payee, remote-access session, coercion flag). Detectors
see only the numeric feature columns and the timeline; the label, scenario, and
identifiers are never exposed.

### 2.2 Synthetic generator (Tier A)

A lightweight scale-free base interaction graph (`base_graph.py`, Barabasi-Albert
under a fixed seed) fixes per-account device fingerprints, habitual payees, and
candidate mule accounts. Four scenario generators (`scenarios/`) emit labeled
fraud events; a legitimate generator emits the negative class. The number of
fraud events follows the configured base rate; the whole stream is deterministic
given the master seed.

The four scenarios map onto the four thematic groups of the public PIX-fraud
taxonomy of Pizzolato et al. (arXiv:2511.20902), which maps 15 scam
methodologies into four groups (Social Engineering by Authority/Trust; Social
Engineering by Refund/Benefit/Urgency; Attacks with Physical Interaction;
Software- and Remote-Access-Based Attacks) and explicitly recommends "the
creation of fraud simulators for controlled testing". This mapping is an
explicit, justified translation, not a claim attributed to that paper:

| PixGuard-Sim scenario | Taxonomy group it instantiates |
|---|---|
| `account_takeover` | Software- and Remote-Access-Based Attacks ("ghost hand" / remote access) |
| `coercion` | Attacks with Physical Interaction (robbery, express kidnapping, extortion) |
| `fake_med_refund` | Social Engineering by Refund/Benefit/Urgency (false-scheduling / wrong-Pix refund), extended to multi-hop MED-2.0 tracing |
| `mule_chain` | Dispersal topology layered on the above (funds spread across mule accounts) |

### 2.3 Detectors and harness

A detector implements `fit` and `score` and reports its decision latency
relative to each event's initiation (`detectors/base.py`). Baselines
(`detectors/`): a rule-threshold floor and scikit-learn LR/RF/GB. The harness
(`harness.py`) fits each detector on a deterministic stratified split and
reports batch precision/recall/F1/PR-AUC, the pre-deadline flag fraction over a
deadline sweep, per-scenario recall, and recall by MED layer.

The headline metric (`metrics.py`): among true-fraud events, the fraction the
detector both flags (score at or above threshold) and decides on within the
deadline window measured from the event's own initiation. A fraud caught only
after the deadline is not actionable for pre-transaction blocking and does not
count.

## 3. Experiments (real captured output)

Environment: Python 3.12 on Linux, CPU only. All four experiments run from one
command and write JSON to `results/` plus a timestamped log to `logs/`. The
configuration used is `configs/default.json` (seed 20260202, 4000 accounts,
40000 legitimate events, fraud base rate 0.012, MED max depth 5, deadline sweep
[100, 250, 500, 1000, 2000, 5000, 10000] ms).

Command:

```
pixguard-sim --config configs/default.json run --experiments E1 E2 E3 E4
```

Generated stream (logged, identical across experiments and reproducible):
`40477 events (fraud=477, rate=0.0118)`, broken down as account_takeover=146,
coercion=97, fake_med_refund=90, legit=40000, mule_chain=144. Content hash of
the generated frame: `a1064a3436566c25`. Train/eval split: train=24286
(fraud=286), eval=16191 (fraud=191).

### 3.1 E1 (main claim): the pre-deadline flag fraction is discriminative

Batch metrics on the evaluation split:

| detector | budget (ms) | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| rule_threshold | 20 | 0.466 | 0.288 | 0.356 | 0.534 |
| lr_fast | 50 | 0.984 | 0.948 | 0.965 | 0.976 |
| rf_fast | 50 | 0.989 | 0.974 | 0.982 | 0.987 |
| gb_slow | 5000 | 1.000 | 0.969 | 0.984 | 0.990 |

Pre-deadline flag fraction vs deadline (ms):

| detector | 100 | 250 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| rule_threshold | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 | 0.288 |
| lr_fast | 0.948 | 0.948 | 0.948 | 0.948 | 0.948 | 0.948 | 0.948 |
| rf_fast | 0.974 | 0.974 | 0.974 | 0.974 | 0.974 | 0.974 | 0.974 |
| gb_slow | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.969 | 0.969 |

**Interpretation.** `rf_fast` and `gb_slow` have near-identical batch metrics
(F1 0.982 vs 0.984, PR-AUC 0.987 vs 0.990), yet the pre-deadline flag fraction
separates them decisively: at the 1000 ms deadline `rf_fast` flags 0.974 of
frauds in time while `gb_slow`, whose 5000 ms inference budget exceeds the
window, flags 0.000 until the deadline reaches 5000 ms. Conventional batch P/R
cannot see this difference; the deadline metric makes it explicit. This is the
central claim of the paper and is the designated reproduction target.

### 3.2 E2: single-hop-trained detectors collapse on the new scenarios

Detectors trained on a single-hop-only subset (legitimate traffic plus
account-takeover), evaluated on the full stream. Per-scenario recall:

| detector | account_takeover | coercion | fake_med_refund | mule_chain |
|---|---|---|---|---|
| rf_single_hop | 1.000 | 0.000 | 0.000 | 0.404 |
| gb_single_hop | 1.000 | 0.000 | 0.000 | 0.404 |

Recall by MED layer (fake_med_refund), single-hop-trained: 0.0 at every layer
0 through 4.

**Interpretation.** A detector that has only ever seen the single-hop
account-takeover case keeps perfect recall on that scenario (1.000) but
collapses to 0.000 on coercion and on multi-hop MED-2.0 refunds, and recovers
only 0.404 of mule chains. The two PIX-native scenarios that prior open
artifacts omit are exactly the ones a single-hop-trained detector cannot catch.

For contrast, the recall-by-MED-layer of the **fully-trained** detectors in E1
shows the rule baseline decaying with depth (layer 0: 0.273, layer 1: 0.231,
layers 2-4: 0.000) because it has no notion of the MED-layer structure, whereas
trained ML detectors that have seen the layer feature recover deep layers (e.g.
`rf_fast`: layer 0 0.545, layers 1-4 1.000). Deep MED layers are only hard for
detectors that lack either the feature or multi-hop training data.

### 3.3 E3: generator-agnosticism on an independent generator

The identical harness and metric run on a second, differently-shaped generator
output (Tide-shaped: uniform random source/destination over a node pool,
exponential inter-arrival timestamps), adapted into the PIX schema. Source hash
`84406336afd34824`, 40000 events, 472 fraud. Split train=23999 (fraud=283),
eval=16001 (fraud=189).

| detector | budget (ms) | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| rule_threshold | 20 | 0.412 | 0.074 | 0.126 | 0.183 |
| rf_fast | 50 | 0.949 | 0.884 | 0.915 | 0.906 |

**Interpretation.** The harness produces the same report structure and the same
metric definitions on an independently-shaped input it did not author, with no
code change, demonstrating generator-agnosticism and breaking the circularity
threat of evaluating only on self-authored data.

### 3.4 E4: determinism

Two independent generations of the Tier A stream produce identical content
hashes: `frame_hash_run1 = a1064a3436566c25`, `frame_hash_run2 =
a1064a3436566c25`, `deterministic = true`. The pipeline is byte-stable under a
fixed seed.

## 4. Tests and lint

- `python -m pytest`: 20 passed (no network, no containers).
- `ruff check src tests`: all checks passed.

## 5. PENDING (harness shipped ready, not runnable here)

No fabricated numbers are reported for these; each ships a ready adapter or
driver that fails loudly without its source data.

- **P1**: reproduce the open PIX generator's four baseline detectors on its
  released 2M-row labeled set, then score them with the deadline metric.
  Adapter: `adapters.adapt_pix_fraud_br`. Needs Kaggle/Hugging Face access to
  the released data; raises `MissingSourceDataError` until the file is present.
- **P2**: AMLSim as a third base generator. Adapter: `adapters.adapt_amlsim`.
  Needs a Java build chain not provisioned here.
- **P3**: a GNN baseline on the multi-hop graph for the MED-2.0 axis. CPU is
  feasible only at small scale; a full run wants a GPU, which we deliberately do
  not provision (the project is pure-Python, no heavy ML stack).
- **P4**: a positional comparison against the FCA/Aizle APP-fraud benchmark.
  That dataset is gated; the comparison stays qualitative until access is
  granted.

Real-data validation (P1) is the strongest external claim and is left as a
ready-to-run, honestly labeled PENDING item.
