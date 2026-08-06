# Optional commands

**Nothing on this page is needed for artifact evaluation.** The reviewer path is the
[README](README.md): the minimal test, the three claims, and `./scripts/make_figures.sh`.
Those cover every seal and every number in the paper. The commands below exist for a reader
who wants to go further, and each one costs something the reviewer path deliberately avoids:
a model download, a GPU, a network credential, or a couple of gigabytes of third-party data.

Every command here is run from the repository root.

## Supporting experiments the claims do not need

**E4, determinism.** Regenerates the event stream twice and checks the frame hashes are equal.
The claims already rest on fixed seeds; this is the explicit check.

```bash
uv run pixguard-sim --config configs/default.json run --experiments E4
```

Writes `results/e4.json` with `"deterministic": true`.

**E7, the GraphSAGE baseline.** A graph neural network scored on the same events, to place the
tabular detectors against a heavier model. Needs the `gnn` extra (`torch`, about 1 GB) and is
much faster with a CUDA GPU.

```bash
uv run --extra gnn pixguard-sim --config configs/default.json run --experiments E7
```

Its published numbers are in [`results/published/e7.json`](results/published/e7.json) and
Section 3.6 of [`DOCUMENTATION.md`](DOCUMENTATION.md).

## The language-model experiments (E8 and E9)

These are the second half of Claim #1: the part where a detector is slow enough that the
deadline actually binds. They are not in the reviewer path because E8 downloads a model and E9
needs a credential, and neither is required to check the claim, whose captured outputs ship in
`results/published/`.

**E8, a local instruct model.** Scores the fraud-enriched 1000-event subsample with a small
model on this machine. Needs the `llm` extra; a CUDA GPU makes it minutes instead of tens of
minutes, and it falls back to CPU.

```bash
uv run --extra llm pixguard-sim --config configs/default.json run --experiments E8
```

**A cheap latency check against a hosted model.** Ten events through one flash-tier model,
instead of the paper's thousand through two reasoning models. Latency is a property of the
round trip rather than of how many events you send, so ten are enough to watch the 1.5 s
budget be missed; accuracy over ten events is meaningless, so the script reports timing and
gates on nothing.

```bash
PIXGUARD_LLM_ENDPOINT=https://your-provider.example/v1/messages \
PIXGUARD_LLM_API_KEY=sk-... \
./scripts/llm_smoke.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `PIXGUARD_LLM_ENDPOINT` | `http://127.0.0.1:8080/v1/messages` | Where requests go. Unset, they go to a local forwarder that holds the credential. |
| `PIXGUARD_LLM_API_KEY` | unset | Sent as a header when set. Read from the environment, never written to a results file, never stored in this repository. |
| `PIXGUARD_LLM_MODEL` | `deepseek-v4-flash` | Model identifier. |
| `PIXGUARD_LLM_SMOKE_N` | `10` | Events to score. |

**E9 as the paper ran it.** The same 1000 events, drawn with the same seed as E8, through two
hosted reasoning models. This is the expensive one.

```bash
uv run python scripts/run_hosted_llm.py --models deepseek-v4-flash,deepseek-v4-pro
```

Published results: PR-AUC 0.846 and 0.849, p95 latency 5025 ms and 10 329 ms, and
`pre_deadline_1500ms` of 0.000 for both, in
[`results/published/e9_hosted.json`](results/published/e9_hosted.json).

## Regenerating one figure at a time

`./scripts/make_figures.sh` runs all of these in order and then verifies the paper values; it
is the command the README gives. Use the individual scripts only when you are changing one
figure. They read the committed results and the generated event stream, so generate the stream
first.

```bash
uv run pixguard-sim generate --output data/events.csv
uv run --extra figures python scripts/make_figure.py results/published figs/fig_results.pdf data/events.csv
uv run --extra figures python scripts/make_dataset_figures.py
uv run --extra figures python scripts/make_pipeline_figure.py results/published figs/pipeline.pdf data/events.csv
uv run python scripts/make_macros.py results/published
```

`make_macros.py` prints the LaTeX macro block to stdout; the paper's copy is the frozen
[`expected/paper_macros.tex`](expected/paper_macros.tex), and
`scripts/verify_paper_values.py` is what compares the two.

## Fetching the third-party generators on their own

`./scripts/claim3.sh --run` does this as part of the claim. Run it separately only to stage the
download ahead of time.

```bash
uv run --extra datasets python scripts/fetch_pix_fraud_br.py data/pix_fraud_br.parquet
uv run --extra datasets pixguard-sim --config configs/default.json --data-dir data manifest
```
