#!/usr/bin/env bash
# Regenerates the three figures the paper prints, plus the LaTeX macro block that
# carries every number in its prose. One command, because the generators need the
# event stream produced first and a reviewer should not have to know that.
#
# Output lands in figs/ next to the committed copies, so `git status` after this run
# is itself the check: no diff means the artifact reproduced the published figures.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Generating the event stream the figures read"
uv run pixguard-sim generate --output data/events.csv >/dev/null

echo "==> Figures"
uv run --extra figures python scripts/make_figure.py results/published figs/fig_results.pdf data/events.csv
uv run --extra figures python scripts/make_dataset_figures.py
uv run --extra figures python scripts/make_pipeline_figure.py results/published figs/pipeline.pdf data/events.csv

echo
echo "==> Paper macros (every empirical number in the prose)"
uv run python scripts/make_macros.py results/published > /dev/null
uv run python scripts/verify_paper_values.py
