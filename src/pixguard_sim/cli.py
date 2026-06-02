"""Command-line interface for PixGuard-Sim.

Exposes three subcommands:

* ``generate``  -- write a labeled synthetic PIX event stream to a CSV.
* ``run``       -- run one or more experiments (E1-E4) and write JSON results.
* ``reproduce`` -- run the designated main-claim experiment end to end.

Every invocation configures logging to console and a timestamped log file,
logs the resolved config, seeds, and input/output content hashes, and records
the exact command and its outputs, per the project's ground rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pixguard_sim import __version__
from pixguard_sim.config import PipelineConfig
from pixguard_sim.experiments import EXPERIMENTS
from pixguard_sim.generator import generate_events
from pixguard_sim.logging_setup import configure_logging, content_hash, log_kv

logger = logging.getLogger("pixguard_sim.cli")


def _load_config(path: str | None) -> PipelineConfig:
    """Load a pipeline config from a file or return defaults."""
    if path:
        cfg = PipelineConfig.load(path)
        logger.info("loaded config from %s", path)
    else:
        cfg = PipelineConfig()
        logger.info("using default config")
    return cfg


def _log_run_header(cfg: PipelineConfig, command: str) -> None:
    """Log version, command, resolved config, seed, and config hash."""
    log_kv(
        logger,
        "run header",
        {
            "version": __version__,
            "command": command,
            "config_hash": content_hash(cfg.to_json()),
            "seed": cfg.generator.seed,
        },
    )
    logger.info("resolved config: %s", cfg.to_json())


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate the synthetic event stream and write it to CSV."""
    cfg = _load_config(args.config)
    _log_run_header(cfg, "generate")
    frame = generate_events(cfg.generator)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_text = frame.to_csv(index=False)
    out.write_text(csv_text, encoding="utf-8")
    logger.info(
        "wrote %d events to %s (hash=%s)", len(frame), out, content_hash(csv_text)
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the requested experiments and write their JSON results."""
    cfg = _load_config(args.config)
    _log_run_header(cfg, f"run {args.experiments}")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    requested = args.experiments or list(EXPERIMENTS.keys())
    summary: dict[str, str] = {}
    for name in requested:
        if name not in EXPERIMENTS:
            logger.error("unknown experiment: %s", name)
            return 2
        logger.info("running experiment %s", name)
        result = EXPERIMENTS[name](cfg)
        out = results_dir / f"{name.lower()}.json"
        payload = json.dumps(result, indent=2, sort_keys=True)
        out.write_text(payload, encoding="utf-8")
        logger.info(
            "wrote %s result to %s (hash=%s)", name, out, content_hash(payload)
        )
        summary[name] = str(out)
    logger.info("experiment summary: %s", summary)
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Run the designated main-claim experiment (E1) end to end."""
    args.experiments = ["E1"]
    return cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="pixguard-sim",
        description="Deadline-aware evaluation harness for PIX fraud detectors.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default=None, help="Path to a JSON config.")
    parser.add_argument("--logs-dir", default="logs", help="Directory for logs.")
    parser.add_argument(
        "--results-dir", default="results", help="Directory for JSON results."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a labeled PIX event stream.")
    gen.add_argument(
        "--output", default="results/events.csv", help="Output CSV path."
    )
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="Run experiments E1-E4.")
    run.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of E1 E2 E3 E4 (default: all).",
    )
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("reproduce", help="Run the main-claim experiment (E1).")
    rep.set_defaults(func=cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(logs_dir=args.logs_dir, run_tag=args.command)
    logger.info("pixguard-sim %s invoked: %s", __version__, " ".join(sys.argv[1:]))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
