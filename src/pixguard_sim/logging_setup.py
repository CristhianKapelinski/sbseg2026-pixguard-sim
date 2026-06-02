"""Logging subsystem for PixGuard-Sim.

Per the project's ground rules, the program must log everything to both the
console and a timestamped file under ``logs/``: the resolved config, all
seeds, content hashes of every input/output, each scenario generated, and each
experiment command together with its outputs. This module provides a single
``configure_logging`` entry point and small helpers for content hashing so
that every run is auditable and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(
    logs_dir: str | Path = "logs",
    level: int = logging.INFO,
    run_tag: str | None = None,
) -> Path:
    """Configure root logging to console and a timestamped file.

    Args:
        logs_dir: Directory in which to create ``run-<timestamp>.log``.
        level: Logging level for both handlers.
        run_tag: Optional suffix added to the log file name to disambiguate
            concurrent runs.

    Returns:
        The path to the created log file.
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"run-{timestamp}"
    if run_tag:
        name = f"{name}-{run_tag}"
    log_file = logs_path / f"{name}.log"

    root = logging.getLogger()
    root.setLevel(level)
    # Remove pre-existing handlers so repeated calls (e.g. in tests) do not
    # duplicate log lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("logging configured; file=%s", log_file)
    return log_file


def content_hash(data: Any) -> str:
    """Return a short, stable SHA-256 hex digest of arbitrary data.

    Dicts are serialized with sorted keys so the digest is deterministic and
    suitable for pinning inputs/outputs in the log.
    """
    if isinstance(data, (bytes, bytearray)):
        payload = bytes(data)
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def log_kv(logger: logging.Logger, title: str, mapping: dict[str, Any]) -> None:
    """Log a titled block of key/value pairs at INFO level."""
    logger.info("%s:", title)
    for key, value in mapping.items():
        logger.info("  %s = %s", key, value)
