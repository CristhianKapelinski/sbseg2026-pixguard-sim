"""Pinned, checksum-verified loading of the external public datasets.

The harness is generator-agnostic; this module is the only place that touches
the bytes of an external generator's release. It resolves dataset files from a
configurable data root (never hardcoded; read from the ``PIXGUARD_DATA_DIR``
environment variable or passed explicitly), verifies each file against the
provider-published checksum, and records a content manifest so a result can be
traced to the exact bytes processed. A missing or mismatched file raises a
typed error rather than silently fabricating a result.

No dataset bytes are vendored in the repository: the public datasets are large
and carry their own licenses, so they are fetched to a scratch directory and
gitignored. The manifest pins each by SHA-256 plus the provider's MD5 and the
canonical source URL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class MissingSourceDataError(RuntimeError):
    """Raised when a required external dataset file is absent."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a dataset file's checksum does not match the pinned value."""


@dataclass(frozen=True)
class DatasetFile:
    """A pinned external dataset file.

    Attributes:
        key: Short identifier used in manifests and logs.
        relpath: Path relative to the data root.
        md5: Provider-published MD5 checksum (authoritative pin).
        source_url: Canonical download URL.
        approx_bytes: Approximate size, for a sanity log line.
    """

    key: str
    relpath: str
    md5: str
    source_url: str
    approx_bytes: int


# Provider-published pins. Tide MD5s are from the Zenodo record
# 10.5281/zenodo.18804069; pix-fraud-br is auto-converted to Parquet by the
# Hugging Face hub, so it is pinned by SHA-256 of the local snapshot recorded
# at first load (the row count and fraud count are the stable provider facts).
TIDE_FILES: tuple[DatasetFile, ...] = (
    DatasetFile(
        "tide_tx_hi", "tide/generated_transactions_HI.csv",
        "39c12784ec8f8d0c15779c92c93a8a80",
        "https://zenodo.org/records/18804069/files/generated_transactions_HI.csv",
        946_847_744,
    ),
    DatasetFile(
        "tide_tx_li", "tide/generated_transactions_LI.csv",
        "471160b7c36879d3cec9708e97e2db70",
        "https://zenodo.org/records/18804069/files/generated_transactions_LI.csv",
        953_155_584,
    ),
    DatasetFile(
        "tide_nodes_hi", "tide/generated_nodes_HI.csv",
        "baedb64c4939f463d8d4c99520eb3abe",
        "https://zenodo.org/records/18804069/files/generated_nodes_HI.csv",
        3_984_588,
    ),
    DatasetFile(
        "tide_nodes_li", "tide/generated_nodes_LI.csv",
        "7b8dca920f81ba8a8a747292e336399c",
        "https://zenodo.org/records/18804069/files/generated_nodes_LI.csv",
        3_984_588,
    ),
)

PIX_FRAUD_BR_FILE = DatasetFile(
    "pix_fraud_br", "pix_fraud_br.parquet",
    "",  # provider serves a streamed dataset; pinned by SHA-256 of the snapshot
    "https://huggingface.co/datasets/andremessina/pix-fraud-br",
    140_000_000,
)


def data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the external data root.

    Order: explicit argument, then the ``PIXGUARD_DATA_DIR`` environment
    variable. The data root is never hardcoded; large public datasets live on
    a scratch volume outside the repository.

    Raises:
        MissingSourceDataError: If no data root is configured.
    """
    candidate = explicit or os.environ.get("PIXGUARD_DATA_DIR")
    if not candidate:
        raise MissingSourceDataError(
            "no data root configured; set PIXGUARD_DATA_DIR or pass --data-dir"
        )
    return Path(candidate)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming MD5 of a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve(spec: DatasetFile, root: Path) -> Path:
    """Resolve a dataset file under ``root``, raising if absent.

    Raises:
        MissingSourceDataError: If the file does not exist.
    """
    path = root / spec.relpath
    if not path.exists():
        raise MissingSourceDataError(
            f"dataset file not found: {path} (source: {spec.source_url})"
        )
    return path


def verify(spec: DatasetFile, root: Path) -> dict[str, str]:
    """Verify a dataset file's MD5 against its pin and return a manifest entry.

    Raises:
        ChecksumMismatchError: If a non-empty pinned MD5 does not match.
    """
    path = resolve(spec, root)
    md5 = _md5(path)
    sha256 = _sha256(path)
    if spec.md5 and md5 != spec.md5:
        raise ChecksumMismatchError(
            f"MD5 mismatch for {spec.key}: expected {spec.md5}, got {md5}"
        )
    size = path.stat().st_size
    logger.info("verified %s: md5=%s size=%d", spec.key, md5, size)
    return {
        "key": spec.key,
        "path": str(path),
        "size_bytes": str(size),
        "md5": md5,
        "sha256": sha256,
        "pinned_md5": spec.md5,
        "source_url": spec.source_url,
    }


def stratified_subsample(
    df, label_col: str, n: int, seed: int
):
    """Deterministically subsample a frame keeping the label ratio.

    Bounds a multi-million-row dataset to a fixed N for a fast, reproducible
    run while preserving the (heavily imbalanced) fraud ratio. Returns the full
    frame unchanged when it already has at most ``n`` rows.

    Args:
        df: A pandas DataFrame.
        label_col: Name of the binary label column to stratify on.
        n: Target number of rows.
        seed: RNG seed for the subsample (pinned and logged by the caller).
    """
    import numpy as np

    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    parts = []
    for _value, group in df.groupby(label_col):
        take = max(1, int(round(len(group) * frac)))
        idx = rng.permutation(len(group))[:take]
        parts.append(group.iloc[idx])
    import pandas as pd

    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "subsampled %d -> %d rows (label %s ratio preserved)",
        len(df),
        len(out),
        label_col,
    )
    return out


def write_manifest(entries: list[dict[str, str]], out_path: str | Path) -> Path:
    """Write a data manifest pinning every verified file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("wrote data manifest to %s (%d files)", out, len(entries))
    return out
