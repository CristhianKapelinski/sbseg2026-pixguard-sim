#!/usr/bin/env python3
"""Materialize the public pix-fraud-br generator as a local Parquet snapshot.

Lives in a file rather than a heredoc inside the claim script so the reviewer never
has to paste multi-line Python, and so the fetch can be rerun on its own.
"""

from __future__ import annotations

import sys

from datasets import load_dataset


def main() -> int:
    out = sys.argv[1]
    load_dataset("andremessina/pix-fraud-br", split="train").to_pandas().to_parquet(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
