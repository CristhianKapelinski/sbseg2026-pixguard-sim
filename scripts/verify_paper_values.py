#!/usr/bin/env python3
"""Check every empirical number in the paper against the artifact's own outputs.

The paper hardcodes no number: each one is a ``\\newcommand`` emitted by
``make_macros.py`` from ``results/published/*.json``. This script regenerates that
macro block and compares it, macro by macro, with the frozen camera-ready copy in
``expected/paper_macros.tex``.

A mismatch means the artifact and the paper have drifted apart, which is the single
defect a reviewer cannot work around: it makes every other number in the repository
suspect. Exits non-zero so the check can be scripted, not only read.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", re.M)
ROOT = Path(__file__).resolve().parent.parent


def parse(text: str) -> dict[str, str]:
    """Map macro name to its literal body, ignoring anything that is not a macro."""
    return {m.group(1): m.group(2) for m in MACRO.finditer(text)}


def main() -> int:
    expected_path = ROOT / "expected" / "paper_macros.tex"
    if not expected_path.is_file():
        print(f"missing {expected_path}", file=sys.stderr)
        return 2

    generated = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_macros.py"), str(ROOT / "results" / "published")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    got = parse(generated)
    want = parse(expected_path.read_text(encoding="utf-8"))

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    differing = sorted(k for k in set(want) & set(got) if want[k] != got[k])

    for name in missing:
        print(f"FAIL  {name:24s} in the paper, not produced by the artifact")
    for name in differing:
        print(f"FAIL  {name:24s} paper={want[name]!r}  artifact={got[name]!r}")
    for name in extra:
        print(f"WARN  {name:24s} produced by the artifact, not used in the paper")

    failed = len(missing) + len(differing)
    passed = len(want) - failed
    print(f"\nPAPER VALUES: {passed} PASS / {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
