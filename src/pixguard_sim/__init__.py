"""PixGuard-Sim: a deadline-aware evaluation harness for PIX fraud detectors.

The package provides (i) a deterministic, clearly-synthetic generator of
labeled PIX-style transaction events covering four PIX-native fraud scenarios,
(ii) a detector plugin interface with baseline detectors, and (iii) a
latency-aware evaluation harness that scores detectors by the fraction of
frauds flagged before a configurable pre-transaction decision deadline, in
addition to conventional precision/recall/F1/PR-AUC.

All data produced by this package is SYNTHETIC. It is generated from an
explicit, seeded generative model anchored to public statistics and authentic
PIX/DICT/MED event-schema fields. It is never a measurement of real-world
fraud rates.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
