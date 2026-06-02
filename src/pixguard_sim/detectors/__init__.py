"""Detector plugin interface and baseline detectors.

A detector implements :class:`~pixguard_sim.detectors.base.Detector`: it is
fit on a labeled training frame and produces a per-event fraud score in
``[0, 1]`` for an evaluation frame. The harness is detector-agnostic; any object
satisfying this protocol can be scored. Baseline detectors ship here so the
harness can be exercised end to end with no external models.
"""

from __future__ import annotations

from pixguard_sim.detectors.base import Detector
from pixguard_sim.detectors.gnn import GraphSageDetector, torch_available
from pixguard_sim.detectors.ml import SklearnDetector, make_ml_detector
from pixguard_sim.detectors.rule_threshold import RuleThresholdDetector

__all__ = [
    "Detector",
    "GraphSageDetector",
    "RuleThresholdDetector",
    "SklearnDetector",
    "make_ml_detector",
    "torch_available",
]
