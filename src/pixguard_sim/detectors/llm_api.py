"""Score events with a hosted reasoning model, one transfer per request.

The local detector in :mod:`pixguard_sim.detectors.llm` measures a small model
on the machine that runs the harness. An institution scoring transfers with a
language model is more likely to call a hosted one, and then the decision
latency includes the network round trip and whatever the provider's queue adds.
That is the cost a deployment actually pays, so it is what this detector
measures: wall-clock from issuing the request to holding a parsed verdict.

Requests carry no key of their own. They go to a local endpoint that holds the
credential and forwards upstream, so nothing here reads or stores a secret.

Concurrency is a measurement decision, not just a speed one. The deadline
metric asks how long *one* transfer waits for its own verdict, which is a
single round trip regardless of how many other transfers are in flight. Issuing
a bounded number of requests in parallel therefore shortens the experiment
without changing the quantity measured, provided the bound stays low enough not
to queue behind itself; :func:`measure_concurrency_effect` checks that.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from pixguard_sim.detectors.base import Detector
from pixguard_sim.detectors.llm import _UNPARSED_SCORE, _event_prompt, _parse_score

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/messages"


class HostedLLMDetector(Detector):
    """Score each event with a hosted instruct or reasoning model."""

    def __init__(
        self,
        model: str,
        name: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        max_new_tokens: int = 1024,
        reasoning: bool = True,
        max_in_flight: int = 8,
        timeout_s: float = 180.0,
    ) -> None:
        """Configure the detector.

        Args:
            model: Provider model identifier.
            name: Detector name in reports; defaults to the model id.
            endpoint: Local forwarding endpoint that injects the credential.
            max_new_tokens: Generation cap, matching the local reasoning regime.
            reasoning: Use the deliberating prompt rather than the terse one.
            max_in_flight: Bound on concurrent requests.
            timeout_s: Per-request timeout.
        """
        self.model = model
        self.name = name or model
        self.endpoint = endpoint
        self.max_new_tokens = max_new_tokens
        self.reasoning = reasoning
        self.max_in_flight = max_in_flight
        self.timeout_s = timeout_s
        self.per_event_latencies_ms: np.ndarray = np.empty(0)
        self.unparsed_count: int = 0
        self.new_tokens: list[int] = []
        self.failed_count: int = 0

    def fit(self, frame: pd.DataFrame) -> HostedLLMDetector:
        """Nothing to fit: the model is used as released."""
        return self

    def _one(self, prompt: str) -> tuple[float, float, int, bool]:
        """Issue one request; return (score, latency_ms, output_tokens, ok)."""
        payload = json.dumps({
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=payload,
            headers={"content-type": "application/json",
                     "anthropic-version": "2023-06-01"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # A failed call is not a fast call: it is reported, never scored as
            # if the model had answered.
            logger.warning("%s: request failed (%s)", self.name, exc)
            return _UNPARSED_SCORE, (time.perf_counter() - t0) * 1000.0, 0, False
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = " ".join(
            b.get("text", "") for b in body.get("content", []) if isinstance(b, dict)
        )
        out_tokens = int(body.get("usage", {}).get("output_tokens", 0))
        return _parse_score(text), latency_ms, out_tokens, True

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score every event, recording each one's own round-trip latency."""
        prompts = [_event_prompt(row, self.reasoning) for _, row in frame.iterrows()]
        scores = np.empty(len(frame), dtype="float64")
        lat = np.empty(len(frame), dtype="float64")
        toks: list[int] = [0] * len(frame)
        unparsed = failed = 0
        with futures.ThreadPoolExecutor(max_workers=self.max_in_flight) as pool:
            for i, (s, ms, n, ok) in enumerate(pool.map(self._one, prompts)):
                scores[i], lat[i], toks[i] = s, ms, n
                unparsed += int(s == _UNPARSED_SCORE and ok)
                failed += int(not ok)
        self.per_event_latencies_ms = lat
        self.unparsed_count = int(unparsed)
        self.failed_count = int(failed)
        self.new_tokens = toks
        self.measured_latency_ms = float(lat.mean()) if lat.size else 0.0
        logger.info(
            "%s: %d events, median %.0f ms, median %d output tokens, "
            "%d unparsed, %d failed",
            self.name, len(frame), float(np.median(lat)) if lat.size else 0.0,
            int(np.median(toks)) if toks else 0, unparsed, failed,
        )
        return scores


def measure_concurrency_effect(
    detector: HostedLLMDetector, frame: pd.DataFrame, n: int = 12
) -> dict[str, float]:
    """Compare per-request latency issued serially and in parallel.

    If the parallel median is close to the serial one, the bound is low enough
    that a request does not wait behind its siblings, and the shortened run
    measures the same per-transfer quantity.
    """
    sample = frame.head(n)
    prompts = [_event_prompt(row, detector.reasoning) for _, row in sample.iterrows()]
    serial = [detector._one(p)[1] for p in prompts]
    with futures.ThreadPoolExecutor(max_workers=detector.max_in_flight) as pool:
        parallel = [r[1] for r in pool.map(detector._one, prompts)]
    return {
        "n": float(len(prompts)),
        "serial_median_ms": float(np.median(serial)),
        "parallel_median_ms": float(np.median(parallel)),
        "inflation": float(np.median(parallel) / np.median(serial)) if serial else 0.0,
    }
