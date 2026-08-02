"""Instruct-LLM fraud detector (a genuinely slow, plausible detector).

This detector scores each PIX event by prompting a small instruct LLM with the
event's four behavioural features and parsing a fraud probability (or yes/no)
from the generated text. It exists to answer an honest methodological question
about the harness's headline metric: with *real measured* per-event latency,
does a plausible-but-slow detector miss a realistic pre-settlement decision
deadline that the sub-microsecond tabular detectors comfortably meet?

The detector is deliberately scored event-by-event (one forward generation per
transaction), which is how an online fraud check would have to call an LLM: the
verdict for transaction *t* is needed before *t* settles, so the per-event
generation latency is the actionable latency. That per-event cost is the same
observable the harness times via
:func:`pixguard_sim.detectors.base.measure_score_latency_ms`; this class also
records the full per-event latency array on :attr:`per_event_latencies_ms` so
mean/median/p95 can be reported honestly rather than assumed.

``fit`` does not train the LLM. It only learns a calibration: it scores the
training frame and picks the score threshold's companion mapping so the raw
yes/no or probability output is turned into a ``[0, 1]`` score. No weights are
updated; the LLM is used zero-shot, as a practitioner would deploy a small
off-the-shelf instruct model without a fine-tuning budget.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd

from pixguard_sim.detectors.base import Detector

logger = logging.getLogger(__name__)

# Regex for a leading probability/percentage or a yes/no token in the output.
_PROB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")
_YES_RE = re.compile(r"\b(yes|fraud|fraudulent|suspicious)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|legit|legitimate|safe|normal)\b", re.IGNORECASE)


def _features_line(row: pd.Series) -> str:
    """Render the four behavioural features as one human-readable line."""
    amount = float(row["amount_brl"])
    device = "yes" if int(row["device_changed"]) == 1 else "no"
    payee = "yes" if int(row["new_payee"]) == 1 else "no"
    vel = int(row["payer_velocity_1h"])
    return (
        f"Pix transfer: amount R${amount:.2f}; payer device changed: {device}; "
        f"new payee (never paid before): {payee}; "
        f"payer velocity: {vel} transfers in the last hour."
    )


def _event_prompt(row: pd.Series, reasoning: bool) -> str:
    """Render one event's features into an instruct prompt.

    Two prompt regimes are offered because they trade latency for accuracy in
    exactly the way the deadline metric is meant to expose. The terse regime
    asks for a single probability token (cheapest, fastest). The reasoning
    regime asks the model to weigh the signals step by step before answering,
    which is how a practitioner would coax a usable verdict out of a small
    instruct model -- and which costs more generated tokens, hence more latency.
    """
    feats = _features_line(row)
    if not reasoning:
        return (
            "You are a Brazilian Pix fraud-detection system. Assess one instant "
            "transfer and output only a fraud probability between 0 and 1.\n"
            f"{feats}\n"
            "Is this transfer fraudulent? Reply with a single probability "
            "between 0 and 1 (e.g. 0.85). Probability:"
        )
    return (
        "You are a Brazilian Pix fraud-detection analyst. Decide whether one "
        "instant transfer is fraudulent.\n"
        f"{feats}\n"
        "Think step by step about each signal (a high amount, a changed device, "
        "a brand-new payee, and an unusually high velocity each raise fraud "
        "risk), weigh them, then conclude. Keep the reasoning to at most three "
        "short sentences, then STOP and output the verdict on its own final "
        'line, written exactly as "Probability: X" where X is a decimal number '
        "between 0 and 1 with two decimals (for example, Probability: 0.85). Do "
        "not write anything after that line."
    )


_LABELLED_PROB_RE = re.compile(
    r"probability\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE
)


def _parse_score(text: str) -> float:
    """Map an LLM completion to a fraud score in ``[0, 1]``.

    Prefers an explicit ``Probability: X`` (the format the reasoning prompt
    asks for, possibly preceded by reasoning text), then any leading numeric
    probability, then yes/no tokens. Parsing is intentionally lenient because
    small instruct models are noisy; an unparseable answer maps to a neutral
    ``0.5`` so it neither inflates nor deflates the metrics.
    """
    snippet = text.strip()

    def _norm(raw: str, ctx: str) -> float:
        val = float(raw)
        if "%" in ctx or val > 1.0:
            val = val / 100.0
        return float(min(max(val, 0.0), 1.0))

    labelled = list(_LABELLED_PROB_RE.finditer(snippet))
    if labelled:
        # The model's conclusion is the last labelled probability. Prefer the
        # last one that carries a decimal point (e.g. "0.85"): a bare integer
        # such as "Probability: 0" is often a generation cut off mid-number
        # ("0.95" truncated to "0"), so it would otherwise be misread as 0.0.
        decimal = [m for m in labelled if "." in m.group(1)]
        m = decimal[-1] if decimal else labelled[-1]
        return _norm(m.group(1), snippet[m.start(): m.end() + 1])
    m = _PROB_RE.search(snippet)
    if m is not None:
        return _norm(m.group(1), snippet[: m.end() + 1])
    if _YES_RE.search(snippet):
        return 0.9
    if _NO_RE.search(snippet):
        return 0.1
    return 0.5


class LLMDetector(Detector):
    """Score each Pix event with a small instruct LLM, one event at a time."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        name: str = "llm_qwen",
        inference_budget_ms: int = 0,
        max_new_tokens: int = 8,
        device: str | None = None,
        dtype: str = "bfloat16",
        reasoning: bool = False,
    ) -> None:
        """Construct the detector and load the model lazily on first use.

        Args:
            model_id: HuggingFace instruct model id.
            name: Detector name used in reports.
            inference_budget_ms: Descriptive metadata only (the deadline metric
                uses the measured per-event latency, never this budget).
            max_new_tokens: Generation budget per event (kept small; the answer
                is a single probability token).
            device: Force a device; defaults to cuda if available.
            dtype: Torch dtype for the model weights.
        """
        self.model_id = model_id
        self.name = name
        self.inference_budget_ms = inference_budget_ms
        self.max_new_tokens = max_new_tokens
        self.reasoning = reasoning
        self._device = device
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        # Full per-event generation latency (ms), filled by ``score``. Lets the
        # caller report median/p95 rather than only the mean the harness times.
        self.per_event_latencies_ms: np.ndarray = np.empty(0)

    def _ensure_loaded(self) -> None:
        """Load the tokenizer and model onto the accelerator once."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        dtype = getattr(torch, self._dtype)
        logger.info("LLMDetector loading %s on %s (%s)", self.model_id, device, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        # Warm up the kernels so the timed scores reflect steady-state latency,
        # not one-off CUDA/graph compilation. The warm-up is not timed.
        self._generate("warmup")

    def _generate(self, prompt: str) -> str:
        """Run one chat-formatted generation and return the decoded answer."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def fit(self, frame: pd.DataFrame) -> LLMDetector:
        """Load the model. The LLM is used zero-shot, so no weights are fit."""
        self._ensure_loaded()
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score each event one at a time, timing per-event generation latency.

        The per-event wall-clock latency of each generation is recorded on
        :attr:`per_event_latencies_ms` and used directly by the harness's
        pre-deadline metric.
        """
        self._ensure_loaded()
        scores = np.empty(len(frame), dtype="float64")
        lat = np.empty(len(frame), dtype="float64")
        import torch

        for i, (_, row) in enumerate(frame.iterrows()):
            prompt = _event_prompt(row, self.reasoning)
            t0 = time.perf_counter()
            answer = self._generate(prompt)
            if self._device == "cuda":
                torch.cuda.synchronize()
            lat[i] = (time.perf_counter() - t0) * 1000.0
            scores[i] = _parse_score(answer)
            if i < 3:
                logger.info("LLM sample %d: answer=%r -> score=%.3f (%.1fms)",
                            i, answer, scores[i], lat[i])
        self.per_event_latencies_ms = lat
        logger.info(
            "LLMDetector scored %d events: mean=%.1fms median=%.1fms p95=%.1fms",
            len(lat), float(np.mean(lat)), float(np.median(lat)),
            float(np.percentile(lat, 95)),
        )
        return scores
