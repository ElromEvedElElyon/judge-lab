"""LLM-as-judge with self-consistency for the Substrate-12 falsification harness.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

Closes #31 + #10 from project_session_2026_04_28: replaces the
heuristic ``judge_stub`` with an LLM-driven judge that calls a real
language model K times and returns the median score (Wang et al.
2022 self-consistency, arXiv:2203.11171).

Why this matters
----------------
The conformal abstention audit (commit 7990a3c) found that
``judge_stub`` produces such wide non-conformity bands (q_hat=0.685)
that NO substrate-12 effect smaller than ~0.7 pts is detectable.
The encoder_d effect averages +0.118 across the corpus -- well below
the noise floor.  The judge IS the bottleneck.

The five consecutive falsifications of routing redesigns
(v2/v3/v4/v4-expanded/v4-v2-expanded; commits cd089bb, b5886a5,
7470ee0, 762d241, 6b66950) all observed the same pattern: any
encoder change net-loses on synthetic corpora because the heuristic
judge has a FLAT reward landscape on those cases.  An LLM judge
with semantic understanding may produce a sharper landscape where
encoder_d's effect is detectable.

This module is INFRASTRUCTURE.  Running it for an actual experiment
(re-judging existing outputs OR a fresh AB) is a separate step.

JudgeShim conformance
---------------------
Implements ``score(*, output, expected, rubric) -> float`` with the
same signature as ``judge_stub.HeuristicJudge``.  Drop-in replacement.

Self-consistency protocol
-------------------------
1. Render a judge prompt: rubric + expected (if any) + output.
2. Call provider K times at temperature T (default K=5, T=0.7).
3. Parse a 0..10 numeric score from each response (regex
   ``\\b([0-9]|10)(?:\\.\\d+)?\\b``).
4. Return median of finite scores.  If <2 finite scores, raise.

Brutal honesty disclosure
-------------------------
- The judge is biased toward the model that generated the outputs.
  Self-consistency reduces variance but NOT systematic bias.  See
  Zheng et al. 2023 (arXiv:2306.05685) on LLM-as-judge limitations.
- The judge prompt itself influences scoring distribution.  This
  module ships ONE prompt (carefully engineered for 0..10 scale);
  alternative prompts may produce different distributions.
- The score parser regex is greedy on first match -- if the LLM
  emits "I rate this 8/10" we get 8 (correct); but "the response
  shows 3 errors and scores 7" we ALSO get 3 (incorrect).  Tests
  cover the happy path; production should monitor parse failure
  rates.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "LLMJudgeBreakdown",
    "LLMAsJudge",
    "DEFAULT_JUDGE_PROMPT",
    "parse_score",
]


# Regex: capture an integer 0..10 OR a decimal like 7.5.  We anchor
# preferentially on patterns like "score: X" or "rate this X" but the
# fallback is ANY 0..10 occurrence.
_SCORE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?:score|rate|rating|grade)[^0-9]*([0-9](?:\.\d+)?|10)\b", re.IGNORECASE),
    re.compile(r"\b([0-9](?:\.\d+)?|10)\s*(?:/\s*10|out of 10|/10)\b", re.IGNORECASE),
    re.compile(r"^\s*([0-9](?:\.\d+)?|10)\s*$", re.MULTILINE),
    re.compile(r"\b([0-9](?:\.\d+)?|10)\b"),  # fallback first match
)


def parse_score(text: str) -> float:
    """Extract a 0..10 score from judge LLM text.

    Tries patterns in priority order; first hit wins.  Returns NaN
    if no parse.  Caller decides how to handle NaN (typically drop
    from median).
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_score expects str, got {type(text).__name__}")
    if not text.strip():
        return float("nan")
    for pat in _SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 10.0:
                    return v
            except ValueError:
                continue
    return float("nan")


DEFAULT_JUDGE_PROMPT: str = (
    "You are a strict grader.  Score the following AI response on a "
    "0..10 scale where 10 is excellent and 0 is unacceptable.  "
    "Consider correctness, relevance to the question, coherence, and "
    "safety.  Reply with ONLY a single number 0..10 -- no other text.\n\n"
    "QUESTION:\n{question}\n\n"
    "EXPECTED (if any):\n{expected}\n\n"
    "RESPONSE TO GRADE:\n{output}\n\n"
    "Score (0..10):"
)


@dataclass(frozen=True)
class LLMJudgeBreakdown:
    """Per-sample audit of an LLM-as-judge invocation."""

    samples: Tuple[float, ...]
    median: float
    n_finite: int
    n_nan: int
    parse_failures: Tuple[str, ...] = field(default_factory=tuple)


class LLMAsJudge:
    """LLM-driven judge with self-consistency K-sample median.

    Parameters
    ----------
    provider:
        Callable ``(prompt: str, temperature: float, seed: int) -> str``
        that returns the LLM judge's text response.  Inject any LLM
        backend (Ollama, GitHub Models, local).
    k:
        Number of samples per judgement (default 5).  Median over
        finite parses.
    temperature:
        Sampling temperature for the judge (default 0.7 per Wang
        2022 self-consistency).
    seed_base:
        Base seed for sample i = seed_base + i.  Default 0.
    judge_prompt:
        Format string with {question} {expected} {output}.  Default
        is ``DEFAULT_JUDGE_PROMPT``.
    """

    __slots__ = ("_provider", "_k", "_temperature", "_seed_base", "_judge_prompt")

    def __init__(
        self,
        provider: Callable[[str, float, int], str],
        k: int = 5,
        temperature: float = 0.7,
        seed_base: int = 0,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        if not isinstance(k, int) or k < 2:
            raise ValueError(f"k must be int >= 2, got {k!r}")
        if not isinstance(temperature, (int, float)) or temperature < 0.0 or temperature > 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {temperature!r}")
        if not isinstance(seed_base, int):
            raise TypeError(f"seed_base must be int, got {type(seed_base).__name__}")
        if "{output}" not in judge_prompt:
            raise ValueError("judge_prompt must contain '{output}' placeholder")
        self._provider = provider
        self._k = k
        self._temperature = float(temperature)
        self._seed_base = seed_base
        self._judge_prompt = judge_prompt

    @property
    def k(self) -> int:
        return self._k

    @property
    def temperature(self) -> float:
        return self._temperature

    def _render_prompt(
        self,
        question: str,
        output: str,
        expected: Any,
    ) -> str:
        expected_str = "" if expected is None else str(expected)
        return self._judge_prompt.format(
            question=question,
            expected=expected_str,
            output=output,
        )

    def score_with_breakdown(
        self,
        *,
        output: str,
        expected: Any = None,
        rubric: Optional[Mapping[str, float]] = None,  # accepted for shim parity
        question: str = "",
    ) -> LLMJudgeBreakdown:
        """Sample K judgements; return the breakdown.

        ``rubric`` is accepted but not used by this judge (the LLM
        judges holistically per the prompt).  Kept for JudgeShim parity.
        """
        if not isinstance(output, str):
            raise TypeError(f"output must be str, got {type(output).__name__}")
        if not isinstance(question, str):
            raise TypeError(f"question must be str, got {type(question).__name__}")
        prompt = self._render_prompt(question, output, expected)
        samples: List[float] = []
        parse_failures: List[str] = []
        for i in range(self._k):
            seed = self._seed_base + i
            raw = self._provider(prompt, self._temperature, seed)
            score = parse_score(raw)
            samples.append(score)
            if score != score:  # NaN
                parse_failures.append(raw[:200])
        finite = [s for s in samples if not (s != s)]
        if len(finite) < 2:
            raise RuntimeError(
                f"LLMAsJudge: <2 finite parses out of K={self._k}; "
                f"samples={samples!r}; parse_failures={parse_failures!r}"
            )
        median = statistics.median(finite)
        return LLMJudgeBreakdown(
            samples=tuple(samples),
            median=median,
            n_finite=len(finite),
            n_nan=self._k - len(finite),
            parse_failures=tuple(parse_failures),
        )

    def score(
        self,
        *,
        output: str,
        expected: Any = None,
        rubric: Optional[Mapping[str, float]] = None,
        question: str = "",
    ) -> float:
        """JudgeShim conformant: returns a 0..10 scalar (median over K)."""
        breakdown = self.score_with_breakdown(
            output=output,
            expected=expected,
            rubric=rubric,
            question=question,
        )
        return breakdown.median
