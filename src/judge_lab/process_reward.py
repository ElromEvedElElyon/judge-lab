"""Process Reward Model (PRM) judge over decomposed reasoning steps.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

This module implements Approach #29 of the Substrate-12 falsification
harness pre-registered in
``docs/r26/ULTRA_DEEP_APPROACHES_PART3_2026_04_28.md``: a Process
Reward Model (PRM) style judge that scores each reasoning step in a
response separately, then aggregates per-step scores via ``min``,
``product`` or ``mean``.  The motivating hypothesis is that the
substrate-12 belief vector may preserve early-step quality while
degrading late-step (or vice versa); a whole-response outcome judge
averages this signal away, while a PRM exposes WHERE the substrate
helps or hurts.

Method
------
1. ``split_steps`` segments the response by:
   * Double-newlines (``\\n\\n+``) as paragraph boundaries.
   * Numbered enumerations (``^\\s*\\d+[\\.\\)]\\s+``).
   * ``Step N:`` / ``Etape N:`` / ``Passo N:`` prefixes (case-insensitive).
   * Markdown bullet markers (``^\\s*[-*+]\\s+``).
   Empty steps are filtered.  A response with no boundary markers is
   returned as a single-element list.

2. ``ProcessRewardJudge.evaluate`` calls the injected
   ``step_judge_callable`` once per step and stores the resulting
   ``StepRecord`` tuple.  When the response has fewer than
   ``min_steps_for_prm`` segments, the judge falls back to a single
   whole-response scoring pass (``n_steps == 1``).

3. Aggregates returned in every ``PRMResult``:
   * ``aggregate_min`` -- strictest: a single bad step caps the score.
   * ``aggregate_product`` -- multiplicative: penalises long bad chains.
   * ``aggregate_mean`` -- simple average: comparable to outcome judge.

4. ``compare_prm`` returns ``aggregate_b - aggregate_a`` under the
   requested aggregator (positive = B wins), and
   ``per_step_delta_distribution`` returns the raw zero-padded
   element-wise delta distribution for downstream KL / Wasserstein
   analyses (kept out of this module to remain hermetic).

Honest disclosure
-----------------
The PRM is only as good as the segmenter.  This regex-based segmenter
is intentionally simple: many real-world LLM outputs do NOT carry
``\\n\\n``, numbered enumerations, ``Step N:`` prefixes, or markdown
bullets, in which case the response collapses to ``n_steps == 1`` and
PRM aggregation reduces to whole-response outcome scoring.  Robust
step segmentation (semantic chunking, dependency parsing, model-based
boundary detection) is out of scope for the hermetic falsification
harness; we ship the brittle-but-deterministic regex variant and
document the failure mode here.

Citations
---------
- H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker,
  T. Lee, J. Leike, J. Schulman, I. Sutskever, K. Cobbe,
  "Let's Verify Step by Step," arXiv:2305.20050, 2023 (PRM800K).
- K. Cobbe, V. Kosaraju, M. Bavarian, et al., "Training Verifiers to
  Solve Math Word Problems," arXiv:2110.14168, 2021 (GSM8K, PRM/ORM
  introduction).
- L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, R. Agarwal,
  "Generative Verifiers: Reward Modeling as Next-Token Prediction"
  (GenRM-CoT), arXiv:2408.15240, 2024.

Design constraints
------------------
* Pure stdlib (``re``, ``math``, ``dataclasses``, ``typing``).
* Hermetic: no I/O, no network, no LLM calls.  The
  ``step_judge_callable`` is injected by the caller.
* Frozen dataclasses for records and report so callers can store and
  hash them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

#: Compiled boundary patterns used by :func:`split_steps`.  Order matters:
#: paragraph (``\n\n``) splits first, then per-line markers re-split each
#: paragraph if it carries an enumerator.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")

# Numbered enumerator at line start, e.g. "1." "2)" " 12. "
_NUMBERED_RE = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)

# "Step N:", "Etape N:", "Passo N:" -- case-insensitive, optional accents
# handled by allowing both ``Etape`` and ``Etape`` ASCII fallback.  Pure
# stdlib means we keep this ASCII; real usage will mostly be English.
_STEP_PREFIX_RE = re.compile(
    r"^\s*(?:step|etape|étape|passo|paso)\s*\d+\s*[:\-\.]\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Markdown bullets at line start: -, *, +
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


STEP_BOUNDARY_PATTERNS: Tuple[re.Pattern, ...] = (
    _PARAGRAPH_SPLIT,
    _NUMBERED_RE,
    _STEP_PREFIX_RE,
    _BULLET_RE,
)

DEFAULT_AGGREGATOR: str = "min"
_VALID_AGGREGATORS: Tuple[str, ...] = ("min", "product", "mean")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One PRM-scored reasoning step.

    Attributes
    ----------
    step_idx
        Zero-based index of the step within the response.
    text
        Trimmed text of the step.  Must be non-empty.
    score
        Per-step score in ``[0, 1]``.
    """

    step_idx: int
    text: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.step_idx, int) or self.step_idx < 0:
            raise ValueError(
                "StepRecord.step_idx must be a non-negative int, "
                f"got {self.step_idx!r}"
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("StepRecord.text must be a non-empty string")
        if not isinstance(self.score, (int, float)) or math.isnan(
            float(self.score)
        ):
            raise ValueError(
                f"StepRecord.score must be a real number, got {self.score!r}"
            )
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(
                f"StepRecord.score must be in [0, 1], got {self.score!r}"
            )


@dataclass(frozen=True)
class PRMResult:
    """Output of a single PRM evaluation.

    Attributes
    ----------
    response
        The full response text that was evaluated.
    steps
        Tuple of :class:`StepRecord` -- one per detected step (or one
        whole-response record when fallback was triggered).
    aggregate_min
        Min over per-step scores.
    aggregate_product
        Product over per-step scores.
    aggregate_mean
        Arithmetic mean of per-step scores.
    n_steps
        Number of steps actually scored (always ``len(steps)``).
    """

    response: str
    steps: Tuple[StepRecord, ...]
    aggregate_min: float
    aggregate_product: float
    aggregate_mean: float
    n_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.response, str):
            raise ValueError("PRMResult.response must be str")
        if not isinstance(self.steps, tuple) or not all(
            isinstance(s, StepRecord) for s in self.steps
        ):
            raise ValueError("PRMResult.steps must be tuple[StepRecord, ...]")
        if self.n_steps != len(self.steps):
            raise ValueError(
                "PRMResult.n_steps must match len(steps): "
                f"{self.n_steps} != {len(self.steps)}"
            )
        if self.n_steps < 1:
            raise ValueError("PRMResult must have at least one step")
        for name, val in (
            ("aggregate_min", self.aggregate_min),
            ("aggregate_product", self.aggregate_product),
            ("aggregate_mean", self.aggregate_mean),
        ):
            if not isinstance(val, (int, float)) or math.isnan(float(val)):
                raise ValueError(f"PRMResult.{name} must be a real number")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(
                    f"PRMResult.{name} must be in [0, 1], got {val!r}"
                )


# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------


def _split_paragraph_by_markers(paragraph: str) -> List[str]:
    """Re-split a single paragraph by per-line enumerator markers.

    Returns at least one entry if the input is non-empty after stripping.
    """

    if not paragraph or not paragraph.strip():
        return []

    # Find all marker positions across the per-line patterns.
    positions: List[int] = []
    for pat in (_NUMBERED_RE, _STEP_PREFIX_RE, _BULLET_RE):
        for m in pat.finditer(paragraph):
            positions.append(m.start())

    if not positions:
        stripped = paragraph.strip()
        return [stripped] if stripped else []

    positions = sorted(set(positions))

    # If the first marker is not at index 0, treat any leading text as
    # its own step.
    chunks: List[str] = []
    if positions[0] > 0:
        head = paragraph[: positions[0]].strip()
        if head:
            chunks.append(head)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(paragraph)
        piece = paragraph[start:end].strip()
        if piece:
            chunks.append(piece)

    return chunks


def split_steps(response: str) -> List[str]:
    """Segment ``response`` into reasoning steps.

    See module docstring for the recognised boundary markers.

    Edge cases
    ----------
    * ``""`` -> ``[]``
    * Whitespace-only -> ``[]``
    * No boundary markers -> single-element list with the trimmed body.
    """

    if not isinstance(response, str):
        raise TypeError("split_steps expects str, got " + type(response).__name__)
    if not response or not response.strip():
        return []

    paragraphs = _PARAGRAPH_SPLIT.split(response)
    steps: List[str] = []
    for para in paragraphs:
        steps.extend(_split_paragraph_by_markers(para))

    return steps


# ---------------------------------------------------------------------------
# PRM judge
# ---------------------------------------------------------------------------


def _aggregate(scores: List[float], aggregator: str) -> float:
    """Aggregate per-step scores under one of the supported reducers."""

    if aggregator not in _VALID_AGGREGATORS:
        raise ValueError(
            f"aggregator must be one of {_VALID_AGGREGATORS}, "
            f"got {aggregator!r}"
        )
    if not scores:
        raise ValueError("Cannot aggregate empty score list")

    if aggregator == "min":
        return float(min(scores))
    if aggregator == "product":
        prod = 1.0
        for s in scores:
            prod *= float(s)
        return float(prod)
    # mean
    return float(sum(scores) / len(scores))


class ProcessRewardJudge:
    """Process Reward Model judge over a step-segmented response.

    Parameters
    ----------
    step_judge_callable
        Callable with signature
        ``(prompt, response, step_text, step_idx) -> float`` returning a
        score in ``[0, 1]``.
    min_steps_for_prm
        Minimum number of detected steps required to apply per-step PRM
        scoring.  Below this threshold the judge falls back to a single
        whole-response scoring call.  Must be >= 1.
    aggregator
        Default aggregator for :class:`PRMResult`.  One of
        ``{"min", "product", "mean"}``.  Note: regardless of this
        setting all three aggregates are always computed and stored on
        the result; this only fixes the *default* used by helpers.
    """

    __slots__ = ("_judge", "_min_steps", "_default_aggregator")

    def __init__(
        self,
        step_judge_callable: Callable[[str, str, str, int], float],
        min_steps_for_prm: int = 2,
        aggregator: str = DEFAULT_AGGREGATOR,
    ) -> None:
        if not callable(step_judge_callable):
            raise TypeError("step_judge_callable must be callable")
        if not isinstance(min_steps_for_prm, int) or min_steps_for_prm < 1:
            raise ValueError(
                "min_steps_for_prm must be a positive int, "
                f"got {min_steps_for_prm!r}"
            )
        if aggregator not in _VALID_AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {_VALID_AGGREGATORS}, "
                f"got {aggregator!r}"
            )
        self._judge = step_judge_callable
        self._min_steps = min_steps_for_prm
        self._default_aggregator = aggregator

    @property
    def aggregator(self) -> str:
        return self._default_aggregator

    @property
    def min_steps_for_prm(self) -> int:
        return self._min_steps

    def _call_judge(
        self, prompt: str, response: str, step_text: str, step_idx: int
    ) -> float:
        raw = self._judge(prompt, response, step_text, step_idx)
        try:
            score = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"step_judge_callable must return a real number, got {raw!r}"
            ) from exc
        if math.isnan(score) or not (0.0 <= score <= 1.0):
            raise ValueError(
                f"step_judge_callable must return a score in [0, 1], "
                f"got {score!r}"
            )
        return score

    def evaluate(self, prompt: str, response: str) -> PRMResult:
        """Score ``response`` step-by-step and return a :class:`PRMResult`.

        Falls back to whole-response scoring if fewer than
        ``min_steps_for_prm`` segments are detected.
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be str")
        if not isinstance(response, str):
            raise TypeError("response must be str")

        segments = split_steps(response)

        if len(segments) < self._min_steps:
            # Fallback: single whole-response scoring pass.
            whole = response.strip() if response.strip() else response
            # StepRecord requires non-empty text -- guard this for
            # whitespace-only responses by substituting a single space
            # placeholder.  (Production callers should pre-validate.)
            if not whole.strip():
                whole = "<empty>"
            score = self._call_judge(prompt, response, whole, 0)
            record = StepRecord(step_idx=0, text=whole, score=score)
            return PRMResult(
                response=response,
                steps=(record,),
                aggregate_min=score,
                aggregate_product=score,
                aggregate_mean=score,
                n_steps=1,
            )

        records: List[StepRecord] = []
        scores: List[float] = []
        for idx, seg in enumerate(segments):
            score = self._call_judge(prompt, response, seg, idx)
            records.append(StepRecord(step_idx=idx, text=seg, score=score))
            scores.append(score)

        agg_min = _aggregate(scores, "min")
        agg_prod = _aggregate(scores, "product")
        agg_mean = _aggregate(scores, "mean")

        return PRMResult(
            response=response,
            steps=tuple(records),
            aggregate_min=agg_min,
            aggregate_product=agg_prod,
            aggregate_mean=agg_mean,
            n_steps=len(records),
        )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _pick_aggregate(result: PRMResult, aggregator: str) -> float:
    if aggregator == "min":
        return result.aggregate_min
    if aggregator == "product":
        return result.aggregate_product
    if aggregator == "mean":
        return result.aggregate_mean
    raise ValueError(
        f"aggregator must be one of {_VALID_AGGREGATORS}, got {aggregator!r}"
    )


def compare_prm(
    prm_a: PRMResult,
    prm_b: PRMResult,
    aggregator: str = DEFAULT_AGGREGATOR,
) -> float:
    """Return ``aggregate_b - aggregate_a`` under ``aggregator``.

    Positive means B beat A.
    """

    if not isinstance(prm_a, PRMResult) or not isinstance(prm_b, PRMResult):
        raise TypeError("compare_prm requires two PRMResult instances")
    if aggregator not in _VALID_AGGREGATORS:
        raise ValueError(
            f"aggregator must be one of {_VALID_AGGREGATORS}, "
            f"got {aggregator!r}"
        )
    return _pick_aggregate(prm_b, aggregator) - _pick_aggregate(
        prm_a, aggregator
    )


def per_step_delta_distribution(
    prm_a: PRMResult, prm_b: PRMResult
) -> Tuple[float, ...]:
    """Step-aligned delta vector ``score_b[i] - score_a[i]``.

    Mismatched lengths zero-pad the shorter side: missing steps are
    treated as score 0.0, so the delta on a padded slot equals the
    score from the longer side (positive if B is longer, negative if A).
    """

    if not isinstance(prm_a, PRMResult) or not isinstance(prm_b, PRMResult):
        raise TypeError(
            "per_step_delta_distribution requires two PRMResult instances"
        )
    n = max(prm_a.n_steps, prm_b.n_steps)
    a_scores = [s.score for s in prm_a.steps] + [0.0] * (n - prm_a.n_steps)
    b_scores = [s.score for s in prm_b.steps] + [0.0] * (n - prm_b.n_steps)
    return tuple(b_scores[i] - a_scores[i] for i in range(n))


__all__ = (
    "StepRecord",
    "PRMResult",
    "ProcessRewardJudge",
    "split_steps",
    "compare_prm",
    "per_step_delta_distribution",
    "STEP_BOUNDARY_PATTERNS",
    "DEFAULT_AGGREGATOR",
)
