"""Split-conformal prediction bands on judge scores (approach #30).

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

This module implements falsification approach #30 (pre-registered in
``docs/r26/ULTRA_DEEP_APPROACHES_PART3_2026_04_28.md``): wrap any
LLM-judge scalar score in a split-conformal prediction band so the
SUBSTRATE-12 benchmark harness can declare a substrate "win" only when
the conformal interval on judge(B) does *not* overlap the conformal
interval on judge(A).

Why this is orthogonal to the prior 27+ falsification approaches in this
benchmark suite: every prior approach reduces to a point-estimate delta
and a naive frequentist t-test (or its bootstrap equivalent).  None of
them certifies the *coverage* of their interval.  Split-conformal
prediction (Vovk et al. 2005; Angelopoulos & Bates 2021) is the only
mechanism in the 42 documented approaches that produces a
**distribution-free** marginal coverage guarantee given only the
exchangeability of calibration and test residuals.  No parametric
assumption on the judge, on the rubric, or on their joint distribution
is required.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in
a Random World*. Springer.

Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification."
arXiv:2107.07511.

Quach, V., Fisch, A., Schuster, T., Yala, A., Sohn, J. H., Jaakkola,
T. S., & Barzilay, R. (2024). "Conformal Language Modeling."
arXiv:2310.01208.

Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized Quantile
Regression." NeurIPS 2019, arXiv:1905.03222.

----------------------------------------------------------------------
Brutal-honest scope of the guarantee
----------------------------------------------------------------------

This implementation provides **marginal** coverage only.  Marginal
coverage means

    P( rubric \\in [judge - q_hat, judge + q_hat] )  >=  1 - alpha

where the probability is over the *joint* random draw of calibration
set and test point.  It does **NOT** mean conditional validity, i.e.

    P( rubric \\in [judge - q_hat, judge + q_hat] | X = x )  >=  1 - alpha

for every individual case ``x``.  In practice the interval can
systematically under-cover hard cases (e.g. long-context, adversarial,
out-of-distribution prompts) and over-cover easy ones, while still
hitting the marginal target on average.  This is a known and inherent
limitation of split-conformal; addressing it requires conditional
conformal methods such as conformalized quantile regression (Romano et
al. 2019) or group-conditional / Mondrian variants, which are out of
scope for this approach.

A second non-obvious caveat: this module assumes the judge is
*post-hoc fixed*.  If you re-tune the judge on the calibration set
after observing the rubric scores, the exchangeability assumption is
broken and the coverage guarantee is void.  Use a held-out calibration
split that the judge has never seen.

A third caveat: clipping the interval to ``[0, max_score]`` is
optically nice but technically only preserves marginal coverage when
the rubric also lives in ``[0, max_score]`` (which it does here,
because the SUBSTRATE-12 5-dim rubric is bounded).

Pure stdlib only: ``math``, ``dataclasses``, ``typing``.  No NumPy,
SciPy or scikit-learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "MIN_CALIBRATION_POINTS",
    "CalibrationPoint",
    "ConformalInterval",
    "OverlapVerdict",
    "ConformalJudge",
    "split_conformal_quantile",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Minimum number of calibration pairs.  Below this the (1-alpha)(n+1)/n
#: quantile becomes degenerate (the index can fall outside ``[0, n)`` for
#: typical ``alpha``), and any quantile estimate from a tiny sample is
#: too noisy to defend.  30 is the conventional rule-of-thumb floor.
MIN_CALIBRATION_POINTS: int = 30


# ---------------------------------------------------------------------------
# Helper: finite-float guard
# ---------------------------------------------------------------------------


def _finite(name: str, value: float) -> float:
    """Coerce ``value`` to ``float`` and reject NaN / +-inf."""
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return fv


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    """A single (judge, rubric) calibration pair.

    Both scores must be finite, non-negative, and non-NaN.  We do **not**
    enforce an upper bound here because :class:`ConformalJudge` accepts a
    user-configurable ``max_score``; the upper-bound check happens there.
    """

    judge_score: float
    rubric_score: float

    def __post_init__(self) -> None:  # noqa: D401 -- validation
        j = _finite("judge_score", self.judge_score)
        r = _finite("rubric_score", self.rubric_score)
        if j < 0.0:
            raise ValueError(f"judge_score must be >= 0, got {j}")
        if r < 0.0:
            raise ValueError(f"rubric_score must be >= 0, got {r}")
        # frozen dataclass: re-bind via object.__setattr__ to ensure
        # downstream code sees plain floats even if user passed e.g. int.
        object.__setattr__(self, "judge_score", j)
        object.__setattr__(self, "rubric_score", r)


@dataclass(frozen=True)
class ConformalInterval:
    """A conformal prediction interval centred on a judge score."""

    lo: float
    hi: float
    point: float

    def __post_init__(self) -> None:  # noqa: D401 -- validation
        lo = _finite("lo", self.lo)
        hi = _finite("hi", self.hi)
        pt = _finite("point", self.point)
        if lo > hi:
            raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
        if not (lo <= pt <= hi):
            raise ValueError(
                f"point ({pt}) must lie in [lo, hi] = [{lo}, {hi}]"
            )
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "point", pt)

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class OverlapVerdict:
    """Verdict on whether two conformal intervals overlap.

    Touching intervals (``a.hi == b.lo``) are a deliberate **third**
    state: ``overlap=False`` AND ``b_strictly_higher=False`` AND
    ``a_strictly_higher=False`` -- "separated by zero gap, neither side
    can claim a strict win".  This is the falsification-friendly
    convention: a substrate "wins" only when its lower bound is
    *strictly* above the other's upper bound, and intervals are
    "in conflict" only when they share a region of *positive* measure.

    The two flags ``overlap`` and ``b_strictly_higher`` are therefore
    not exact mutual negations on the boundary; the user must inspect
    both.
    """

    interval_a: ConformalInterval
    interval_b: ConformalInterval
    overlap: bool
    b_strictly_higher: bool
    a_strictly_higher: bool

    @classmethod
    def from_intervals(
        cls,
        interval_a: ConformalInterval,
        interval_b: ConformalInterval,
    ) -> "OverlapVerdict":
        a, b = interval_a, interval_b
        # ``overlap`` is True only when the intersection has positive
        # measure: touching at a single point is NOT overlap.
        overlap = (a.hi > b.lo) and (b.hi > a.lo)
        # Strict-win flags use strict ``>`` so touching does NOT trigger.
        b_strictly_higher = b.lo > a.hi
        a_strictly_higher = a.lo > b.hi
        return cls(
            interval_a=a,
            interval_b=b,
            overlap=overlap,
            b_strictly_higher=b_strictly_higher,
            a_strictly_higher=a_strictly_higher,
        )


# ---------------------------------------------------------------------------
# Pure-stdlib quantile primitive
# ---------------------------------------------------------------------------


def split_conformal_quantile(
    nonconformities: Sequence[float],
    alpha: float,
) -> float:
    """Return the split-conformal ``(1 - alpha)(n + 1)/n`` quantile.

    Per Angelopoulos & Bates (2021), eq. (3.2), given ``n`` non-conformity
    scores ``s_1, ..., s_n`` from the calibration set, the conformal
    prediction band uses

        q_hat = the ceil((1 - alpha)(n + 1)) / n -th sample quantile

    of the sorted ``s_i``.  Equivalently, with 0-indexed sorted scores
    ``s_(0) <= s_(1) <= ... <= s_(n-1)``, ``q_hat = s_(k - 1)`` where
    ``k = ceil((1 - alpha)(n + 1))``.  When ``k > n`` (which happens for
    very small ``n`` and very small ``alpha``) we return the maximum
    non-conformity, which is the most conservative finite-sample value.

    Parameters
    ----------
    nonconformities :
        Iterable of non-negative finite floats, ``len >= 1``.
    alpha :
        Miscoverage rate, in the open interval ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``nonconformities`` is empty, contains a non-finite value,
        or if ``alpha`` is outside ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    n = len(nonconformities)
    if n < 1:
        raise ValueError("nonconformities must be non-empty")

    sorted_s = sorted(_finite("nonconformity", s) for s in nonconformities)
    # Index k via ceil((1 - alpha)(n + 1)); cap at n (use the max).
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k < 1:
        k = 1
    if k > n:
        k = n
    return sorted_s[k - 1]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConformalJudge:
    """Wrap a scalar LLM judge in a split-conformal prediction band.

    Usage
    -----

    >>> cj = ConformalJudge(alpha=0.1, max_score=10.0)
    >>> cj.calibrate([
    ...     CalibrationPoint(judge_score=j, rubric_score=r)
    ...     for j, r in zip(judges, rubrics)
    ... ])
    >>> interval = cj.predict(judge_score=7.4)
    >>> verdict = cj.compare(judge_a=6.1, judge_b=8.2)
    >>> verdict.b_strictly_higher  # True iff substrate B beats A
    """

    def __init__(self, alpha: float = 0.1, max_score: float = 10.0) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if not (float(max_score) > 0.0) or math.isinf(float(max_score)):
            raise ValueError(
                f"max_score must be finite and > 0, got {max_score!r}"
            )
        self._alpha: float = float(alpha)
        self._max_score: float = float(max_score)
        self._q_hat: float | None = None
        self._n_calibration: int = 0

    # ----------------------------- properties -----------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def max_score(self) -> float:
        return self._max_score

    @property
    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError(
                "ConformalJudge is not calibrated; call calibrate() first."
            )
        return self._q_hat

    @property
    def n_calibration(self) -> int:
        return self._n_calibration

    # ----------------------------- calibration ----------------------------

    def calibrate(self, points: Sequence[CalibrationPoint]) -> None:
        """Fit ``q_hat`` from ``points`` via split-conformal calibration.

        We compute non-conformities as absolute residuals
        ``|judge - rubric|``.  An asymmetric (signed) variant is possible
        but produces two-sided intervals of unequal width, which we do
        not need for the simple ``compare(A, B)`` use case.

        Raises
        ------
        ValueError
            If ``len(points) < MIN_CALIBRATION_POINTS``, if any pair has
            judge or rubric outside ``[0, max_score]``, or if any pair
            contains non-finite values.
        """
        n = len(points)
        if n < MIN_CALIBRATION_POINTS:
            raise ValueError(
                f"Need at least {MIN_CALIBRATION_POINTS} calibration "
                f"points, got {n}."
            )

        scores: list[float] = []
        for p in points:
            if p.judge_score > self._max_score:
                raise ValueError(
                    f"judge_score {p.judge_score} exceeds max_score "
                    f"{self._max_score}"
                )
            if p.rubric_score > self._max_score:
                raise ValueError(
                    f"rubric_score {p.rubric_score} exceeds max_score "
                    f"{self._max_score}"
                )
            scores.append(abs(p.judge_score - p.rubric_score))

        self._q_hat = split_conformal_quantile(scores, self._alpha)
        self._n_calibration = n

    # ----------------------------- prediction -----------------------------

    def predict(self, judge_score: float) -> ConformalInterval:
        """Return the conformal interval centred on ``judge_score``.

        The interval is clipped to ``[0, max_score]``.  Clipping
        preserves marginal coverage given the (already enforced)
        constraint that the rubric lives in the same range.
        """
        if self._q_hat is None:
            raise RuntimeError(
                "ConformalJudge is not calibrated; call calibrate() first."
            )
        j = _finite("judge_score", judge_score)
        if j < 0.0 or j > self._max_score:
            raise ValueError(
                f"judge_score {j} outside [0, {self._max_score}]"
            )
        q = self._q_hat
        lo = max(0.0, j - q)
        hi = min(self._max_score, j + q)
        # ``j`` is guaranteed in [lo, hi] because lo <= j <= hi
        # holds by construction (clipping pulls toward j, never past).
        return ConformalInterval(lo=lo, hi=hi, point=j)

    # ------------------------------- compare ------------------------------

    def compare(self, judge_a: float, judge_b: float) -> OverlapVerdict:
        """Predict intervals for two judge scores and return the verdict.

        ``OverlapVerdict.b_strictly_higher`` is the canonical falsifier
        for a SUBSTRATE-12 "B beats A" claim: the harness should accept
        the win only when ``b_strictly_higher`` is True.
        """
        ia = self.predict(judge_a)
        ib = self.predict(judge_b)
        return OverlapVerdict.from_intervals(ia, ib)
