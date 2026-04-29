"""Bradley-Terry pairwise judge with bootstrap CIs.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

This module implements Approach #28 of the Substrate-12 falsification
harness pre-registered in
``docs/r26/ULTRA_DEEP_APPROACHES_PART3_2026_04_28.md``: a pairwise
preference judge over ``(question, response_a, response_b)`` triples
with full position-swap auditing.  It returns a Bradley-Terry MLE for
B's win-rate against A, accompanied by a percentile bootstrap CI and
a position-bias diagnostic.

Method
------
For each input pair the underlying ``judge_callable`` is invoked twice:

1. ``(question, A, B)`` -> ``verdict_ab``
2. ``(question, B, A)`` -> ``verdict_ba``

A judge that is order-invariant should return semantically identical
verdicts: ``verdict_ab == -verdict_ba`` (the sign of the winner flips
with the swap).  Any other outcome is flagged as
``position_inconsistent``.  When the position-consistency rate falls
below the configured threshold the report is marked as
``"position_biased"`` regardless of the win-rate magnitude.

The Bradley-Terry estimate is the closed-form MLE for the 2-class
case: ``P(B beats A) = (wins_B + 0.5 * ties + alpha) / (n + 2*alpha)``
where ``alpha`` is a small Laplace prior (1.0) keeping the estimate
bounded away from {0, 1} for finite samples.

Citations
---------
- L.-W. Chiang, W.-L. Chiang, et al., "Chatbot Arena: An Open Platform
  for Evaluating LLMs by Human Preference," ICML 2024,
  arXiv:2403.04132.
- R. A. Bradley and M. E. Terry, "Rank Analysis of Incomplete Block
  Designs: I. The Method of Paired Comparisons," Biometrika 39
  (3/4), 1952.
- L. Zheng, W.-L. Chiang, Y. Sheng, et al., "Judging LLM-as-a-Judge
  with MT-Bench and Chatbot Arena," NeurIPS 2023, arXiv:2306.05685
  (PINs / position-bias audit).
- B. Efron and R. J. Tibshirani, "An Introduction to the Bootstrap,"
  Chapman & Hall, 1993 (percentile bootstrap CIs).

Design constraints
------------------
* Pure stdlib (``enum``, ``random``, ``math``, ``statistics``,
  ``dataclasses``, ``typing``).
* Deterministic given ``seed``.
* Hermetic: no I/O, no network, no LLM calls.  ``judge_callable`` is
  injected.
* Frozen dataclasses for records and report so callers can store and
  hash them.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

__all__ = [
    "PairwiseVerdict",
    "PairwiseRecord",
    "BradleyTerryReport",
    "PairwiseJudge",
    "flip_verdict",
]


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class PairwiseVerdict(enum.IntEnum):
    """Outcome of a single pairwise judgement.

    Values are signed so that a position swap can be checked via
    ``verdict_ab == -verdict_ba``: B winning in the original order
    corresponds to A winning in the swapped order, both encoded as
    opposite signs.

    Ordering: ``A_WINS < TIE < B_WINS``.
    """

    A_WINS = -1
    TIE = 0
    B_WINS = 1


def flip_verdict(v: PairwiseVerdict) -> PairwiseVerdict:
    """Return the verdict with its sign flipped (helper for swaps).

    ``A_WINS`` <-> ``B_WINS``; ``TIE`` is invariant.
    """
    if not isinstance(v, PairwiseVerdict):
        raise TypeError(
            f"flip_verdict expects PairwiseVerdict, got {type(v).__name__}"
        )
    if v is PairwiseVerdict.TIE:
        return PairwiseVerdict.TIE
    if v is PairwiseVerdict.A_WINS:
        return PairwiseVerdict.B_WINS
    return PairwiseVerdict.A_WINS


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseRecord:
    """A single pairwise judgement with its position-swap audit.

    ``verdict`` is the verdict in the original ``(A, B)`` order.
    ``swap_verdict`` is the verdict in the swapped ``(B, A)`` order
    expressed in the *original* coordinate system, i.e. already
    flipped back so that comparing to ``verdict`` yields a direct
    consistency check.

    ``position_consistent`` is computed from the two verdicts.
    """

    pair_id: str
    verdict: PairwiseVerdict
    swap_verdict: PairwiseVerdict
    position_consistent: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("pair_id must be a non-empty string")
        if not isinstance(self.verdict, PairwiseVerdict):
            raise TypeError("verdict must be a PairwiseVerdict")
        if not isinstance(self.swap_verdict, PairwiseVerdict):
            raise TypeError("swap_verdict must be a PairwiseVerdict")
        # ``swap_verdict`` is stored already in original coordinates,
        # so consistency is plain equality.
        object.__setattr__(
            self,
            "position_consistent",
            self.verdict == self.swap_verdict,
        )


@dataclass(frozen=True)
class BradleyTerryReport:
    """Aggregate report from a Bradley-Terry pairwise evaluation."""

    bt_rate_b: float
    bt_ci_low: float
    bt_ci_high: float
    position_consistency_rate: float
    n_pairs: int
    verdict: str
    failures: Tuple[str, ...]

    @property
    def b_wins_significantly(self) -> bool:
        """True iff CI lower bound is above 0.5 AND position-consistent.

        Position consistency must be at least 0.85 (the standard
        threshold from ``PairwiseJudge`` defaults).  This is a
        hard-coded floor on the property so callers do not silently
        accept biased judges.
        """
        return self.bt_ci_low > 0.50 and self.position_consistency_rate >= 0.85


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


# Type alias for the injected judge callable.
JudgeCallable = Callable[[str, str, str], PairwiseVerdict]


class PairwiseJudge:
    """Pairwise judge with Bradley-Terry MLE and bootstrap CIs.

    The judge is injected as a callable taking
    ``(question, response_a, response_b)`` and returning a
    :class:`PairwiseVerdict`.  The judge is invoked twice per pair
    (original and swapped order) so position bias can be detected.

    Parameters
    ----------
    judge_callable:
        Callable mapping ``(question, response_a, response_b)`` ->
        :class:`PairwiseVerdict`.
    ci_alpha:
        Significance level for the bootstrap CI; the CI spans the
        ``[alpha/2, 1 - alpha/2]`` percentiles.  Must be in ``(0, 1)``.
    n_bootstrap:
        Number of bootstrap resamples.  Must be at least 10.
    position_consistency_threshold:
        Minimum fraction of position-consistent pairs required for the
        verdict to be anything other than ``"position_biased"``.  Must
        be in ``[0, 1]``.
    seed:
        Seed for the bootstrap RNG (deterministic).
    """

    # Laplace smoothing constant for the closed-form BT MLE.
    _BT_PRIOR: float = 1.0

    def __init__(
        self,
        judge_callable: JudgeCallable,
        ci_alpha: float = 0.05,
        n_bootstrap: int = 1000,
        position_consistency_threshold: float = 0.85,
        seed: int = 0,
    ) -> None:
        if not callable(judge_callable):
            raise TypeError("judge_callable must be callable")
        if not isinstance(ci_alpha, (int, float)) or isinstance(
            ci_alpha, bool
        ):
            raise TypeError("ci_alpha must be a float")
        if not (0.0 < float(ci_alpha) < 1.0):
            raise ValueError("ci_alpha must be in (0, 1)")
        if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool):
            raise TypeError("n_bootstrap must be an int")
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be >= 10")
        if not isinstance(
            position_consistency_threshold, (int, float)
        ) or isinstance(position_consistency_threshold, bool):
            raise TypeError("position_consistency_threshold must be a float")
        if not (0.0 <= float(position_consistency_threshold) <= 1.0):
            raise ValueError(
                "position_consistency_threshold must be in [0, 1]"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an int")

        self._judge = judge_callable
        self._ci_alpha = float(ci_alpha)
        self._n_bootstrap = int(n_bootstrap)
        self._pos_threshold = float(position_consistency_threshold)
        self._seed = int(seed)

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------

    def bradley_terry_estimate(
        self, records: Sequence[PairwiseRecord]
    ) -> float:
        """Closed-form 2-class Bradley-Terry MLE with Laplace prior.

        Each record contributes its *original-order* verdict.  Ties
        count as half-wins for B (and half for A).  An additive
        Laplace prior of ``_BT_PRIOR`` on each side keeps the estimate
        away from the boundary on small samples.
        """
        if not records:
            return 0.5
        wins_b = 0.0
        for r in records:
            if r.verdict is PairwiseVerdict.B_WINS:
                wins_b += 1.0
            elif r.verdict is PairwiseVerdict.TIE:
                wins_b += 0.5
            # A_WINS contributes 0
        n = len(records)
        prior = self._BT_PRIOR
        return (wins_b + prior) / (n + 2.0 * prior)

    def bootstrap_ci(
        self,
        records: Sequence[PairwiseRecord],
        alpha: float,
        n_boot: int,
        rng: random.Random,
    ) -> Tuple[float, float]:
        """Percentile bootstrap CI on the BT estimate.

        Resamples ``records`` with replacement ``n_boot`` times and
        returns ``(low, high)`` at the
        ``[alpha/2, 1 - alpha/2]`` percentiles.
        """
        if not records:
            return (0.0, 1.0)
        n = len(records)
        estimates = []
        for _ in range(n_boot):
            sample = [records[rng.randrange(n)] for _ in range(n)]
            estimates.append(self.bradley_terry_estimate(sample))
        estimates.sort()
        lo_idx = int(math.floor((alpha / 2.0) * n_boot))
        hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
        lo_idx = max(0, min(lo_idx, n_boot - 1))
        hi_idx = max(0, min(hi_idx, n_boot - 1))
        return (float(estimates[lo_idx]), float(estimates[hi_idx]))

    # ------------------------------------------------------------------
    # Public eval API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pairs: Sequence[Tuple[str, str, str]],
    ) -> BradleyTerryReport:
        """Evaluate a sequence of ``(question, response_a, response_b)``.

        For each pair the judge is invoked twice -- once in the
        original order and once with the responses swapped -- to
        detect position bias.  The Bradley-Terry MLE on B's win rate
        is returned together with a percentile bootstrap CI.
        """
        if pairs is None:
            raise TypeError("pairs must not be None")
        records = []
        failures = []
        for idx, item in enumerate(pairs):
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or not all(isinstance(x, str) for x in item)
            ):
                raise TypeError(
                    f"pairs[{idx}] must be a 3-tuple of strings"
                )
            question, resp_a, resp_b = item
            v_ab = self._judge(question, resp_a, resp_b)
            v_ba_raw = self._judge(question, resp_b, resp_a)
            if not isinstance(v_ab, PairwiseVerdict) or not isinstance(
                v_ba_raw, PairwiseVerdict
            ):
                raise TypeError(
                    f"judge_callable must return PairwiseVerdict at index "
                    f"{idx}"
                )
            # Translate the swapped-order verdict back into the
            # original (A, B) coordinate system: a swap-order win for
            # the *first* response (i.e. B in original) maps back to
            # B_WINS in original coordinates.
            v_ba = flip_verdict(v_ba_raw)
            pair_id = f"pair-{idx:04d}"
            records.append(
                PairwiseRecord(
                    pair_id=pair_id,
                    verdict=v_ab,
                    swap_verdict=v_ba,
                )
            )

        n_pairs = len(records)
        if n_pairs == 0:
            failures.append("no_pairs")
            return BradleyTerryReport(
                bt_rate_b=0.5,
                bt_ci_low=0.0,
                bt_ci_high=1.0,
                position_consistency_rate=1.0,
                n_pairs=0,
                verdict="inconclusive",
                failures=tuple(failures),
            )

        rng = random.Random(self._seed)
        bt_rate = self.bradley_terry_estimate(records)
        ci_low, ci_high = self.bootstrap_ci(
            records, self._ci_alpha, self._n_bootstrap, rng
        )
        n_consistent = sum(1 for r in records if r.position_consistent)
        consistency = n_consistent / n_pairs

        if consistency < self._pos_threshold:
            failures.append("position_biased")
            verdict_str = "position_biased"
        elif ci_low > 0.5:
            verdict_str = "B_wins"
        elif ci_high < 0.5:
            verdict_str = "A_wins"
        else:
            verdict_str = "inconclusive"

        return BradleyTerryReport(
            bt_rate_b=float(bt_rate),
            bt_ci_low=float(ci_low),
            bt_ci_high=float(ci_high),
            position_consistency_rate=float(consistency),
            n_pairs=n_pairs,
            verdict=verdict_str,
            failures=tuple(failures),
        )
