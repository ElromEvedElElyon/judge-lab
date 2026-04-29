"""Best-of-N reranker (Approach #38) for the Substrate-12 falsification harness.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.  Do not
publish.  Do not export.  Repository is private.

Approach
--------
This is the **compute-as-quality lever**.  Given a base provider and a verifier
(reward model / judge) callable, sample ``N`` candidate completions at
temperature ``T`` and pick the candidate that maximises the verifier score.
The chosen response is returned through the same one-shot ``complete(prompt)``
surface used by the rest of the falsification harness, so this reranker is a
drop-in replacement wherever a ``ProviderShim`` is expected.

The mechanism is orthogonal to every encoder-side approach already in the
substrate12 benchmark folder: it amplifies whichever provider is wired in,
at the cost of ``N`` forward passes per inference and ``N`` verifier calls.
The verifier in production is typically the existing ``judge_stub`` (or a
process-reward callable); the reranker treats it as a black-box scalar reward.

Determinism
-----------
The ``i``-th candidate is drawn at seed ``base_seed + i`` so the entire
candidate set is reproducible from a single seed.  Tie-breaking in the
``argmax`` is deterministic: when two candidates share the highest verifier
score, the lower seed wins.  This guarantees that a fixed seed yields a fixed
chosen response, which is what the falsification harness relies on.

References
----------
* Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems"
  (GSM8K), arXiv:2110.14168.  Original outcome-supervised verifier.
* Lightman et al. 2023, "Let's Verify Step by Step" (PRM800K),
  arXiv:2305.20050.  Process-reward model variant.
* Brown et al. 2024, "Large Language Monkeys: Scaling Inference Compute with
  Repeated Sampling", arXiv:2407.21787.  pass@k scaling with N samples.

Constraints
-----------
* Pure stdlib (``dataclasses``, ``typing``, ``math``).
* No numpy, no network, no I/O.
* All inputs (provider, verifier) are injected callables; tests stub them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

__all__ = [
    "CandidateResponse",
    "BestOfNResult",
    "BestOfNReranker",
    "aggregate_pass_at_k",
]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResponse:
    """One sampled candidate together with its verifier score.

    Frozen so the harness can store these in tuples / sets without worrying
    about accidental mutation.  Ordering is not implemented on purpose:
    callers that need to sort should sort by ``verifier_score`` explicitly,
    which makes the tie-breaking rule obvious at the call-site.
    """

    text: str
    seed: int
    verifier_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("CandidateResponse.text must be a string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("CandidateResponse.seed must be int")
        if isinstance(self.verifier_score, bool) or not isinstance(
            self.verifier_score, (int, float)
        ):
            raise TypeError("CandidateResponse.verifier_score must be numeric")
        score = float(self.verifier_score)
        if math.isnan(score) or math.isinf(score):
            raise ValueError("CandidateResponse.verifier_score must be finite")


@dataclass(frozen=True)
class BestOfNResult:
    """Full record of one Best-of-N rerank invocation.

    ``runners_up`` is a tuple sorted descending by ``verifier_score`` (with
    lower seed as tiebreaker), so the first element is the strongest non-chosen
    candidate.  ``verifier_total_calls`` is exactly ``n_samples`` for a single
    invocation; ``aggregate_pass_at_k`` sums it across multiple results.
    """

    chosen: CandidateResponse
    runners_up: Tuple[CandidateResponse, ...]
    n_samples: int
    verifier_total_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.chosen, CandidateResponse):
            raise TypeError(
                "BestOfNResult.chosen must be a CandidateResponse"
            )
        if not isinstance(self.runners_up, tuple):
            raise TypeError("BestOfNResult.runners_up must be a tuple")
        for i, ru in enumerate(self.runners_up):
            if not isinstance(ru, CandidateResponse):
                raise TypeError(
                    f"BestOfNResult.runners_up[{i}] must be a "
                    "CandidateResponse"
                )
        if not isinstance(self.n_samples, int) or isinstance(
            self.n_samples, bool
        ):
            raise TypeError("BestOfNResult.n_samples must be int")
        if self.n_samples < 1:
            raise ValueError("BestOfNResult.n_samples must be >= 1")
        if not isinstance(self.verifier_total_calls, int) or isinstance(
            self.verifier_total_calls, bool
        ):
            raise TypeError(
                "BestOfNResult.verifier_total_calls must be int"
            )
        if self.verifier_total_calls < 0:
            raise ValueError(
                "BestOfNResult.verifier_total_calls must be non-negative"
            )
        # Internal invariant: chosen + runners_up should equal n_samples.
        if 1 + len(self.runners_up) != self.n_samples:
            raise ValueError(
                f"BestOfNResult invariant violated: 1 + len(runners_up)="
                f"{1 + len(self.runners_up)} but n_samples="
                f"{self.n_samples}"
            )


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


# Type aliases for clarity.  ``ProviderCallable`` is *not* the harness's
# ``ProviderShim`` Protocol; it is the lower-level seeded-sampling callable
# this reranker wraps.  The reranker itself implements ProviderShim through
# the ``complete(prompt)`` method.
ProviderCallable = Callable[[str, float, int], str]
VerifierCallable = Callable[[str, str], float]


class BestOfNReranker:
    """Best-of-N reranker that wraps a base provider with a verifier.

    Conforms to the ``ProviderShim`` Protocol declared in
    ``rex26_llm.substrate12.benchmark.harness``: exposes a ``name`` attribute
    and a ``complete(prompt: str) -> str`` method, so the falsification
    harness can use it anywhere a provider is expected.

    Cost
    ----
    Each ``complete`` call invokes ``provider_callable`` exactly ``n_samples``
    times and ``verifier_callable`` exactly ``n_samples`` times.  This is
    intentional: Best-of-N is a compute-as-quality trade.
    """

    name: str

    def __init__(
        self,
        provider_callable: ProviderCallable,
        verifier_callable: VerifierCallable,
        n_samples: int = 8,
        temperature: float = 0.7,
        seed: int = 0,
        name: str = "best_of_n_reranker",
    ) -> None:
        if not callable(provider_callable):
            raise TypeError(
                "provider_callable must be callable (prompt, temperature, "
                "seed) -> str"
            )
        if not callable(verifier_callable):
            raise TypeError(
                "verifier_callable must be callable (prompt, response) -> "
                "float"
            )
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise TypeError("n_samples must be int")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if isinstance(temperature, bool) or not isinstance(
            temperature, (int, float)
        ):
            raise TypeError("temperature must be numeric")
        temperature_f = float(temperature)
        if math.isnan(temperature_f) or math.isinf(temperature_f):
            raise ValueError("temperature must be finite")
        if temperature_f < 0.0:
            raise ValueError("temperature must be non-negative")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be int")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        self._provider = provider_callable
        self._verifier = verifier_callable
        self._n_samples = n_samples
        self._temperature = temperature_f
        self._seed = seed
        self.name = name

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def seed(self) -> int:
        return self._seed

    def complete(self, prompt: str) -> str:
        """Return the chosen best-of-N response as a plain string.

        This is the ProviderShim-compatible surface.
        """
        return self.complete_with_details(prompt).chosen.text

    def complete_with_details(self, prompt: str) -> BestOfNResult:
        """Return the full BestOfNResult with chosen + runners_up."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        candidates: List[CandidateResponse] = []
        for i in range(self._n_samples):
            sample_seed = self._seed + i
            text = self._provider(prompt, self._temperature, sample_seed)
            if not isinstance(text, str):
                raise TypeError(
                    f"provider_callable returned {type(text).__name__} for "
                    f"seed={sample_seed}, expected str"
                )
            score = self._verifier(prompt, text)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(
                    "verifier_callable must return a numeric score"
                )
            score_f = float(score)
            if math.isnan(score_f) or math.isinf(score_f):
                raise ValueError(
                    "verifier_callable returned a non-finite score"
                )
            candidates.append(
                CandidateResponse(
                    text=text,
                    seed=sample_seed,
                    verifier_score=score_f,
                )
            )

        # Sort descending by score, ascending by seed for tie-breaks.
        # Negating the score flips it to descending while keeping seed asc.
        ranked = sorted(
            candidates, key=lambda c: (-c.verifier_score, c.seed)
        )
        chosen = ranked[0]
        runners_up = tuple(ranked[1:])
        return BestOfNResult(
            chosen=chosen,
            runners_up=runners_up,
            n_samples=self._n_samples,
            verifier_total_calls=self._n_samples,
        )


# ---------------------------------------------------------------------------
# pass@k aggregation
# ---------------------------------------------------------------------------


def aggregate_pass_at_k(
    results: Iterable[BestOfNResult],
    correct_predicate: Callable[[str], bool],
) -> Dict[str, Any]:
    """Aggregate pass@1 / pass@N over a batch of Best-of-N results.

    Parameters
    ----------
    results
        An iterable of ``BestOfNResult`` records (one per task / prompt).
    correct_predicate
        A callable ``(text: str) -> bool`` that decides whether a response
        is "correct".  It is applied to (a) ``chosen.text`` for ``pass@1``
        and (b) every candidate text for ``pass@N``.

    Returns
    -------
    dict with keys:
        ``pass_at_1`` — fraction of results whose chosen candidate is
            correct.  This is what you actually ship.
        ``pass_at_n`` — fraction of results where *any* of the N candidates
            was correct.  Upper bound on what a perfect verifier could buy.
        ``n_samples`` — N from the first result (assumed homogeneous).  All
            results must agree on N or ``ValueError`` is raised.
        ``n_total`` — number of results aggregated.
        ``frac_helped_by_n`` — fraction of cases where ``pass@N`` was True
            but ``pass@1`` was False.  This is the headroom Best-of-N could
            still buy with a better verifier; if it is large, the verifier
            is the bottleneck.

    Raises
    ------
    ValueError
        If ``results`` is empty (no sensible default; pass@k of nothing is
        undefined and silently returning zeros would mask wiring bugs).
    TypeError
        If ``correct_predicate`` is not callable, or if any result is not a
        ``BestOfNResult``.
    """
    if not callable(correct_predicate):
        raise TypeError("correct_predicate must be callable (str) -> bool")

    materialised = list(results)
    if not materialised:
        raise ValueError(
            "aggregate_pass_at_k requires at least one BestOfNResult; "
            "got an empty iterable"
        )

    n_samples_ref: int | None = None
    pass_at_1_count = 0
    pass_at_n_count = 0
    helped_by_n_count = 0

    for i, r in enumerate(materialised):
        if not isinstance(r, BestOfNResult):
            raise TypeError(
                f"results[{i}] must be a BestOfNResult, got "
                f"{type(r).__name__}"
            )
        if n_samples_ref is None:
            n_samples_ref = r.n_samples
        elif r.n_samples != n_samples_ref:
            raise ValueError(
                f"results disagree on n_samples: results[0].n_samples="
                f"{n_samples_ref} vs results[{i}].n_samples={r.n_samples}"
            )

        chosen_correct = bool(correct_predicate(r.chosen.text))
        any_correct = chosen_correct or any(
            bool(correct_predicate(c.text)) for c in r.runners_up
        )
        if chosen_correct:
            pass_at_1_count += 1
        if any_correct:
            pass_at_n_count += 1
        if any_correct and not chosen_correct:
            helped_by_n_count += 1

    n_total = len(materialised)
    return {
        "pass_at_1": pass_at_1_count / n_total,
        "pass_at_n": pass_at_n_count / n_total,
        "n_samples": int(n_samples_ref) if n_samples_ref is not None else 0,
        "n_total": n_total,
        "frac_helped_by_n": helped_by_n_count / n_total,
    }
