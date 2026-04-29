# judge.lab

> **Your LLM judge is being gamed. We prove it.**

Pre-registered, brutally honest LLM evaluation. Bradley-Terry pairwise audits, conformal abstention bands, judge bias detection, and execution-based eval. Free, open-source, **use at your own risk** — this tool will expose your eval pipeline weaknesses.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Headline Finding

We thought we had a SIX-SIGMA win for in-context learning retrieval (+0.457 stub-judge on Qwen 3B local, codegen Δ=+1.212 Bonferroni-significant). Then we measured the Jaccard token overlap between ICL outputs and `expected_output`:

```
mean Jaccard zero-shot   = 0.155
mean Jaccard ICL-5shot   = 0.558      <-- +0.403 (t=+7.83, p<1e-9)
cod_009 ICL Jaccard      = 1.000      <-- VERBATIM copy of expected_output
```

The model wasn't reasoning. It was **copying**. And `judge_stub.correctness` is literally Jaccard token overlap with `expected_output`, so it rewarded the gaming directly.

We then audited the same ICL outputs with three pairwise BT judges:

| Judge | P(ICL beats baseline) | 95% CI | Position consistency |
|---|---:|---|---:|
| Qwen 3B (weak local) | 0.7727 | [0.66, 0.89] | 30% (biased) |
| Phi-4 (strong RLHF) | 0.4091 | [0.25, 0.55] | 60% |
| **gpt-4o-mini** | **0.2273** | **[0.09, 0.41]** | **40%** |

CI strictly below 0.5. Under stronger pairwise judging, ICL is **inferior** to zero-shot. The "win" was format-matching, not capability.

Then we ran the antidote — execution-based HumanEval (pass@1 cannot be rubric-gamed):

| Condition | pass@1 (n=20, Qwen 3B) | Δ |
|---|---:|---:|
| baseline (raw HumanEval prompt) | 0/20 = 0% | — |
| terseness wrapper (`max 1 sentence`) | 4/20 = **20%** | +0.200 |
| **codegen wrapper (`ONLY Python code`)** | **13/20 = 65%** | **+0.650** |

Real, ungameable lift exists when the wrapper is right. Judge gauntlet escaped.

---

## Install

```bash
git clone https://github.com/ElromEvedElElyon/judge-lab
cd judge-lab
pip install -e .
```

Or once on PyPI:

```bash
pip install judge-lab  # planned
```

---

## Quick start

```python
from judge_lab import (
    PairwiseJudge, PairwiseVerdict,
    ConformalBands, empirical_coverage,
    LLMAsJudge,
    run_humaneval_case,
)

# 1. PAIRWISE BT AUDIT -- detect biased judges
def my_judge(question, response_a, response_b) -> PairwiseVerdict:
    """Inject your existing judge here."""
    ...

bt = PairwiseJudge(judge_callable=my_judge, n_bootstrap=400)
report = bt.evaluate(pairs)  # list of (question, A, B)
print(f"P(B beats A) = {report.bt_rate_b:.4f}")
print(f"95% CI       = [{report.bt_ci_low:.4f}, {report.bt_ci_high:.4f}]")
print(f"position_consistency = {report.position_consistency_rate:.2%}")

if report.position_consistency_rate < 0.85:
    print("WARNING: judge is biased -- verdict is suspect")


# 2. CONFORMAL ABSTENTION -- 90% coverage prediction bands
bands = ConformalBands(scores, alpha=0.1)
print(f"q_hat = {bands.q_hat:.4f}  (noise floor)")
# If your effect size < q_hat, ABSTAIN.


# 3. EXECUTION-BASED EVAL -- ungameable pass@1
import json
case = json.loads(open("examples/data/HumanEval.jsonl").readline())
result = run_humaneval_case(case, my_completion_string)
print(f"passed={result.passed}, elapsed={result.elapsed_s:.2f}s")


# 4. LLM-AS-JUDGE WITH SELF-CONSISTENCY
def provider(prompt, temperature, seed):
    """Your provider closure."""
    ...

judge = LLMAsJudge(provider=provider, k=5, temperature=0.7)
score = judge.score(output="...", expected="...", question="...")
```

See `examples/humaneval_ab.py` for a full three-condition AB harness on HumanEval.

---

## What's in the package

| Module | Purpose | Citation |
|---|---|---|
| `pairwise` | Bradley-Terry MLE + bootstrap CI + position-consistency | Chiang ICML 2024 (Chatbot Arena) |
| `conformal` | 90%-coverage prediction bands on judge scores | Vovk 2005, Angelopoulos & Bates 2021 |
| `hygiene` | Self-consistency + PIN audits + injection detection | Zheng NeurIPS 2023 LLM-as-Judge survey |
| `process_reward` | Per-step PRM aggregator | Khalifa 2025 ThinkPRM-compatible |
| `best_of_n` | Judge-as-verifier reranker | Lightman 2023 ORM/PRM |
| `llm_judge` | Scalar LLM-as-judge with K self-consistency | Wang 2022 self-consistency |
| `sandbox` | Minimal Python executor for HumanEval/MBPP/LiveCodeBench | OpenAI HumanEval 2021 |
| `tools/secrets/scan` | Pre-commit credential leak scanner (16 rules) | — |

---

## Why this exists

Most LLM eval pipelines run a stub-rubric or LLM-as-judge once and call it done. Across 9 honest falsifications, we've shown that this surfaces **format-matching, position bias, and domain confounding as if they were capability lift**. Every "win" we tested under stub-judge collapsed under stronger validation.

judge.lab is the X-ray. If your judge can be gamed, your eval is fiction.

### Pre-registered discipline

Every encoder, every approach, every claim ships with **kill thresholds before data collection**:

```python
# Pre-registered: ICL retrieval beats baseline iff:
#   delta >= +0.05 AND p < 0.10
# Otherwise -> falsified.
```

Five routing redesigns and one ICL gaming theory **falsified by their own pre-registered traps**. Empirical discipline by default.

---

## Falsifications shipped (selected)

1. Naive BV-prefix encoder — Δ=+0.001, p=0.987, n=100 (ZERO empirical)
2. ICL retrieval +0.457 — exposed as Jaccard format gaming via three-judge BT collapse
3. SCM "predictive" — domain-confounded (within-codegen gap NEGATIVE in 3 cross-model runs)
4. APE/OPRO n=3 — Goodhart sign-flip (+0.197 → -0.085 on n=22 hold-out)
5. K=10 self-consistency — does NOT fix systematically-biased local judge

---

## Disclaimer (OpenClaw-style)

> Open source under MIT. **Use at your own risk** — this tool will expose your eval pipeline weaknesses. The authors are not responsible for findings that contradict prior published benchmarks.

---

## Run the tests

```bash
pip install -e .[dev]
pytest                          # 17+ tests in tools/secrets/
pytest examples/                # if you add eval probes
```

---

## Credits

Built on the empirical scaffolding of **Sintex.AI / Rex-26** (private). Released as MIT-licensed subset for the LLM eval community.

- Repository: https://github.com/ElromEvedElElyon/judge-lab
- Homepage: https://sintex.ai

---

## Citation

If you use judge.lab in published research:

```bibtex
@software{judgelab2026,
  title  = {judge.lab: Pre-Registered Brutally Honest LLM Evaluation},
  author = {Sintex.AI},
  year   = {2026},
  url    = {https://github.com/ElromEvedElElyon/judge-lab}
}
```
