"""judge.lab — pre-registered, brutally honest LLM evaluation.

Public OSS subset of the Rex-26 LLM eval framework.  The empirical
record that motivated this package:

- 9 falsifications shipped (5 routing redesigns, ICL gaming, SCM
  domain-confounding, APE Goodhart, and one stub-judge flat-line)
- ICL retrieval +0.457 stub-judge "win" exposed as Jaccard format
  gaming (cod_009 = 1.000 verbatim copy of expected_output)
- Cross-judge collapse: P(ICL beats baseline) Qwen 3B 0.77 -> Phi-4
  0.41 -> gpt-4o-mini 0.23 (CI strictly below 0.5)
- HumanEval execution-based n=20 (the antidote): codegen-tuned
  wrapper +65% pass@1 over raw prompt; terseness wrapper +20%

What this package contains
- pairwise.PairwiseJudge: Bradley-Terry MLE + bootstrap CI +
  position-consistency audit (Chiang ICML 2024 Chatbot Arena style)
- conformal.ConformalJudge: 90% coverage prediction bands on judge
  scores (Vovk 2005 / Angelopoulos & Bates 2021)
- hygiene.JudgeHygieneRunner: self-consistency + PIN audits +
  injection detection (Zheng et al. NeurIPS 2023 LLM-as-Judge survey)
- process_reward.ProcessRewardJudge: per-step PRM aggregator
  (Khalifa 2025 ThinkPRM-compatible)
- best_of_n.BestOfNReranker: judge-as-verifier reranker
- llm_judge.LLMAsJudge: K self-consistency scalar judge
- sandbox.run_humaneval_case: minimal sandboxed Python executor
  for execution-based eval

Use at your own risk.  This tool will expose your eval pipeline
weaknesses.
"""

from judge_lab.pairwise import (
    PairwiseJudge,
    PairwiseVerdict,
    PairwiseRecord,
    BradleyTerryReport,
    flip_verdict,
)
from judge_lab.conformal import (
    ConformalJudge,
    ConformalInterval,
    CalibrationPoint,
    OverlapVerdict,
    split_conformal_quantile,
)
from judge_lab.hygiene import (
    JudgeHygieneRunner,
    HygieneReport,
    detect_injection,
    score_variance,
    position_swap_inconsistency,
)
from judge_lab.process_reward import (
    ProcessRewardJudge,
    StepRecord,
    PRMResult,
    split_steps,
    compare_prm,
)
from judge_lab.best_of_n import (
    BestOfNReranker,
    CandidateResponse,
    BestOfNResult,
    aggregate_pass_at_k,
)
from judge_lab.llm_judge import (
    LLMAsJudge,
    LLMJudgeBreakdown,
    parse_score,
)
from judge_lab.sandbox import (
    ExecResult,
    run_python_test,
    run_humaneval_case,
)

__version__ = "0.1.0"
__all__ = [
    "PairwiseJudge", "PairwiseVerdict", "PairwiseRecord",
    "BradleyTerryReport", "flip_verdict",
    "ConformalJudge", "ConformalInterval", "CalibrationPoint",
    "OverlapVerdict", "split_conformal_quantile",
    "JudgeHygieneRunner", "HygieneReport", "detect_injection",
    "score_variance", "position_swap_inconsistency",
    "ProcessRewardJudge", "StepRecord", "PRMResult",
    "split_steps", "compare_prm",
    "BestOfNReranker", "CandidateResponse", "BestOfNResult",
    "aggregate_pass_at_k",
    "LLMAsJudge", "LLMJudgeBreakdown", "parse_score",
    "ExecResult", "run_python_test", "run_humaneval_case",
]
