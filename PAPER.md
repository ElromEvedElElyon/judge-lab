# Empirical Falsification of Substrate-Prefix Encoders for LLM Routing under Cross-Judge Bradley-Terry and Execution-Based Evaluation

**Sintex.AI**
2026-04-29

---

## Abstract

We present a 48-hour empirical investigation into structured-prefix encoders for LLM workflow routing, conducted under brutal pre-registered falsification protocols. Across nine candidate approaches — including a 12-dimensional cognitive substrate prefix, in-context learning retrieval over belief-vector cosine, automatic prompt optimization (APE/OPRO), and structural causal model (SCM) sanity checks — we report eight empirical falsifications and one surviving claim. The primary mechanistic finding is that ICL retrieval of demonstration shots whose `expected_output` text is semantically similar to the target's `expected_output` produces measured Jaccard token overlaps approaching 1.000 (cod_009 = 1.000, verbatim echo), inflating stub-judge correctness scores by Δ=+0.40 (t=+7.83, p<10⁻⁹, n=20) without genuine capability lift. We confirm this as judge-stub gaming via three independent pairwise Bradley-Terry judges showing monotone-decreasing P(ICL beats baseline): Qwen 3B 0.77 → Phi-4 0.41 → gpt-4o-mini 0.23 (95% CI [0.09, 0.41] strictly below chance). Cross-model codegen-only analysis (Phi-4 +1.022, gpt-4o +0.864, Llama-3.3-70B +0.267, all p<0.05 stub-judge) reduces to a domain-confounded codegen-vs-other-domains marker; SCM-flagged subset gaps within codegen are NEGATIVE (-0.40 to -0.75) across all three models. Under execution-based evaluation on HumanEval (pass@1, ungameable), a domain-specialised codegen wrapper achieves 72% pass@1 (n=50, +0.720 absolute over raw-prompt baseline 0/50) on Qwen 3B local — a real but capability-adjacent lift driven by emitting-only-code discipline rather than reasoning improvement. We release [judge.lab](https://github.com/ElromEvedElElyon/judge-lab) and [skills.lab](https://github.com/ElromEvedElElyon/skills-lab) as MIT-licensed companions.

**Keywords:** LLM evaluation, Bradley-Terry pairwise, conformal prediction, ICL retrieval, judge bias, format gaming, execution-based eval, HumanEval, pre-registered falsification.

**Subject classification:** cs.CL (primary), cs.LG (secondary).

---

## 1. Introduction

LLM evaluation pipelines routinely report effect sizes of small encoders, prompt wrappers, and routing systems based on a single rubric judge or a self-consistency LLM-as-judge protocol [Zheng et al., NeurIPS 2023; Wang et al., 2022]. We present an empirical investigation into how reliable such reports are when the same encoders are subjected to (a) cross-judge Bradley-Terry pairwise validation [Chiang et al., ICML 2024], (b) direct mechanistic measurement of format-matching behaviour, and (c) execution-based eval that is binary and rubric-immune.

Our findings suggest a striking pattern: every encoder claiming a positive effect under a single rubric or a single-judge LLM-as-Judge protocol collapsed under cross-judge validation, with magnitude often disappearing or sign-flipping. The lone surviving claim — a domain-specialised codegen wrapper — survives execution-based eval (HumanEval pass@1) but is mechanistically driven by emitting-only-code discipline rather than reasoning improvement.

We release `judge.lab` and `skills.lab` as MIT-licensed open-source frameworks that operationalise the falsification discipline used in this work.

## 2. Methodology

### 2.1 Pre-registered kill thresholds

Every encoder, judge, and approach in this study ships with kill thresholds defined *before* data collection:

- For pairwise BT judges: P(B beats A) ≥ 0.55 AND position-consistency ≥ 0.70.
- For routing redesigns: Δ pass@1 ≥ +0.05 absolute over baseline.
- For ICL retrieval: Δ stub-judge ≥ +0.05 with p < 0.10.
- For SCM falsifier: gap (flagged_d − unflagged_d) ≥ +0.05.

Violation of any kill threshold triggers immediate abandonment of the approach, with the result reported regardless of investigator preference.

### 2.2 Corpus

The proprietary 100-case Substrate-12 corpus comprises 25 cases each across four domains (defensive_ai, solana_audit, codegen, explanation), with hand-coded BeliefVectors. We additionally use HumanEval [Chen et al., 2021] (164 problems) and MBPP [Austin et al., 2021] for execution-based eval validation.

### 2.3 Models

Generation: Qwen 2.5 3B Q4_K_M (Ollama local), Phi-4, gpt-4o, gpt-4o-mini, Llama-3.3-70B-Instruct (GitHub Models 50/24h tier). All runs T=0.0 deterministic.

Judging:
- `judge_stub`: Jaccard-token-overlap correctness + length-bonus helpfulness + refusal-detection defensive_alignment + rubric-completeness.
- `judge_pairwise` Bradley-Terry MLE on swap-paired prompts via three judge models.
- Execution sandbox: subprocess + 10s timeout + restricted env + `-I -S` Python isolation.

## 3. Falsification Record

### 3.1 Naive substrate-prefix encoder (LEGACY)

Pre-registration: legacy substrate-12-as-prefix encoder beats raw prompt iff Δ ≥ +0.05 with p < 0.10 on n=100 paired Qwen 3B T=0.

Result: Δ = +0.001, p = 0.987, n = 100. **FALSIFIED**.

### 3.2 encoder_d v2/v3/v4 (3 routing redesigns)

Three iterative redesigns of the workflow-decomposition encoder:
- v2 (multi_step default): Δ = -0.050, p = 0.49. **FALSIFIED**.
- v3 (loose 0.5/0.5 thresholds): Δ = -0.050 (numerically identical to v2). **FALSIFIED**.
- v4 (text-feature codegen detector): Δ = +0.008, below +0.02 floor. **FALSIFIED**.

### 3.3 APE/OPRO (Goodhart sign-flip)

APE/OPRO at n=3 codegen-eval discovered a hybrid template scoring +0.197 over best seed.
Hold-out validation on n=22: Δ = -0.085 (sign-flip). **FALSIFIED**.

### 3.4 In-context learning retrieval (the headline finding)

Pre-registration: ICL 5-shot retrieval over BeliefVector cosine beats zero-shot iff Δ ≥ +0.05 with p < 0.10.

Stub-judge result: Δ = +0.457, t = +6.91, p < 10⁻⁹ on n=100 Qwen 3B T=0. Codegen-only Δ = +1.212 Bonferroni-significant. *Apparent* SIX-SIGMA win.

**Mechanistic falsification (this paper's main contribution):** judge_stub.correctness is implemented as Jaccard token overlap between output and `expected_output`. ICL retrieval picks shots whose expected_output text is similar (because BVs are similar). Demonstration includes the answer template. Model echoes it.

```
mean Jaccard zero-shot = 0.155
mean Jaccard ICL-5shot = 0.558    (Δ = +0.403, t = +7.83, p < 10⁻⁹, n = 20)
cod_009 ICL Jaccard    = 1.000    (verbatim copy of expected_output)
```

Cross-judge Bradley-Terry confirmation on the same 20 codegen pairs:

| Judge | P(ICL > baseline) | 95% bootstrap CI | Position consistency |
|---|---:|---|---:|
| Qwen 3B BT | 0.7727 | [0.66, 0.89] | 30% (biased) |
| Phi-4 BT (paced) | 0.4091 | [0.25, 0.55] | 60% |
| **gpt-4o-mini BT** | **0.2273** | **[0.09, 0.41]** | **40%** |

Monotone-decreasing P as judge quality rises. Under gpt-4o-mini, CI strictly below 0.5 — ICL is *worse* than zero-shot. **FALSIFIED**.

### 3.5 SCM "predictive" claim (domain-confounded)

Pre-registration: SCM-flagged subset (kl > threshold) shows higher encoder_d benefit Δ ≥ +0.05.

Mixed-domain n=100 result: at calibrated thr=5.0, flagged Δ = +0.248 vs unflagged Δ = +0.053 (gap +0.195). **Apparent** confirmation.

Cross-model codegen-only re-analysis (within-codegen, no domain confound):
- Phi-4 codegen: gap = -0.665
- gpt-4o codegen: gap = -0.746
- Llama-3.3-70B codegen: gap = -0.397

All gaps NEGATIVE. The mixed-domain "win" reduced to a codegen-vs-other-domains marker; codegen has the largest encoder_d effect (+0.330 stratified vs +0.009 to +0.116 other domains), so SCM-flagging happened to correlate with codegen presence rather than capturing a deeper substrate signal. **FALSIFIED**.

### 3.6 K=10 self-consistency does not fix Qwen 3B judge

Per-case median swings ±6 between runs persist at K=10 (vs K=3). Within-case stdev decreases ~2.5× as expected, but between-run median bias persists. **NOT a falsification of K-self-consistency in general; a falsification of it as a fix for systematically biased local judges**.

### 3.7 SCM threshold portability

Calibrated thr=5.0 on hand-coded BV corpus does not transfer: HeuristicEncoder-derived BVs (codx corpus) all KL ∈ [5.48, 6.41]. At thr=5.0, 25/25 flagged → degenerate split. Percentile rule (top 33%) recovers calibration portability but gap on codx = +0.018 (FAIL kill threshold +0.05). **FALSIFIED**.

## 4. The Antidote: Execution-Based Eval

We ran HumanEval (pass@1) on Qwen 3B local with three conditions:
- baseline: raw HumanEval prompt
- encoder_d: "Answer concisely (max 1 sentence): {prompt}"
- codegen_v1: "Complete the following Python function. Output ONLY Python code, no markdown fences, no explanation, no docstring repeat: {prompt}"

Results:

| Condition | n=20 pass@1 | n=50 pass@1 | Δ at n=50 |
|---|---:|---:|---:|
| baseline | 0/20 = 0% | 0/50 = 0% | — |
| encoder_d (terseness) | 4/20 = 20% | 15/50 = 30% | +0.300 |
| **codegen_v1 (specialised)** | **13/20 = 65%** | **36/50 = 72%** | **+0.720** |

Both encoder_d and codegen_v1 IMPROVE at larger n. Findings hold and strengthen.

**Mechanism:** raw Qwen 3B emits "Certainly! Here's the implementation..." preamble + markdown fences → parser rejects 100% of completions. The terseness wrapper reduces preamble; the explicit codegen wrapper ("Output ONLY Python code, no markdown fences") matches the parser's exact requirements.

The lift is **emitting-only-code discipline**, not reasoning gain. But it is REAL and ungameable — pass@1 is binary code execution against test cases.

## 5. Implications for LLM Eval Practice

1. **Single-rubric or single-LLM-judge claims are unreliable.** Every "win" we tested under stub-judge or single-judge LLM-as-Judge collapsed under cross-judge BT validation.

2. **Position bias is no longer the dominant judge bias.** Per the April 2026 LLM-judge bias survey [arXiv:2604.23178], position bias is now negligible across modern judges; **style bias is dominant** (0.76-0.92 across all tested models). Style-strip pre-judging is recommended.

3. **Conformal abstention bands reveal effects below noise floors.** judge_stub q̂ = 0.685 in our setup. Any effect < 0.685 cannot be claimed significant. Rejected approaches that "won" under stub-judge but fell under conformal validation: ICL +0.457, SCM gap +0.195, encoder_d +0.118.

4. **Execution-based eval is the gold standard for codegen.** HumanEval pass@1 is binary. Specialised wrappers (codegen_v1) achieve massive gains (72%) over raw prompt (0%) on small open-weights models. The lift is mechanistic (parser-pass discipline), not capability-core, but it transfers across deployments.

5. **BeliefVector / cognitive substrate prefixes are empirically dead.** All five routing redesigns (legacy, v2, v3, v4 plus APE-discovered hybrid) and the ICL retrieval variant falsified. The thesis "structured prefix substrate replaces or augments prompts in deployed LLMs" is not supported by 9 falsifications.

## 6. Released Artifacts

- **judge.lab** (https://github.com/ElromEvedElElyon/judge-lab, MIT) — pre-registered LLM eval framework with BT pairwise audits, conformal abstention bands, judge bias detection, execution-based eval sandbox.
- **skills.lab** (https://github.com/ElromEvedElElyon/skills-lab, MIT) — Markdown skill catalogue with belief-vector routing, inspired by [google/skills](https://github.com/google/skills) + [Warp](https://github.com/warpdotdev/warp).

Both ship LAUNCH.md with HN post template, X thread, and this paper as supplementary material.

## 7. Conclusion

We documented eight pre-registered falsifications of substrate-prefix encoders in 48 hours and one surviving claim (specialised codegen wrapper, mechanistic, ungameable under HumanEval pass@1). The methodological lesson: every claim from a single-rubric or single-LLM-judge eval should be re-validated under (a) cross-judge BT, (b) direct mechanistic measurement (e.g., Jaccard with expected text), and (c) execution-based eval where applicable.

We invite the LLM eval community to use judge.lab to audit their own pipelines.

## Acknowledgements

Built on the empirical scaffolding of Sintex.AI / Rex-26 (private). Released as MIT-licensed subset for the LLM eval community. Inspired by the engineering disciplines of OpenAI HumanEval (2021), Chatbot Arena (Chiang ICML 2024), and the LLM-as-Judge survey (Zheng NeurIPS 2023).

## References

[1] Chen, M., et al. *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374, 2021.

[2] Chiang, W.-L., et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. ICML 2024, arXiv:2403.04132.

[3] Zheng, L., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023, arXiv:2306.05685.

[4] Wang, X., et al. *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models*. arXiv:2203.11171, 2022.

[5] Vovk, V., et al. *Algorithmic Learning in a Random World*. Springer, 2005. (Conformal prediction.)

[6] Angelopoulos, A. N. & Bates, S. *A Gentle Introduction to Conformal Prediction*. arXiv:2107.07511, 2021.

[7] Khalifa, M., et al. *Process Reward Models That Think*. arXiv:2504.16828, 2025.

[8] *Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines*. arXiv:2604.23178, April 2026.

[9] Austin, J., et al. *Program Synthesis with Large Language Models*. arXiv:2108.07732, 2021. (MBPP.)

[10] Sanchez, G., et al. *Stay on Topic with Classifier-Free Guidance*. arXiv:2306.17806, 2023.

[11] Jain, N., et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*. arXiv:2403.07974, 2024.

---

*Open-source, MIT-licensed, brutally honest. Use at your own risk — this tool will expose your eval pipeline weaknesses.*
