# Launch artifacts

Ready-to-post text for the v0.1.0 release of judge.lab + skills.lab.

---

## Hacker News post

**Title (≤80 chars):**

> Show HN: judge.lab — we tried to ship an LLM eval framework. Then we found cod_009 was a 1.000 verbatim copy.

**URL:** `https://github.com/ElromEvedElElyon/judge-lab`

**Body (Show HN):**

We've been building an LLM evaluation framework for 48 hours and ran 9 falsifications. The headline finding shocked us: an in-context learning encoder we thought won by SIX SIGMA on Qwen 3B local turned out to be format-matching, not capability lift.

Specifically, we measured Jaccard token overlap between ICL outputs and the `expected_output` text:

```
mean Jaccard zero-shot   = 0.155
mean Jaccard ICL-5shot   = 0.558      <-- +0.403 (t=+7.83, p<1e-9)
cod_009 ICL Jaccard      = 1.000      <-- VERBATIM copy
```

The model wasn't reasoning. It was copying. And `judge_stub.correctness` is literally Jaccard token overlap with `expected_output`, so the rubric rewarded the gaming directly.

We then audited the same outputs with three pairwise BT judges:

| Judge | P(ICL beats baseline) | 95% CI |
|---|---:|---|
| Qwen 3B (weak local) | 0.7727 | [0.66, 0.89] |
| Phi-4 (strong RLHF) | 0.4091 | [0.25, 0.55] |
| **gpt-4o-mini** | **0.2273** | **[0.09, 0.41]** |

CI strictly below 0.5. The win disappeared under stronger judging.

Then we ran the antidote — execution-based HumanEval (pass@1 cannot be rubric-gamed). Three conditions:

| Condition | n=20 | n=50 |
|---|---:|---:|
| baseline (raw HumanEval prompt) | 0/20 | 0/50 |
| terseness wrapper (`max 1 sentence`) | 4/20 = 20% | 15/50 = 30% |
| **codegen wrapper (`ONLY Python code`)** | **13/20 = 65%** | **36/50 = 72%** |

Findings strengthened at larger n. Real, ungameable lift exists when the wrapper is right.

judge.lab packages this discipline: pre-registered kill thresholds, BT pairwise audits, conformal abstention bands, judge bias detection, execution-based eval. MIT licensed.

Companion: skills.lab (https://github.com/ElromEvedElElyon/skills-lab) for hot-reloadable Markdown skills — inspired by google/skills + Warp.

Use at your own risk — this tool will expose your eval pipeline weaknesses.

---

## X / Twitter thread (8 tweets)

**1/**
We thought we had a SIX-SIGMA LLM eval win.

Then we measured Jaccard token overlap with the expected output.

Our model wasn't reasoning. It was copying.

Here's what we found, and what we built. 🧵

**2/**
Our in-context learning encoder beat baseline by Δ=+0.457 on Qwen 3B local.
Codegen domain: Δ=+1.212 Bonferroni-significant.

Stub-judge said: SIX SIGMA win.

We almost shipped it.

**3/**
Then we measured Jaccard(output, expected_output):

mean zero-shot   = 0.155
mean ICL-5shot   = 0.558  ← +0.403 (p<1e-9)

cod_009 ICL Jaccard = **1.000**

Verbatim copy. The judge rewards token overlap. The encoder retrieved the answer in the demos.

**4/**
Three independent pairwise BT judges, same 20 codegen pairs:

Qwen 3B BT:    P(ICL>baseline) = 0.77 [0.66, 0.89]  cons=30%
Phi-4 BT:      P = 0.41 [0.25, 0.55]  cons=70%
gpt-4o-mini:   P = 0.23 [0.09, 0.41]  cons=40%

Stronger judge → smaller win. CI strictly below 0.5 at the top.

The "win" was format-matching.

**5/**
The antidote: execution-based eval. pass@1 is binary — code runs or doesn't.

HumanEval n=50, Qwen 3B local:

baseline (raw prompt):           0/50 = 0%
terseness wrapper:              15/50 = 30%
codegen-strict wrapper:         36/50 = 72%  ←

Real, ungameable lift when the wrapper is right.

**6/**
Mechanism: raw Qwen 3B emits "Certainly! Here's..." preamble + markdown fences. Parser rejects everything.

A wrapper saying "Output ONLY Python code, no fences" matches the parser exactly.

The lift is EMITTING-DISCIPLINE, not reasoning gain. But it's REAL.

**7/**
We packaged the discipline:

🔬 judge.lab — BT pairwise audits, conformal abstention, judge bias detection, execution sandbox
🛠 skills.lab — Markdown-defined templates with hot-reload (google/skills + Warp pattern)

MIT licensed. Use at your own risk.

github.com/ElromEvedElElyon/judge-lab
github.com/ElromEvedElElyon/skills-lab

**8/**
9 falsifications shipped. 1049 tests. 2230 real LLM calls.

If your judge can be gamed, your eval is fiction.

What's your eval gaming?  Run our Jaccard probe. We dare you.

---

## arXiv preprint (abstract draft)

**Title:** *Empirical Falsification of Substrate-Prefix Encoders for LLM Routing under Cross-Judge Bradley-Terry and Execution-Based Evaluation*

**Abstract:**

We present a 48-hour empirical investigation into structured-prefix encoders for LLM workflow routing, conducted under brutal pre-registered falsification protocols. Across nine candidate approaches — including a 12-dimensional cognitive substrate prefix, in-context learning retrieval over belief-vector cosine, automatic prompt optimization (APE/OPRO), and structural causal model sanity checks — we report eight empirical falsifications and one surviving claim.

The primary mechanistic finding is that ICL retrieval of demonstration shots whose `expected_output` text is semantically similar to the target's `expected_output` produces measured Jaccard token overlaps approaching 1.000 (cod_009 = 1.000, verbatim echo), inflating stub-judge correctness scores by Δ=+0.40 (t=+7.83, p<1e-9, n=20) without genuine capability lift. We confirm this is judge-stub gaming via three independent pairwise Bradley-Terry judges showing monotone-decreasing P(ICL beats baseline) — Qwen 3B 0.77 → Phi-4 0.41 → gpt-4o-mini 0.23 (95% CI [0.09, 0.41] strictly below chance).

We further show via cross-model codegen-only analysis (Phi-4 +1.022, gpt-4o +0.864, Llama-3.3-70B +0.267, all p<0.05 stub-judge) that the stub-judge "wins" reduce to a domain-confounded codegen-vs-other-domains marker; SCM-flagged subset gaps within codegen are NEGATIVE (-0.40 to -0.75) across all three models.

Finally, under execution-based evaluation on HumanEval (pass@1 ungameable), a domain-specialised codegen wrapper achieves 72% pass@1 (n=50, +0.720 absolute over raw-prompt baseline 0/50) on Qwen 3B local — a real but capability-adjacent lift driven by emitting-only-code discipline rather than reasoning improvement.

We release `judge.lab` (MIT) — a pre-registered evaluation framework comprising Bradley-Terry pairwise audits, conformal abstention bands, judge bias detection, and minimal sandboxed execution-based eval — and `skills.lab` (MIT) — a Markdown-defined skill catalogue with belief-vector routing, inspired by google/skills and Warp.

**Keywords:** LLM evaluation, Bradley-Terry pairwise, conformal prediction, ICL retrieval, judge bias, format gaming, execution-based eval, HumanEval, pre-registered falsification.

**Subject classification:** cs.CL (Computation and Language) primary; cs.LG (Machine Learning) secondary.

---

## Awesome-list PR template

For repos like `CSHaitao/Awesome-LLMs-as-Judges` or `onejune2018/Awesome-LLM-Eval`:

**Title:** Add judge-lab — pre-registered LLM eval framework with BT audits + conformal + execution-based

**Body:**

Hi! I'd like to add **judge-lab** to the list:

> [judge-lab](https://github.com/ElromEvedElElyon/judge-lab) — Pre-registered, brutally honest LLM evaluation. Bradley-Terry pairwise audits + bootstrap CIs, conformal abstention bands, judge bias detection (position + style), and minimal sandboxed Python executor for HumanEval/MBPP/LiveCodeBench. Ships with empirical demos showing ICL retrieval gaming via direct Jaccard measurement (cod_009 = 1.000 verbatim copy of expected_output). MIT-licensed.

Headline empirical finding: stub-judge claimed +0.457 SIX SIGMA win for ICL retrieval on Qwen 3B; three independent strong judges (Phi-4, gpt-4o-mini) collapsed it to P=0.23 [CI 0.09, 0.41] strictly below chance. We package the discipline that catches this kind of gaming.

Companion: [skills-lab](https://github.com/ElromEvedElElyon/skills-lab) — Markdown-defined hot-reloadable skills with belief-vector routing.

Happy to adjust positioning/wording per the list's conventions. Thanks!
