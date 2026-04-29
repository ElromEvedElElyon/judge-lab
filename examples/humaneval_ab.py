"""#47 HumanEval execution-based AB — encoder_d vs zero-shot vs codegen-tuned.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

Tests the surviving claim from Rex-26 falsification work
(encoder_d codegen-fast_answer cross-model: Phi-4 +1.022, gpt-4o
+0.864 BUT under stub-judge only) under EXECUTION-BASED eval.

Pass@1 cannot be rubric-gamed.  If encoder_d helps under execution
eval, the win is REAL.  If it ties or loses, the cross-model
"win" was the same gaming as ICL +0.457 (commit d84fd4b).

Three conditions
- baseline:    raw HumanEval prompt (function header + docstring)
- encoder_d:   encoder_d's fast_answer template wrapping the prompt
- codegen_v1:  custom prompt that asks for code-only output

Pre-registered kill: encoder_d pass@1 < baseline pass@1 + 0.05 -> the
cross-model stub-judge "win" was format gaming, not capability lift.

Usage::

    py -3 scripts/run_humaneval_ab.py [--n N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from sandbox_exec import run_humaneval_case  # noqa: E402


def _ollama(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=json.dumps({
            "model": "local-qwen-3b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        obj = json.loads(r.read().decode("utf-8", errors="replace"))
        return obj["choices"][0]["message"]["content"] or ""


# Conditions:
TEMPLATE_ENCODER_D = "Answer concisely (max 1 sentence): {prompt}"
TEMPLATE_CODEGEN_V1 = (
    "Complete the following Python function. Output ONLY Python code, "
    "no markdown fences, no explanation, no docstring repeat:\n\n{prompt}"
)


def _wrap(template: str | None, prompt: str) -> str:
    if template is None:
        return prompt  # baseline
    return template.replace("{prompt}", prompt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="Number of HumanEval cases (default 20)")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    ds_path = _HERE / "datasets" / "HumanEval.jsonl"
    if not ds_path.is_file():
        print(f"ERR: dataset missing at {ds_path}", file=sys.stderr)
        return 2
    cases = [json.loads(l) for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = cases[:args.n]
    n = len(cases)

    conditions = [
        ("baseline", None),
        ("encoder_d", TEMPLATE_ENCODER_D),
        ("codegen_v1", TEMPLATE_CODEGEN_V1),
    ]

    results = {label: [] for label, _ in conditions}
    raw_outputs = {label: [] for label, _ in conditions}
    print(f"[humaneval-ab] n={n}  conditions={[c[0] for c in conditions]}",
          file=sys.stderr)

    t0 = time.time()
    for i, case in enumerate(cases, 1):
        for label, template in conditions:
            wrapped = _wrap(template, case["prompt"])
            try:
                out = _ollama(wrapped, max_tokens=512, temperature=0.0)
            except Exception as e:
                print(f"  ERR {case['task_id']}/{label}: {type(e).__name__}",
                      file=sys.stderr)
                out = ""
            r = run_humaneval_case(case, out, timeout_s=args.timeout)
            results[label].append(r.passed)
            raw_outputs[label].append({
                "task_id": case["task_id"],
                "passed": r.passed,
                "rc": r.returncode,
                "timeout": r.timeout_hit,
                "stderr": r.stderr[:200],
                "output_head": out[:300],
            })
        if i % 5 == 0 or i == n:
            elapsed = time.time() - t0
            row = " ".join(
                f"{label}={sum(results[label])}/{i}"
                for label, _ in conditions
            )
            print(f"[humaneval-ab] {i}/{n}  elapsed={elapsed:.1f}s  {row}",
                  file=sys.stderr)

    # Aggregate
    print()
    print("=" * 76)
    print(f"#47 HumanEval execution-based AB  n={n}  Qwen 3B local T=0")
    print("=" * 76)
    print(f"  {'condition':16s}  {'pass@1':>8s}  {'rate':>6s}")
    print("  " + "-" * 36)
    pass_counts = {}
    for label, _ in conditions:
        pcount = sum(results[label])
        rate = pcount / n
        pass_counts[label] = pcount
        print(f"  {label:16s}  {pcount:>4}/{n:<4}  {100*rate:>5.1f}%")

    print()
    enc_d_delta = (pass_counts["encoder_d"] - pass_counts["baseline"]) / n
    cg_v1_delta = (pass_counts["codegen_v1"] - pass_counts["baseline"]) / n

    print(f"encoder_d delta vs baseline:   {enc_d_delta:+.3f} ({pass_counts['encoder_d'] - pass_counts['baseline']:+d}/{n})")
    print(f"codegen_v1 delta vs baseline:  {cg_v1_delta:+.3f} ({pass_counts['codegen_v1'] - pass_counts['baseline']:+d}/{n})")

    print()
    print("PRE-REGISTERED VERDICTS")
    if enc_d_delta >= 0.05:
        print(f"  encoder_d: WIN ({enc_d_delta:+.3f} >= +0.05)")
    else:
        print(f"  encoder_d: FAIL ({enc_d_delta:+.3f} < +0.05) -- cross-model stub-judge win was format-gaming, not capability lift")
    if cg_v1_delta >= 0.05:
        print(f"  codegen_v1: WIN ({cg_v1_delta:+.3f} >= +0.05) -- proper codegen wrapper does help")
    else:
        print(f"  codegen_v1: FAIL ({cg_v1_delta:+.3f} < +0.05)")

    out_path = _HERE / f"humaneval_ab_qwen3b_n{n}.json"
    out_path.write_text(json.dumps({
        "model": "local-qwen-3b",
        "n": n,
        "pass_counts": pass_counts,
        "encoder_d_delta": enc_d_delta,
        "codegen_v1_delta": cg_v1_delta,
        "elapsed_s": time.time() - t0,
        "raw_outputs": raw_outputs,
    }, indent=2), encoding="utf-8")
    print()
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
