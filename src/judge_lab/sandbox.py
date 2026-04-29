"""Minimal sandboxed Python executor for execution-based eval.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

Used by HumanEval/MBPP/LiveCodeBench harnesses to run generated code +
test cases.  Sandboxing is **best-effort** for honest research use,
NOT a security boundary.  Do NOT use to evaluate untrusted code from
the internet without additional isolation (e.g. Docker, gVisor).

Approach
- subprocess.run with timeout (default 10s).
- cwd = fresh tempdir per call (deleted afterward).
- env stripped to PATH + SYSTEMROOT only (no API keys, no HOME).
- stdin=DEVNULL.

Returns ExecResult with passed (True/False), stdout, stderr,
returncode, elapsed_s, timeout_hit.

Usage::

    from sandbox_exec import run_python_test
    result = run_python_test(
        code="def add(a,b): return a+b",
        test="assert add(1,2) == 3",
    )
    print(result.passed)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExecResult", "run_python_test", "run_humaneval_case"]


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timeout_hit: bool


def _restricted_env() -> dict:
    """Return minimal env -- no secrets, no HOME, but PATH + SYSTEMROOT preserved."""
    keep = ("PATH", "SYSTEMROOT", "PYTHONPATH", "TEMP", "TMP",
            "ProgramFiles", "ProgramFiles(x86)", "ProgramData",
            "WINDIR", "LOCALAPPDATA", "APPDATA")
    out = {}
    for k in keep:
        v = os.environ.get(k)
        if v is not None:
            out[k] = v
    # Disable any third-party Python packages that might leak: PYTHONNOUSERSITE
    out["PYTHONNOUSERSITE"] = "1"
    return out


def run_python_test(
    code: str,
    test: str,
    entry_point: str = "candidate",
    timeout_s: float = 10.0,
    python_bin: str | None = None,
) -> ExecResult:
    """Execute ``code`` then ``test``; return ExecResult.

    The test is wrapped to call ``check(<entry_point>)``.  If
    ``test`` already calls check() at module level, no extra wrapper.

    Parameters
    ----------
    code:
        Python code defining the entry_point function.
    test:
        Python code containing a ``check(candidate)`` function and
        any setup.  HumanEval convention.
    entry_point:
        Name of the function in ``code`` to be evaluated.
    timeout_s:
        Hard wall timeout.
    python_bin:
        Optional explicit Python executable.  Defaults to current.
    """
    if python_bin is None:
        python_bin = sys.executable

    # Compose a self-contained script.
    script = (
        code.rstrip()
        + "\n\n# --- TEST BLOCK ---\n"
        + test.rstrip()
        + f"\n\ncheck({entry_point})\nprint('OK')\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")

        t0 = time.time()
        try:
            r = subprocess.run(
                [python_bin, "-I", "-S", str(path)],  # -I isolates user site, -S skips site.py
                capture_output=True, text=True, errors="replace",
                stdin=subprocess.DEVNULL,
                env=_restricted_env(),
                cwd=tmp,
                timeout=timeout_s,
            )
            elapsed = time.time() - t0
            passed = (r.returncode == 0 and "OK" in r.stdout)
            return ExecResult(
                passed=passed,
                returncode=r.returncode,
                stdout=r.stdout[:2000],
                stderr=r.stderr[:2000],
                elapsed_s=elapsed,
                timeout_hit=False,
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            return ExecResult(
                passed=False,
                returncode=-1,
                stdout=(e.stdout or b"").decode("utf-8", errors="replace")[:2000] if e.stdout else "",
                stderr="TIMEOUT after %.1fs" % timeout_s,
                elapsed_s=elapsed,
                timeout_hit=True,
            )
        except Exception as ex:
            return ExecResult(
                passed=False,
                returncode=-2,
                stdout="",
                stderr=f"{type(ex).__name__}: {ex}",
                elapsed_s=time.time() - t0,
                timeout_hit=False,
            )


def run_humaneval_case(
    case: dict,
    completion: str,
    timeout_s: float = 10.0,
) -> ExecResult:
    """Run a HumanEval case with an LLM-generated completion.

    HumanEval format: case has ``prompt`` (function header + docstring),
    ``test`` (with check()), ``entry_point`` (function name).

    The completion can be either:
    - The function body only (typical raw LLM output): we prepend prompt.
    - A full function definition starting with `def`: we use as-is.
    - Code with markdown fences: stripped before eval.
    """
    code = _compose_full_source(case, completion)
    return run_python_test(
        code=code,
        test=case["test"],
        entry_point=case["entry_point"],
        timeout_s=timeout_s,
    )


def _strip_markdown_fences(text: str) -> str:
    """Strip ```python ... ``` markdown fences if present.

    Preserves leading whitespace AFTER the fence is removed (HumanEval
    canonical solutions are body-only with 4-space indent).
    """
    t = text
    # Strip leading/trailing whitespace ONLY around markdown fences.
    stripped = t.strip()
    if stripped.startswith("```"):
        # Find end of opening fence line
        nl = stripped.find("\n")
        if nl > 0:
            t = stripped[nl + 1:]
        # Drop closing fence
        if "```" in t:
            t = t[:t.rindex("```")]
        # Now t may have its own indentation; preserve as-is
        # but trim trailing whitespace.
        return t.rstrip()
    return t


def _compose_full_source(case: dict, completion: str) -> str:
    """Combine prompt header with body to form executable code.

    HumanEval canonical solutions are function bodies with 4-space
    indent.  We must preserve that indent so the body becomes the
    function body of the prompt's def signature.
    """
    cleaned = _strip_markdown_fences(completion)
    # If completion already has 'def <entry_point>', use as-is.
    if f"def {case['entry_point']}" in cleaned:
        return cleaned
    # Otherwise, prepend the prompt header (which contains def + docstring).
    # Preserve cleaned's leading indent.  Ensure prompt ends with newline.
    prompt = case["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt + cleaned
