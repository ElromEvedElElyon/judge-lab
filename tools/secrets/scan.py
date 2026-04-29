"""Stdlib-only secrets scanner for Rex-26 pre-commit + CI use.

PRIVATE / Trade-secret per IMUTAVEL ``feedback_ip_protection_rule``.

Scans staged files (or files passed on argv) for real-looking credential
patterns.  Test fixtures (obvious zeros / canonical examples) are
allow-listed by entropy heuristic.

Exit codes:
- 0 = clean
- 1 = real-looking secret detected (commit should abort)
- 2 = invocation error

Usage::

    # CI/scheduled
    py -3 tools/security/scan_secrets.py path/to/file [more...]

    # Pre-commit hook (called by .git/hooks/pre-commit)
    py -3 tools/security/scan_secrets.py --staged
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Rules: (regex, name, min_shannon_entropy_bits, captured_group_index)
# entropy_min filters out test fixtures like ghp_0123456789abcdef* (low entropy
# from sequential pattern) or AKIA0000...0 (zero entropy).
RULES: List[Tuple[re.Pattern, str, float, int]] = [
    # GitHub PAT (classic + fine-grained)
    (re.compile(r"\b(ghp_[A-Za-z0-9]{36,})\b"), "github_pat_classic", 4.0, 1),
    (re.compile(r"\b(github_pat_[A-Za-z0-9_]{82,})\b"), "github_pat_fg", 4.0, 1),
    (re.compile(r"\b(gho_[A-Za-z0-9]{36,})\b"), "github_oauth", 4.0, 1),
    (re.compile(r"\b(ghu_[A-Za-z0-9]{36,})\b"), "github_user_to_server", 4.0, 1),
    # OpenAI / Anthropic
    (re.compile(r"\b(sk-proj-[A-Za-z0-9_\-]{50,})\b"), "openai_project", 4.0, 1),
    (re.compile(r"\b(sk-ant-api03-[A-Za-z0-9_\-]{50,})\b"), "anthropic_api03", 4.0, 1),
    (re.compile(r"\b(sk-[A-Za-z0-9]{40,})\b"), "openai_legacy", 4.0, 1),
    # AWS
    (re.compile(r"\b(AKIA[A-Z0-9]{16})\b"), "aws_access_key", 3.5, 1),
    (re.compile(r"\b(ASIA[A-Z0-9]{16})\b"), "aws_session_token", 3.5, 1),
    # Google
    (re.compile(r"\b(AIza[A-Za-z0-9_\-]{35,})\b"), "google_api_key", 4.0, 1),
    # NVIDIA NIM / Hugging Face / Slack / Firecrawl
    (re.compile(r"\b(nvapi-[A-Za-z0-9_\-]{20,})\b"), "nvidia_nim", 4.0, 1),
    (re.compile(r"\b(hf_[A-Za-z0-9]{30,})\b"), "huggingface", 4.0, 1),
    (re.compile(r"\b(xox[bpoa]-[A-Za-z0-9\-]{20,})\b"), "slack_bot", 4.0, 1),
    (re.compile(r"\b(fc-[A-Za-z0-9]{30,})\b"), "firecrawl", 4.0, 1),
    # Generic high-entropy bearer in code (catches "Authorization: Bearer XXX")
    (re.compile(r"Bearer\s+([A-Za-z0-9_\-]{40,})"), "bearer_token", 4.5, 1),
    # SSH / PEM private keys (any length signals)
    (re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PRIVATE) ?PRIVATE KEY-----"),
     "private_key_block", 0.0, 0),
]

# File globs to skip (test fixtures, intentional examples, etc.).
SKIP_PATH_SUBSTRINGS = (
    "/tests/", "\\tests\\",
    "/test_", "\\test_",
    "/rules_expanded/secret_leak.yaml",
    "\\rules_expanded\\secret_leak.yaml",
    # Detection-rules/probes that intentionally include sample patterns:
    "/detector/", "\\detector\\",
    "/safety/", "\\safety\\",
    "/eval_mythos/", "\\eval_mythos\\",
    "scan_secrets.py",  # this file itself
)

# Files that LEGITIMATELY contain test sample tokens in fixture data.
SKIP_PATH_SUFFIXES = (
    ".sample", ".example",
    "secret_leak.yaml",
    "PHASE_4_SMOKE_RESULT_2026_04_27.md",  # contains documented test prompts
)


def _shannon_entropy(s: str) -> float:
    """Bits per character.  Real PATs have ~5-6 bits/char; test fixtures < 3."""
    if not s:
        return 0.0
    freq: dict = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    h = 0.0
    for c in freq.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _is_test_fixture_value(value: str) -> bool:
    """Heuristic: obvious zero patterns, sequential alphabets, EXAMPLE markers."""
    lower = value.lower()
    if "0000000000000" in value:
        return True
    if "abcdef0123456789" in lower or "0123456789abcdef" in lower:
        return True
    if "abcdefghijklmnop" in lower:
        return True
    if "EXAMPLE" in value:  # AWS canonical example
        return True
    if "FAKE" in value.upper() or "DUMMY" in value.upper():
        return True
    return False


def _should_skip_path(path: str) -> bool:
    p = path.replace("\\", "/")
    for s in SKIP_PATH_SUBSTRINGS:
        if s.replace("\\", "/") in p:
            return True
    for sfx in SKIP_PATH_SUFFIXES:
        if path.endswith(sfx):
            return True
    return False


def scan_text(text: str) -> List[Tuple[str, str, int]]:
    """Return list of (rule_name, matched_value, line_number)."""
    findings: List[Tuple[str, str, int]] = []
    if not text:
        return findings
    lines = text.splitlines()
    for rule_re, rule_name, min_entropy, group_idx in RULES:
        for m in rule_re.finditer(text):
            value = m.group(group_idx) if group_idx else m.group(0)
            if _is_test_fixture_value(value):
                continue
            if _shannon_entropy(value) < min_entropy:
                continue
            # Find line number.
            line_no = text[:m.start()].count("\n") + 1
            findings.append((rule_name, value[:8] + "..." + value[-4:], line_no))
    return findings


def scan_file(path: Path) -> List[Tuple[str, str, int]]:
    if _should_skip_path(str(path)):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IsADirectoryError):
        return []
    return scan_text(text)


def get_staged_files() -> List[Path]:
    """Return paths of files currently staged for commit (Added or Modified)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            text=True, encoding="utf-8", errors="ignore",
        )
    except subprocess.CalledProcessError:
        return []
    return [Path(p.strip()) for p in out.splitlines() if p.strip()]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Scan files for credential leaks.")
    parser.add_argument("paths", nargs="*", help="Files to scan.  Empty + --staged uses git index.")
    parser.add_argument("--staged", action="store_true", help="Scan only staged files.")
    args = parser.parse_args(argv)

    if args.staged:
        targets = get_staged_files()
    else:
        targets = [Path(p) for p in args.paths]

    if not targets:
        if args.staged:
            print("[secrets-scan] no staged files; OK")
            return 0
        print("[secrets-scan] no targets given; pass --staged or paths", file=sys.stderr)
        return 2

    total_findings: List[Tuple[Path, str, str, int]] = []
    for path in targets:
        if not path.is_file():
            continue
        for rule, redacted, line_no in scan_file(path):
            total_findings.append((path, rule, redacted, line_no))

    if not total_findings:
        print(f"[secrets-scan] {len(targets)} file(s) clean")
        return 0

    print("[secrets-scan] BLOCKED -- credential pattern(s) detected:", file=sys.stderr)
    for path, rule, redacted, line_no in total_findings:
        print(f"  {path}:{line_no}  rule={rule}  value={redacted}", file=sys.stderr)
    print(file=sys.stderr)
    print("Remediation:", file=sys.stderr)
    print("  1. Move the secret to E:\\Dev\\.secrets\\.env", file=sys.stderr)
    print("  2. Replace inline reference with os.environ.get(...) or _load_token()", file=sys.stderr)
    print("  3. Re-stage and re-commit", file=sys.stderr)
    print("  Bypass (use SPARINGLY, document why): git commit --no-verify", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
