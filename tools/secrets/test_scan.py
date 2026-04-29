"""Tests for tools/security/scan_secrets.py.

Run via: py -3 -m pytest tools/security/test_scan_secrets.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make tools/security importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scan_secrets import (  # noqa: E402
    _is_test_fixture_value,
    _shannon_entropy,
    _should_skip_path,
    scan_text,
)


# ---------------------------------------------------------------------------
# Entropy / fixture heuristics
# ---------------------------------------------------------------------------


def test_entropy_random_string_high() -> None:
    assert _shannon_entropy("ghp_zR3xT9bM4kQyN8aFp7Wj2HsLcE5UvGdT1Ko6") > 4.0


def test_entropy_zero_string_low() -> None:
    assert _shannon_entropy("ghp_000000000000000000000000000000000000") < 1.0


def test_entropy_empty() -> None:
    assert _shannon_entropy("") == 0.0


def test_fixture_zero_pattern() -> None:
    assert _is_test_fixture_value("AKIA0000000000000000")


def test_fixture_canonical_aws_example() -> None:
    assert _is_test_fixture_value("AKIAIOSFODNN7EXAMPLE")


def test_fixture_alphabetic_pattern() -> None:
    assert _is_test_fixture_value("ghp_abcdefghijklmnopqrstuvwxyz0123456789")


def test_fixture_real_random_not_skipped() -> None:
    assert not _is_test_fixture_value("ghp_zR3xT9bM4kQyN8aFp7Wj2HsLcE5UvGdT1Ko6")


# ---------------------------------------------------------------------------
# Path skip
# ---------------------------------------------------------------------------


def test_skip_test_subdir() -> None:
    assert _should_skip_path("src/rex26_llm/foo/tests/test_bar.py")
    assert _should_skip_path("src\\rex26_llm\\foo\\tests\\test_bar.py")


def test_skip_safety_rules() -> None:
    assert _should_skip_path("src/rex26_llm/safety/rules_expanded/secret_leak.yaml")


def test_normal_module_not_skipped() -> None:
    assert not _should_skip_path("src/rex26_llm/encoder.py")


# ---------------------------------------------------------------------------
# scan_text end-to-end
# ---------------------------------------------------------------------------


def test_scan_real_github_pat_caught() -> None:
    text = 'TOKEN = "ghp_zR3xT9bM4kQyN8aFp7Wj2HsLcE5UvGdT1Ko6"'
    findings = scan_text(text)
    assert any(rule == "github_pat_classic" for rule, _, _ in findings)


def test_scan_test_fixture_aws_skipped() -> None:
    text = 'AWS_KEY = "AKIA0000000000000000"  # test fixture'
    findings = scan_text(text)
    assert all(rule != "aws_access_key" for rule, _, _ in findings)


def test_scan_canonical_aws_example_skipped() -> None:
    text = 'docs say AKIAIOSFODNN7EXAMPLE is the canonical example'
    findings = scan_text(text)
    assert all(rule != "aws_access_key" for rule, _, _ in findings)


def test_scan_clean_text_returns_empty() -> None:
    text = """
    def hello():
        return "world"
    """
    assert scan_text(text) == []


def test_scan_pem_private_key_block_caught() -> None:
    text = """leading
    -----BEGIN RSA PRIVATE KEY-----
    MIIEpAIBAAKCAQEA...
    -----END RSA PRIVATE KEY-----
    trailing"""
    findings = scan_text(text)
    assert any(rule == "private_key_block" for rule, _, _ in findings)


def test_scan_real_anthropic_caught() -> None:
    # 50+ random chars -- ENTROPY high, no test fixture markers.
    suffix = "qZ7vN3mB8fC4xK9pT2hL6yE5rS1uW0iD7gJ4aP3oM9nQ2bX6rT8c"
    text = f'KEY = "sk-ant-api03-{suffix}"'
    findings = scan_text(text)
    assert any(rule == "anthropic_api03" for rule, _, _ in findings)


def test_scan_real_bearer_caught() -> None:
    # Synthetic 50-char high-entropy bearer
    text = 'Authorization: Bearer eyJ0eXAiOiJKV1QzNzU4OWFiY2RlZjAxMjM0NTY3ODkwYWJjZGVmMDE'
    findings = scan_text(text)
    # bearer rule has min entropy 4.5, this should pass
    assert any(rule == "bearer_token" for rule, _, _ in findings)
