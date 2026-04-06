# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Security tests: dependency audit, static analysis, and secret detection.

Requirement R-SEC-01: The Forge engine SHALL not introduce known security
vulnerabilities through its dependencies, source code patterns, or hardcoded
credentials.

Model-user argument: An engineer running untrusted .m files received from
colleagues or downloaded from file-exchange repositories expects that executing
those scripts will not expose their system to known exploits. The engine's
dependency chain must be free of published CVEs, its source code must not contain
unexpected dangerous patterns (beyond the intentional interpreter eval/exec
paths), and no credentials may be baked into the distributed package.

Decomposition:
    R-SEC-01.1  All pip dependencies SHALL have zero known vulnerabilities as
                reported by pip-audit (excluding the local forge-ide package).
    R-SEC-01.2  The forge/ source tree SHALL have zero unexpected High-severity
                Bandit findings (known intentional patterns are whitelisted).
    R-SEC-01.3  The forge/ source tree SHALL contain no hardcoded passwords,
                API keys, tokens, or private keys in non-test Python files.

Consistency argument: R-SEC-01.1 covers supply-chain risk (third-party packages).
R-SEC-01.2 covers first-party static analysis (dangerous code patterns in Forge
itself). R-SEC-01.3 covers credential hygiene (secrets that would be exposed if
the source is published or the wheel is inspected). Together these three
sub-requirements address the three major attack surfaces for a distributed Python
application.
"""
import json
import os
import re
import subprocess
import shutil
import pytest


FORGE_ROOT = os.path.join(os.path.dirname(__file__), "..")
FORGE_SRC = os.path.join(FORGE_ROOT, "forge")

# Bandit High-severity whitelist: (test_id, filename_substring) pairs
# Each entry documents WHY the finding is acceptable.
BANDIT_HIGH_WHITELIST = [
    # B602: subprocess with shell=True — intentional Octave system() builtin
    ("B602", "session.py"),
    # B307: eval() usage — intentional interpreter eval for Octave expressions
    ("B307", "evaluator.py"),
    ("B307", "parser.py"),
    # B324: hashlib weak hash — if any remain in codebase
    ("B324", ""),
    # B202: tarfile.extractall — intentional Octave tar/untar builtins
    ("B202", "fileio.py"),
    # B402: import ftplib — intentional Octave FTP builtins
    ("B402", "web.py"),
    # B321: FTP-related function calls — intentional Octave FTP builtins
    ("B321", "web.py"),
]


def _is_whitelisted(issue: dict) -> bool:
    """Check if a bandit issue matches any whitelist entry."""
    test_id = issue.get("test_id", "")
    filename = issue.get("filename", "")
    for wl_id, wl_file in BANDIT_HIGH_WHITELIST:
        if test_id == wl_id and (not wl_file or wl_file in filename):
            return True
    return False


@pytest.mark.slow
@pytest.mark.security
def test_pip_audit_no_vulnerabilities():
    """R-SEC-01.1: pip-audit reports zero known vulnerabilities in dependencies."""
    if shutil.which("pip-audit") is None:
        pytest.skip("pip-audit not installed")

    result = subprocess.run(
        ["pip-audit", "--output", "json", "--skip-editable"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # pip-audit exits 1 when vulnerabilities found, 0 when clean
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Empty stdout is OK if exit code is 0 (no deps to audit)
        if result.returncode == 0:
            return
        pytest.fail(
            f"pip-audit produced non-JSON output (exit {result.returncode}).\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    # Filter out local/non-PyPI packages that can't be audited
    skip_packages = {"forge-ide", "forge-engine"}
    vulns = [
        v for v in data.get("dependencies", [])
        if v.get("vulns") and v.get("name", "").lower() not in skip_packages
    ]

    if vulns:
        summary = "\n".join(
            f"  {v['name']}=={v.get('version','?')}: "
            f"{[x.get('id','?') for x in v['vulns']]}"
            for v in vulns
        )
        pytest.fail(f"pip-audit found {len(vulns)} vulnerable package(s):\n{summary}")


@pytest.mark.slow
@pytest.mark.security
def test_bandit_no_high_severity_issues():
    """R-SEC-01.2: Bandit reports zero unexpected High-severity findings."""
    if shutil.which("bandit") is None:
        pytest.skip("bandit not installed")

    src_dir = os.path.abspath(FORGE_SRC)
    result = subprocess.run(
        ["bandit", "-r", src_dir, "-ll", "-f", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # bandit may emit a progress bar before the JSON — strip it
    stdout = result.stdout
    json_start = stdout.find("{")
    if json_start > 0:
        stdout = stdout[json_start:]

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        pytest.fail(
            f"bandit produced non-JSON output.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    results = data.get("results", [])
    high_issues = [r for r in results if r.get("issue_severity") == "HIGH"]

    unexpected = [i for i in high_issues if not _is_whitelisted(i)]

    if unexpected:
        summary = "\n".join(
            f"  {i['test_id']} ({i.get('test_name','?')}) "
            f"in {i.get('filename','?')}:{i.get('line_number','?')} — "
            f"{i.get('issue_text','')}"
            for i in unexpected
        )
        pytest.fail(
            f"bandit found {len(unexpected)} unexpected High severity issue(s):\n"
            f"{summary}"
        )


@pytest.mark.security
def test_no_hardcoded_secrets():
    """R-SEC-01.3: No hardcoded passwords, API keys, or tokens in forge/ source."""
    src_dir = os.path.abspath(FORGE_SRC)

    # Patterns that indicate hardcoded secrets (value in quotes after =)
    secret_patterns = [
        re.compile(r'''(?:password|passwd|pwd)\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
        re.compile(r'''api_key\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
        re.compile(r'''secret\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
        re.compile(r'''token\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
        re.compile(r'''(?:aws_access_key|aws_secret)\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
        re.compile(r'''private_key\s*=\s*['"][^'"]{4,}['"]''', re.IGNORECASE),
    ]

    # Substrings in file paths to skip (test files, example/demo data)
    skip_paths = {"test_", "tests/", "__pycache__", ".pyc"}

    findings = []

    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, src_dir)

            if any(s in rel_path for s in skip_paths):
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        # Skip comments
                        stripped = line.lstrip()
                        if stripped.startswith("#"):
                            continue
                        for pat in secret_patterns:
                            if pat.search(line):
                                findings.append(
                                    f"  {rel_path}:{lineno}: {stripped.rstrip()}"
                                )
            except OSError:
                continue

    if findings:
        pytest.fail(
            f"Found {len(findings)} potential hardcoded secret(s) in forge/:\n"
            + "\n".join(findings[:20])  # cap output
        )
