# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: verify basic engine startup and framework.

Requirement R-SMOKE-01: The Forge engine SHALL be importable and capable of
evaluating a trivial arithmetic expression within a fresh session.

Model-user argument: An engineer migrating from MATLAB/Octave expects to install
Forge, launch it, and immediately type "2+2" at the prompt. If the engine cannot
even start or evaluate simple expressions, no other functionality matters. These
smoke tests are the first line of defense against broken releases.

Decomposition:
    R-SMOKE-01.1  The validation framework SHALL compare numeric arrays within
                  floating-point tolerance (arrays_close).
    R-SMOKE-01.2  The forge package SHALL expose a valid __version__ string.
    R-SMOKE-01.3  A ForgeSession SHALL evaluate "2+2" and return a result
                  containing "4".

Consistency argument: R-SMOKE-01.1 verifies the test infrastructure itself is
working. R-SMOKE-01.2 confirms the package metadata is intact (needed for
updates, about dialogs, and traceability). R-SMOKE-01.3 confirms the full
parse-evaluate pipeline produces correct output for the simplest possible case.
Together these three sub-requirements establish that the engine is alive,
correctly versioned, and capable of basic evaluation.
"""
import pytest


def test_validation_framework():
    """R-SMOKE-01.1: Tolerance comparison produces correct accept/reject."""
    from forge.validation.framework import arrays_close
    import numpy as np
    ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ok, msg
    ok, msg = arrays_close([1.0], [1.0 + 1e-15])
    assert ok, msg
    ok, _ = arrays_close([1.0], [2.0])
    assert not ok


def test_forge_version():
    """R-SMOKE-01.2: Package version string is set and current."""
    import forge
    assert forge.__version__ == "0.3.10"


def test_session_basic():
    """R-SMOKE-01.3: Fresh session evaluates '2+2' to a result containing '4'."""
    from forge.engine.session import ForgeSession
    s = ForgeSession()
    result = s.eval("2+2")
    assert "4" in str(result)
