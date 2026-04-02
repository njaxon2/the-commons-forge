# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: verify basic engine startup and framework."""
import pytest


def test_validation_framework():
    """Tolerance comparison works."""
    from forge.validation.framework import arrays_close
    import numpy as np
    ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert ok, msg
    ok, msg = arrays_close([1.0], [1.0 + 1e-15])
    assert ok, msg
    ok, _ = arrays_close([1.0], [2.0])
    assert not ok


def test_forge_version():
    """Version is set."""
    import forge
    assert forge.__version__ == "0.3.5"


def test_session_basic():
    """Engine session can evaluate simple expressions."""
    from forge.engine.session import ForgeSession
    s = ForgeSession()
    result = s.eval("2+2")
    assert "4" in str(result)
