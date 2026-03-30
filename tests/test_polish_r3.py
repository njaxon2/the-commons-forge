# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 3: utility builtins tests."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestUtilityBuiltins:
    """Test system info, timing, and environment builtins."""

    def test_computer(self, s):
        r = s.eval("computer()")
        assert "linux" in str(r).lower() or "darwin" in str(r).lower() or "mingw" in str(r).lower()

    def test_version(self, s):
        r = s.eval("version()")
        assert "Forge" in str(r)

    def test_ver(self, s):
        r = s.eval("ver()")
        assert "Forge" in str(r)
        assert "Python" in str(r)

    def test_license(self, s):
        r = s.eval("license()")
        assert "Apache" in str(r)

    def test_getenv(self, s):
        r = s.eval("getenv('HOME')")
        assert len(str(r)) > 0

    def test_setenv_getenv(self, s):
        s.eval("setenv('FORGE_TEST_VAR', 'hello123')")
        r = s.eval("getenv('FORGE_TEST_VAR')")
        assert "hello123" in str(r)

    def test_system(self, s):
        r = s.eval("[status, output] = system('echo forge_test'); status")
        assert float(r) == 0.0

    def test_tic_toc(self, s):
        s.eval("tic()")
        import time; time.sleep(0.05)
        r = s.eval("toc()")
        assert float(r) >= 0.01

    def test_clock(self, s):
        r = s.eval("clock()")
        # Should be 6-element vector [year month day hour minute second]
        r_val = str(r); parts = r_val.strip().split()
        assert len(parts) >= 6

    def test_now(self, s):
        r = s.eval("now()")
        # Datenum for 2026 should be large
        assert float(r) > 700000


class TestClassdefExpanded2:
    """More classdef edge cases."""

    def test_method_two_args(self, s):
        """Method with two extra arguments."""
        code = """classdef Calc
  properties
    base = 0
  end
  methods
    function obj = Calc(b)
      obj.base = b;
    end
    function r = compute(obj, x, y)
      r = obj.base + x * y;
    end
  end
end"""
        s.eval(code)
        s.eval("c = Calc(100)")
        r = s.eval("c.compute(3, 4)")
        assert float(r) == 112.0

    def test_class_redefine(self, s):
        """Redefining a class should work."""
        s.eval("classdef Tmp\n  properties\n    v = 1\n  end\nend")
        s.eval("t = Tmp()")
        assert float(s.eval("t.v")) == 1.0
        # Redefine with different default
        s.eval("classdef Tmp\n  properties\n    v = 99\n  end\nend")
        s.eval("t2 = Tmp()")
        assert float(s.eval("t2.v")) == 99.0
