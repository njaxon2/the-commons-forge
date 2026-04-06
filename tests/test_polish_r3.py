# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 3: utility builtins tests.

V-model traceability backfill: R-POL3-01 through R-POL3-02.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestUtilityBuiltins:
    """R-POL3-01: System information, timing, and environment builtins
    (computer, version, ver, license, getenv, setenv, system, tic/toc,
    clock, now) SHALL return correct results matching Octave semantics.

    Model-user argument: Engineers writing portable scripts depend on
    computer() to detect the platform, version()/ver() to check the
    runtime, tic/toc for benchmarking, and system() to shell out.
    These are basic infrastructure functions; if any is missing or
    broken, the engineer cannot replicate their existing Octave
    workflow.

    Decomposition:
      R-POL3-01a: computer() returns a platform string.
      R-POL3-01b: version() includes 'Forge' identifier.
      R-POL3-01c: ver() includes 'Forge' and 'Python'.
      R-POL3-01d: license() returns 'Apache'.
      R-POL3-01e: getenv retrieves an environment variable.
      R-POL3-01f: setenv/getenv round-trip preserves the value.
      R-POL3-01g: system() runs a shell command and returns status 0.
      R-POL3-01h: tic/toc measures elapsed time.
      R-POL3-01i: clock() returns a 6-element date vector.
      R-POL3-01j: now() returns a datenum value.

    Consistency: These ten sub-requirements cover every builtin listed
    in the parent. Each is independently testable. Together they confirm
    the complete system-information surface is operational.
    """

    def test_computer(self, s):
        """R-POL3-01a: computer() returns a platform identifier string."""
        r = s.eval("computer()")
        assert "linux" in str(r).lower() or "darwin" in str(r).lower() or "mingw" in str(r).lower()

    def test_version(self, s):
        """R-POL3-01b: version() includes 'Forge'."""
        r = s.eval("version()")
        assert "Forge" in str(r)

    def test_ver(self, s):
        """R-POL3-01c: ver() includes 'Forge' and 'Python'."""
        r = s.eval("ver()")
        assert "Forge" in str(r)
        assert "Python" in str(r)

    def test_license(self, s):
        """R-POL3-01d: license() returns 'Apache'."""
        r = s.eval("license()")
        assert "Apache" in str(r)

    def test_getenv(self, s):
        """R-POL3-01e: getenv retrieves an environment variable."""
        r = s.eval("getenv('HOME')")
        assert len(str(r)) > 0

    def test_setenv_getenv(self, s):
        """R-POL3-01f: setenv/getenv round-trip preserves value."""
        s.eval("setenv('FORGE_TEST_VAR', 'hello123')")
        r = s.eval("getenv('FORGE_TEST_VAR')")
        assert "hello123" in str(r)

    def test_system(self, s):
        """R-POL3-01g: system() runs shell command, returns status 0."""
        r = s.eval("[status, output] = system('echo forge_test'); status")
        assert float(r) == 0.0

    def test_tic_toc(self, s):
        """R-POL3-01h: tic/toc measures positive elapsed time."""
        s.eval("tic()")
        import time; time.sleep(0.05)
        r = s.eval("toc()")
        assert float(r) >= 0.01

    def test_clock(self, s):
        """R-POL3-01i: clock() returns a 6-element date vector."""
        r = s.eval("clock()")
        # Should be 6-element vector [year month day hour minute second]
        r_val = str(r); parts = r_val.strip().split()
        assert len(parts) >= 6

    def test_now(self, s):
        """R-POL3-01j: now() returns a datenum value."""
        r = s.eval("now()")
        # Datenum for 2026 should be large
        assert float(r) > 700000


class TestClassdefExpanded2:
    """R-POL3-02: Classdef definitions SHALL support multi-argument
    methods and class redefinition within the same session.

    Model-user argument: Engineers iterating on class designs in the
    command window expect to redefine a classdef and have new instances
    reflect the updated definition. They also need methods with multiple
    arguments beyond just 'obj'. Both patterns are standard interactive
    Octave/MATLAB workflow.

    Decomposition:
      R-POL3-02a: A method with two extra arguments computes correctly.
      R-POL3-02b: Redefining a classdef updates the default property value
                   for new instances.

    Consistency: Multi-argument methods test the method dispatch path,
    while redefinition tests the class registry. Together they cover the
    two classdef edge cases in this round.
    """

    def test_method_two_args(self, s):
        """R-POL3-02a: Method with two extra arguments computes correctly."""
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
        """R-POL3-02b: Redefining classdef updates property defaults."""
        s.eval("classdef Tmp\n  properties\n    v = 1\n  end\nend")
        s.eval("t = Tmp()")
        assert float(s.eval("t.v")) == 1.0
        # Redefine with different default
        s.eval("classdef Tmp\n  properties\n    v = 99\n  end\nend")
        s.eval("t2 = Tmp()")
        assert float(s.eval("t2.v")) == 99.0
