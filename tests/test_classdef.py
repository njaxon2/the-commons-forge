"""Tests for classdef OOP and unwind_protect."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestClassdef:
    """Test basic classdef OOP support."""

    def test_define_class(self, s):
        code = "classdef MyClass\n  properties\n    x = 0\n    y = 0\n  end\nend\n"
        s.eval(code)
        from forge.engine.classdef import class_exists
        assert class_exists("MyClass")

    def test_construct_instance(self, s):
        code = "classdef Pt\n  properties\n    x = 0\n    y = 0\n  end\n  methods\n    function obj = Pt(obj, x, y)\n      obj.x = x;\n      obj.y = y;\n    end\n  end\nend\n"
        s.eval(code)
        s.eval("p = Pt(3, 4)")
        x = s.eval("p.x")
        assert float(x) == 3.0

    def test_method_call(self, s):
        code = "classdef Pt2\n  properties\n    x = 0\n    y = 0\n  end\n  methods\n    function obj = Pt2(obj, x, y)\n      obj.x = x;\n      obj.y = y;\n    end\n    function d = dist(obj)\n      d = sqrt(obj.x^2 + obj.y^2);\n    end\n  end\nend\n"
        s.eval(code)
        s.eval("p = Pt2(3, 4)")
        d = s.eval("p.dist()")
        assert abs(float(d) - 5.0) < 1e-10

    def test_property_default(self, s):
        code = "classdef Counter\n  properties\n    count = 0\n  end\nend\n"
        s.eval(code)
        s.eval("c = Counter()")
        count = s.eval("c.count")
        assert float(count) == 0.0

    def test_property_assignment(self, s):
        code = "classdef Box\n  properties\n    width = 1\n    height = 1\n  end\nend\n"
        s.eval(code)
        s.eval("b = Box()")
        s.eval("b.width = 5")
        w = s.eval("b.width")
        assert float(w) == 5.0


class TestUnwindProtect:
    """Test unwind_protect/unwind_protect_cleanup."""

    def test_cleanup_after_error(self, s):
        code = "x = 0\nunwind_protect\n  x = 1\n  error('test')\nunwind_protect_cleanup\n  x = x + 10\nend"
        s.eval(code)
        assert float(s.eval("x")) == 11.0

    def test_cleanup_without_error(self, s):
        code = "y = 5\nunwind_protect\n  y = y * 2\nunwind_protect_cleanup\n  y = y + 100\nend"
        s.eval(code)
        assert float(s.eval("y")) == 110.0

    def test_cleanup_preserves_workspace(self, s):
        s.eval("a = 1; b = 2")
        code = "unwind_protect\nc = a + b\nunwind_protect_cleanup\nd = c * 10\nend"
        s.eval(code)
        assert float(s.eval("d")) == 30.0
