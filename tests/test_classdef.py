# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
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


class TestClassdefExpanded:
    """Extended classdef tests: new-style constructors, multi-method, returns."""

    def test_new_style_constructor(self, s):
        """Constructor where obj is NOT in params: function obj = Cls(a, b)."""
        code = """classdef Vec3
  properties
    x = 0; y = 0; z = 0
  end
  methods
    function obj = Vec3(a, b, c)
      obj.x = a; obj.y = b; obj.z = c;
    end
  end
end"""
        s.eval(code)
        s.eval("v = Vec3(1, 2, 3)")
        assert float(s.eval("v.x")) == 1.0
        assert float(s.eval("v.y")) == 2.0
        assert float(s.eval("v.z")) == 3.0

    def test_method_with_args(self, s):
        """Method that takes additional arguments beyond self."""
        code = """classdef Adder
  properties
    base = 0
  end
  methods
    function obj = Adder(val)
      obj.base = val;
    end
    function r = add(obj, x)
      r = obj.base + x;
    end
  end
end"""
        s.eval(code)
        s.eval("a = Adder(10)")
        r = s.eval("a.add(5)")
        assert float(r) == 15.0

    def test_method_returning_object(self, s):
        """Method that constructs and returns a new object."""
        code = """classdef Pt3
  properties
    x = 0; y = 0
  end
  methods
    function obj = Pt3(a, b)
      obj.x = a; obj.y = b;
    end
    function d = dist(obj)
      d = sqrt(obj.x^2 + obj.y^2);
    end
    function r = scale(obj, f)
      r = Pt3(obj.x * f, obj.y * f);
    end
  end
end"""
        s.eval(code)
        s.eval("p = Pt3(3, 4)")
        assert abs(float(s.eval("p.dist()")) - 5.0) < 1e-10
        s.eval("q = p.scale(2)")
        assert float(s.eval("q.x")) == 6.0
        assert float(s.eval("q.y")) == 8.0

    def test_no_constructor(self, s):
        """Class with no explicit constructor uses default."""
        code = """classdef Simple
  properties
    val = 42
  end
end"""
        s.eval(code)
        s.eval("s = Simple()")
        assert float(s.eval("s.val")) == 42.0

    def test_multiple_methods(self, s):
        """Class with several methods."""
        code = """classdef Rect
  properties
    w = 0; h = 0
  end
  methods
    function obj = Rect(width, height)
      obj.w = width; obj.h = height;
    end
    function a = area(obj)
      a = obj.w * obj.h;
    end
    function p = perimeter(obj)
      p = 2 * (obj.w + obj.h);
    end
    function d = diagonal(obj)
      d = sqrt(obj.w^2 + obj.h^2);
    end
  end
end"""
        s.eval(code)
        s.eval("r = Rect(3, 4)")
        assert float(s.eval("r.area()")) == 12.0
        assert float(s.eval("r.perimeter()")) == 14.0
        assert abs(float(s.eval("r.diagonal()")) - 5.0) < 1e-10

    def test_old_style_constructor(self, s):
        """Old-style: function obj = Cls(obj, x, y) where obj appears in params."""
        code = """classdef OldPt
  properties
    x = 0; y = 0
  end
  methods
    function obj = OldPt(obj, a, b)
      obj.x = a; obj.y = b;
    end
    function d = dist(obj)
      d = sqrt(obj.x^2 + obj.y^2);
    end
  end
end"""
        s.eval(code)
        s.eval("p = OldPt(3, 4)")
        assert float(s.eval("p.x")) == 3.0
        assert abs(float(s.eval("p.dist()")) - 5.0) < 1e-10

    def test_method_modifies_property(self, s):
        """Method that modifies a property on the object."""
        code = """classdef Acc
  properties
    total = 0
  end
  methods
    function obj = Acc(init)
      obj.total = init;
    end
    function obj = accumulate(obj, val)
      obj.total = obj.total + val;
    end
  end
end"""
        s.eval(code)
        s.eval("a = Acc(0)")
        s.eval("a = a.accumulate(5)")
        s.eval("a = a.accumulate(3)")
        assert float(s.eval("a.total")) == 8.0
