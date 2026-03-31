# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""
Tests for classdef OOP round 2: inheritance, isa, class(), value/handle
semantics, fieldnames, isobject, nargin-aware constructors.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.classdef import CLASS_REGISTRY, clear_registry


@pytest.fixture
def s():
    """Fresh ForgeSession for each test."""
    clear_registry()
    return ForgeSession()


# ---------------------------------------------------------------------------
# Helper: define the reusable Point and ColorPoint classes
# ---------------------------------------------------------------------------
POINT_DEF = """
classdef Point
    properties
        x = 0
        y = 0
    end
    methods
        function obj = Point(x, y)
            if nargin > 0; obj.x = x; end
            if nargin > 1; obj.y = y; end
        end
        function d = distance(obj)
            d = sqrt(obj.x^2 + obj.y^2);
        end
        function obj = translate(obj, dx, dy)
            obj.x = obj.x + dx;
            obj.y = obj.y + dy;
        end
        function s = to_string(obj)
            s = sprintf("(%g, %g)", obj.x, obj.y);
        end
    end
end
"""

COLORPOINT_DEF = """
classdef ColorPoint < Point
    properties
        color = "red"
    end
    methods
        function obj = ColorPoint(x, y, c)
            if nargin > 0; obj.x = x; end
            if nargin > 1; obj.y = y; end
            if nargin > 2; obj.color = c; end
        end
    end
end
"""


def _ws(s, name):
    """Shortcut to get a raw workspace variable."""
    return s._engine.workspace.get(name)


class TestClassR2:
    """Round 2 classdef OOP tests."""

    def test_class_returns_classdef_name(self, s):
        """class(obj) should return the classdef name, not 'ForgeObject'."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("cn = class(p)")
        assert str(_ws(s, "cn")) == "Point"

    def test_isa_own_class(self, s):
        """isa(p, 'Point') should return true."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval('r = isa(p, "Point")')
        assert float(_ws(s, "r")) == 1.0

    def test_isa_wrong_class(self, s):
        """isa(p, 'SomeOtherClass') should return false."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval('r = isa(p, "NotAClass")')
        assert float(_ws(s, "r")) == 0.0

    def test_isa_builtin_type(self, s):
        """isa on built-in types should still work."""
        s.eval('r1 = isa(3.14, "double")')
        assert float(_ws(s, "r1")) == 1.0
        s.eval('r2 = isa("hello", "char")')
        assert float(_ws(s, "r2")) == 1.0

    def test_inheritance_construct(self, s):
        """Subclass can be instantiated and has parent properties."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        assert float(s.eval("cp.x")) == 1.0
        assert float(s.eval("cp.y")) == 2.0
        assert "blue" in str(s.eval("cp.color"))

    def test_inheritance_method(self, s):
        """Subclass inherits parent methods."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(3, 4, "green")')
        d = s.eval("cp.distance()")
        assert abs(float(d) - 5.0) < 1e-10

    def test_isa_with_inheritance(self, s):
        """isa(subclass_obj, 'ParentClass') should return true."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        s.eval('r1 = isa(cp, "Point")')
        s.eval('r2 = isa(cp, "ColorPoint")')
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 1.0

    def test_class_of_subclass(self, s):
        """class(subclass_obj) should return the subclass name."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        s.eval("cn = class(cp)")
        assert str(_ws(s, "cn")) == "ColorPoint"

    def test_nargin_zero_args_constructor(self, s):
        """Calling constructor with zero args uses property defaults."""
        s.eval(POINT_DEF)
        s.eval("p0 = Point()")
        assert float(s.eval("p0.x")) == 0.0
        assert float(s.eval("p0.y")) == 0.0

    def test_value_semantics_no_mutation(self, s):
        """For value classes, method calls should not mutate the original."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("p2 = p.translate(10, 20)")
        # Original should be unchanged
        assert float(s.eval("p.x")) == 3.0
        assert float(s.eval("p.y")) == 4.0
        # New object should have translated values
        assert float(s.eval("p2.x")) == 13.0
        assert float(s.eval("p2.y")) == 24.0

    def test_handle_semantics_mutation(self, s):
        """For Handle classes, method calls mutate the original object."""
        s.eval("""
classdef (Handle) HCounter
    properties
        count = 0
    end
    methods
        function obj = HCounter()
        end
        function increment(obj)
            obj.count = obj.count + 1;
        end
    end
end
""")
        s.eval("c = HCounter()")
        s.eval("c.increment()")
        assert float(s.eval("c.count")) == 1.0
        s.eval("c.increment()")
        assert float(s.eval("c.count")) == 2.0

    def test_fieldnames_on_object(self, s):
        """fieldnames(obj) returns property names as cell array."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("fn = fieldnames(p)")
        fn = _ws(s, "fn")
        from forge.engine.containers import ForgeCell
        assert isinstance(fn, ForgeCell)
        names = [str(c) for c in fn._data]
        assert "x" in names
        assert "y" in names

    def test_fieldnames_inheritance(self, s):
        """fieldnames on subclass includes inherited properties."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        s.eval("fn = fieldnames(cp)")
        fn = _ws(s, "fn")
        names = [str(c) for c in fn._data]
        assert "x" in names
        assert "y" in names
        assert "color" in names

    def test_isobject(self, s):
        """isobject returns 1 for classdef objects, 0 otherwise."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("r1 = isobject(p)")
        s.eval("r2 = isobject(42)")
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 0.0

    def test_method_with_extra_args(self, s):
        """Method that takes arguments beyond self."""
        s.eval(POINT_DEF)
        s.eval("p = Point(1, 1)")
        s.eval("p2 = p.translate(5, 10)")
        assert float(s.eval("p2.x")) == 6.0
        assert float(s.eval("p2.y")) == 11.0

    def test_method_returning_string(self, s):
        """Method that returns a string value."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        result = s.eval('p.to_string()')
        assert "(3, 4)" in str(result)

    def test_property_set_from_outside(self, s):
        """Setting a public property from outside the class."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("p.x = 99")
        assert float(s.eval("p.x")) == 99.0

    def test_multiple_instances_independent(self, s):
        """Two instances of the same class are independent."""
        s.eval(POINT_DEF)
        s.eval("a = Point(1, 2)")
        s.eval("b = Point(10, 20)")
        assert float(s.eval("a.x")) == 1.0
        assert float(s.eval("b.x")) == 10.0
        s.eval("a.x = 999")
        assert float(s.eval("a.x")) == 999.0
        assert float(s.eval("b.x")) == 10.0

    def test_method_calls_other_method(self, s):
        """A method that internally calls another method on the same object."""
        s.eval("""
classdef Circle
    properties
        radius = 1
    end
    methods
        function obj = Circle(r)
            if nargin > 0; obj.radius = r; end
        end
        function a = area(obj)
            a = pi * obj.radius^2;
        end
        function r = ratio(obj)
            r = obj.area() / (obj.radius * 2);
        end
    end
end
""")
        s.eval("c = Circle(5)")
        a = float(s.eval("c.area()"))
        assert abs(a - np.pi * 25) < 1e-3
        r = float(s.eval("c.ratio()"))
        assert abs(r - (np.pi * 25) / 10) < 1e-3
