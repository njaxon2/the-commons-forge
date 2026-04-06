# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""
Tests for classdef OOP round 2: inheritance, isa, class(), value/handle
semantics, fieldnames, isobject, nargin-aware constructors.

V-Model Traceability
--------------------
R-OOP-04: The engine SHALL return the correct classdef name from class()
          and support isa() queries against own class, parent class, and
          built-in types.

          Model-user argument: A MATLAB/Octave engineer routinely uses
          class(obj) and isa(obj, 'TypeName') to dispatch behavior or
          validate inputs in polymorphic code. If class() returns an
          internal wrapper name instead of the declared classdef name,
          or if isa() fails on parent types, the user's guard clauses
          and switch/case dispatch break silently.

          Decomposition:
            R-OOP-04a: class(obj) returns the classdef name string.
            R-OOP-04b: isa(obj, 'OwnClass') returns 1.
            R-OOP-04c: isa(obj, 'UnrelatedClass') returns 0.
            R-OOP-04d: isa(scalar, 'double') and isa(string, 'char')
                       return 1 for built-in types.
            R-OOP-04e: isa(subclass_obj, 'ParentClass') returns 1 and
                       isa(subclass_obj, 'SubClass') also returns 1.
            R-OOP-04f: class(subclass_obj) returns the subclass name,
                       not the parent name.

          Consistency: 04a-04c cover the base-class identity contract.
          04d proves isa() still works for non-classdef scalars. 04e-04f
          extend identity queries to inheritance, completing the type
          introspection surface.

R-OOP-05: The engine SHALL support classdef inheritance where a subclass
          declared with "< ParentClass" inherits parent properties and
          methods, and subclass constructors may accept additional
          arguments.

          Model-user argument: Scientists extend a base Point class with
          a ColorPoint that adds a color property while reusing the
          parent's distance() and translate() methods. Without working
          inheritance, they must duplicate every method in the subclass,
          which defeats the purpose of OOP and diverges from Octave
          behavior.

          Decomposition:
            R-OOP-05a: Subclass construction sets both parent and own
                       properties.
            R-OOP-05b: Subclass instances can call inherited parent
                       methods.

          Consistency: 05a proves property inheritance and construction;
          05b proves method inheritance. Together they cover the two
          facets of classdef single inheritance.

R-OOP-06: The engine SHALL support nargin-aware constructors that handle
          zero-argument calls gracefully, using property defaults when
          no arguments are provided.

          Model-user argument: Octave and MATLAB internally call
          constructors with zero arguments during array pre-allocation
          and deserialization. If the constructor crashes on zero args,
          operations like "a(5) = Point(1,2)" fail because the engine
          cannot default-construct the preceding elements.

          Decomposition:
            R-OOP-06a: Point() with zero args yields default property
                       values (x=0, y=0).

          Consistency: A single sub-requirement suffices because the
          nargin guard is the only behavioral branch under test; the
          multi-arg path is already covered by R-OOP-04a and R-OOP-05a.

R-OOP-07: The engine SHALL enforce value semantics for regular classdef
          objects (methods return modified copies) and handle semantics
          for Handle-derived classes (methods mutate the original).

          Model-user argument: An engineer who calls p2 = p.translate(5,10)
          expects p to remain unchanged (value semantics), exactly as in
          Octave. Conversely, a GUI callback counter derived from Handle
          must mutate in place so all references see the updated state.
          Incorrect semantics cause silent data corruption in numerical
          workflows or broken shared-state patterns.

          Decomposition:
            R-OOP-07a: For value classes, translate() returns a new
                       object; the original is unchanged.
            R-OOP-07b: For Handle classes, increment() mutates the
                       original object's count property.

          Consistency: 07a and 07b are the two mutually exclusive
          semantics branches; testing both covers the complete
          value/handle contract.

R-OOP-08: The engine SHALL provide fieldnames() and isobject() builtins
          that correctly introspect classdef objects, including inherited
          properties.

          Model-user argument: Engineers use fieldnames(obj) to
          dynamically iterate over object properties for serialization
          or display, and isobject() to branch between struct and
          classdef handling. Without these, generic I/O routines that
          work in Octave break when ported to Forge.

          Decomposition:
            R-OOP-08a: fieldnames(obj) returns a cell array containing
                       property names of the base class.
            R-OOP-08b: fieldnames(subclass_obj) includes inherited
                       properties alongside subclass-own properties.
            R-OOP-08c: isobject(classdef_obj) returns 1;
                       isobject(scalar) returns 0.

          Consistency: 08a covers base-class introspection, 08b extends
          to inheritance, and 08c provides the boolean type predicate.
          Together they cover the full introspection surface.

R-OOP-09: The engine SHALL support methods that accept extra arguments
          beyond self, methods that return string values, external
          property assignment, independent instance state, and intra-
          object method calls.

          Model-user argument: Real-world classdef usage includes
          translate(dx, dy), to_string(), direct property writes like
          p.x = 99, creating multiple independent instances, and methods
          that call other methods on the same object (e.g., ratio()
          calling area()). These are everyday patterns; any gap here
          blocks practical adoption.

          Decomposition:
            R-OOP-09a: translate(5, 10) passes extra args correctly.
            R-OOP-09b: to_string() returns a formatted string.
            R-OOP-09c: p.x = 99 sets the property from outside.
            R-OOP-09d: Two instances of the same class maintain
                       independent state.
            R-OOP-09e: A method (ratio) can call another method (area)
                       on the same object.

          Consistency: 09a-09e each cover a distinct usage pattern that
          is orthogonal to the others; collectively they span the
          practical surface of everyday classdef interaction.
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
    """Round 2 classdef OOP tests.

    Covers requirements R-OOP-04 through R-OOP-09: type identity and
    isa queries, single inheritance, nargin-aware constructors,
    value vs. handle semantics, fieldnames/isobject introspection,
    and everyday method/property interaction patterns.
    """

    def test_class_returns_classdef_name(self, s):
        """R-OOP-04a: class(obj) returns the declared classdef name."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("cn = class(p)")
        assert str(_ws(s, "cn")) == "Point"

    def test_isa_own_class(self, s):
        """R-OOP-04b: isa(obj, 'OwnClass') returns 1."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval('r = isa(p, "Point")')
        assert float(_ws(s, "r")) == 1.0

    def test_isa_wrong_class(self, s):
        """R-OOP-04c: isa(obj, 'UnrelatedClass') returns 0."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval('r = isa(p, "NotAClass")')
        assert float(_ws(s, "r")) == 0.0

    def test_isa_builtin_type(self, s):
        """R-OOP-04d: isa() returns 1 for built-in double and char types."""
        s.eval('r1 = isa(3.14, "double")')
        assert float(_ws(s, "r1")) == 1.0
        s.eval('r2 = isa("hello", "char")')
        assert float(_ws(s, "r2")) == 1.0

    def test_inheritance_construct(self, s):
        """R-OOP-05a: Subclass construction sets parent and own properties."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        assert float(s.eval("cp.x")) == 1.0
        assert float(s.eval("cp.y")) == 2.0
        assert "blue" in str(s.eval("cp.color"))

    def test_inheritance_method(self, s):
        """R-OOP-05b: Subclass instances can call inherited parent methods."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(3, 4, "green")')
        d = s.eval("cp.distance()")
        assert abs(float(d) - 5.0) < 1e-10

    def test_isa_with_inheritance(self, s):
        """R-OOP-04e: isa(subclass_obj, 'ParentClass') and isa(subclass_obj, 'SubClass') both return 1."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        s.eval('r1 = isa(cp, "Point")')
        s.eval('r2 = isa(cp, "ColorPoint")')
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 1.0

    def test_class_of_subclass(self, s):
        """R-OOP-04f: class(subclass_obj) returns the subclass name, not the parent."""
        s.eval(POINT_DEF)
        s.eval(COLORPOINT_DEF)
        s.eval('cp = ColorPoint(1, 2, "blue")')
        s.eval("cn = class(cp)")
        assert str(_ws(s, "cn")) == "ColorPoint"

    def test_nargin_zero_args_constructor(self, s):
        """R-OOP-06a: Zero-arg constructor uses property defaults (x=0, y=0)."""
        s.eval(POINT_DEF)
        s.eval("p0 = Point()")
        assert float(s.eval("p0.x")) == 0.0
        assert float(s.eval("p0.y")) == 0.0

    def test_value_semantics_no_mutation(self, s):
        """R-OOP-07a: Value class translate() returns a new object; original unchanged."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("p2 = p.translate(10, 20)")
        assert float(s.eval("p.x")) == 3.0
        assert float(s.eval("p.y")) == 4.0
        assert float(s.eval("p2.x")) == 13.0
        assert float(s.eval("p2.y")) == 24.0

    def test_handle_semantics_mutation(self, s):
        """R-OOP-07b: Handle class increment() mutates the original object in place."""
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
        """R-OOP-08a: fieldnames(obj) returns a cell array of property names."""
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
        """R-OOP-08b: fieldnames(subclass_obj) includes inherited properties."""
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
        """R-OOP-08c: isobject returns 1 for classdef objects, 0 for scalars."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("r1 = isobject(p)")
        s.eval("r2 = isobject(42)")
        assert float(_ws(s, "r1")) == 1.0
        assert float(_ws(s, "r2")) == 0.0

    def test_method_with_extra_args(self, s):
        """R-OOP-09a: translate(5, 10) correctly passes extra arguments beyond self."""
        s.eval(POINT_DEF)
        s.eval("p = Point(1, 1)")
        s.eval("p2 = p.translate(5, 10)")
        assert float(s.eval("p2.x")) == 6.0
        assert float(s.eval("p2.y")) == 11.0

    def test_method_returning_string(self, s):
        """R-OOP-09b: to_string() returns a formatted string value."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        result = s.eval('p.to_string()')
        assert "(3, 4)" in str(result)

    def test_property_set_from_outside(self, s):
        """R-OOP-09c: p.x = 99 sets a public property from outside the class."""
        s.eval(POINT_DEF)
        s.eval("p = Point(3, 4)")
        s.eval("p.x = 99")
        assert float(s.eval("p.x")) == 99.0

    def test_multiple_instances_independent(self, s):
        """R-OOP-09d: Two instances of the same class maintain independent state."""
        s.eval(POINT_DEF)
        s.eval("a = Point(1, 2)")
        s.eval("b = Point(10, 20)")
        assert float(s.eval("a.x")) == 1.0
        assert float(s.eval("b.x")) == 10.0
        s.eval("a.x = 999")
        assert float(s.eval("a.x")) == 999.0
        assert float(s.eval("b.x")) == 10.0

    def test_method_calls_other_method(self, s):
        """R-OOP-09e: A method (ratio) can call another method (area) on the same object."""
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
