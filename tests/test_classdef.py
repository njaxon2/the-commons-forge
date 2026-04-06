# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for classdef OOP and unwind_protect.

V-Model Traceability
====================
R-OOP-01: Classdef definition and property defaults
R-OOP-02: Classdef construction and property access
R-OOP-03: Classdef method invocation
R-OOP-04: Classdef property mutation after construction
R-OOP-05: unwind_protect cleanup execution
R-OOP-06: Extended classdef patterns (constructors, multi-method, returns)
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestClassdef:
    """Test basic classdef OOP support.

    Requirement R-OOP-01: The engine SHALL parse a classdef block and register
    the class so it is available for construction.

    Model-user argument: The engineer defines classdef blocks inline in scripts
    to model structured data (e.g., Point with x, y). They expect the class to
    be recognized immediately after evaluation, just as it would be in Octave.
    Without this, the entire classdef feature is inert.

    Requirement R-OOP-02: The engine SHALL construct an instance of a registered
    classdef by calling ClassName(args) and populate properties via the
    constructor.

    Model-user argument: After defining a Point class with a constructor, the
    engineer writes p = Pt(3, 4) and expects p.x to be 3. This is the most
    basic OOP operation and the foundation for all further classdef usage.

    Requirement R-OOP-03: The engine SHALL dispatch method calls on classdef
    instances using obj.method() syntax and return computed results.

    Model-user argument: The engineer defines a dist() method on a Point class
    and calls p.dist(). They expect the method body to access obj.x and obj.y
    and return the Euclidean distance. Without method dispatch, classdef offers
    no advantage over plain structs.

    Requirement R-OOP-04: The engine SHALL allow property assignment on classdef
    instances using obj.prop = value syntax after construction.

    Model-user argument: The engineer constructs a Box with default dimensions,
    then writes b.width = 5 to resize it. Mutable properties are essential for
    iterative workflows where objects are configured after creation.

    Decomposition:
      R-OOP-01.1 Parse classdef block and register class
      R-OOP-01.2 Verify property defaults are set on construction
      R-OOP-02.1 Construct instance with positional arguments
      R-OOP-03.1 Call zero-argument method and verify return value
      R-OOP-04.1 Assign to property after construction and verify new value

    Consistency argument: R-OOP-01.1 ensures the class is parseable and
    registered. R-OOP-01.2 verifies defaults propagate. R-OOP-02.1 verifies
    constructor argument binding. R-OOP-03.1 verifies method dispatch and
    return. R-OOP-04.1 verifies post-construction mutation. Together these
    cover the full lifecycle: define, construct, query, mutate.
    """

    def test_define_class(self, s):
        """R-OOP-01.1: Parse classdef block and register the class name."""
        code = "classdef MyClass\n  properties\n    x = 0\n    y = 0\n  end\nend\n"
        s.eval(code)
        from forge.engine.classdef import class_exists
        assert class_exists("MyClass")

    def test_construct_instance(self, s):
        """R-OOP-02.1: Construct instance via ClassName(args), bind constructor params to properties."""
        code = "classdef Pt\n  properties\n    x = 0\n    y = 0\n  end\n  methods\n    function obj = Pt(obj, x, y)\n      obj.x = x;\n      obj.y = y;\n    end\n  end\nend\n"
        s.eval(code)
        s.eval("p = Pt(3, 4)")
        x = s.eval("p.x")
        assert float(x) == 3.0

    def test_method_call(self, s):
        """R-OOP-03.1: Dispatch obj.method() and return computed scalar result."""
        code = "classdef Pt2\n  properties\n    x = 0\n    y = 0\n  end\n  methods\n    function obj = Pt2(obj, x, y)\n      obj.x = x;\n      obj.y = y;\n    end\n    function d = dist(obj)\n      d = sqrt(obj.x^2 + obj.y^2);\n    end\n  end\nend\n"
        s.eval(code)
        s.eval("p = Pt2(3, 4)")
        d = s.eval("p.dist()")
        assert abs(float(d) - 5.0) < 1e-10

    def test_property_default(self, s):
        """R-OOP-01.2: Properties initialize to declared defaults when no constructor args given."""
        code = "classdef Counter\n  properties\n    count = 0\n  end\nend\n"
        s.eval(code)
        s.eval("c = Counter()")
        count = s.eval("c.count")
        assert float(count) == 0.0

    def test_property_assignment(self, s):
        """R-OOP-04.1: Assign new value to property via obj.prop = value after construction."""
        code = "classdef Box\n  properties\n    width = 1\n    height = 1\n  end\nend\n"
        s.eval(code)
        s.eval("b = Box()")
        s.eval("b.width = 5")
        w = s.eval("b.width")
        assert float(w) == 5.0


class TestUnwindProtect:
    """Test unwind_protect/unwind_protect_cleanup.

    Requirement R-OOP-05: The engine SHALL execute the unwind_protect_cleanup
    block unconditionally after the unwind_protect body completes, whether the
    body raised an error or not, and the cleanup block SHALL have access to
    workspace variables set during the body.

    Model-user argument: The engineer wraps risky file I/O or temporary state
    changes in unwind_protect so that cleanup (closing handles, restoring
    defaults) always runs. In Octave, this is the standard idiom for resource
    safety. If cleanup does not run after an error, the engineer's workspace
    is left in a corrupted state, breaking subsequent calculations.

    Decomposition:
      R-OOP-05.1 Cleanup runs after error in body; body-set variables visible
      R-OOP-05.2 Cleanup runs after normal body completion
      R-OOP-05.3 Cleanup can read variables from enclosing workspace

    Consistency argument: R-OOP-05.1 covers the error path (the primary use
    case). R-OOP-05.2 covers the normal path (cleanup must be unconditional).
    R-OOP-05.3 verifies that workspace variables set before the block are
    accessible inside both body and cleanup. Together these confirm that
    unwind_protect behaves as a reliable finally-equivalent.
    """

    def test_cleanup_after_error(self, s):
        """R-OOP-05.1: Cleanup block runs after error; body-set variable visible in cleanup."""
        code = "x = 0\nunwind_protect\n  x = 1\n  error('test')\nunwind_protect_cleanup\n  x = x + 10\nend"
        s.eval(code)
        assert float(s.eval("x")) == 11.0

    def test_cleanup_without_error(self, s):
        """R-OOP-05.2: Cleanup block runs after normal body completion."""
        code = "y = 5\nunwind_protect\n  y = y * 2\nunwind_protect_cleanup\n  y = y + 100\nend"
        s.eval(code)
        assert float(s.eval("y")) == 110.0

    def test_cleanup_preserves_workspace(self, s):
        """R-OOP-05.3: Enclosing workspace variables accessible in body and cleanup."""
        s.eval("a = 1; b = 2")
        code = "unwind_protect\nc = a + b\nunwind_protect_cleanup\nd = c * 10\nend"
        s.eval(code)
        assert float(s.eval("d")) == 30.0


class TestClassdefExpanded:
    """Extended classdef tests: new-style constructors, multi-method, returns.

    Requirement R-OOP-06: The engine SHALL support the full range of classdef
    constructor styles (new-style without obj in params, old-style with obj in
    params, and implicit default constructor), methods that accept additional
    arguments, methods that return new objects of the same class, methods that
    mutate and return the modified object, and classes with multiple methods.

    Model-user argument: Real engineering classdef usage goes beyond simple
    construction and property access. The engineer defines a Vec3 with a
    new-style constructor, an Adder whose method takes extra arguments, a Pt3
    whose scale() method returns a new Pt3, and a Rect with area/perimeter/
    diagonal methods. These patterns appear constantly in geometry, signal
    processing, and data modeling code. If any of these patterns fail, the
    engineer cannot port their existing Octave class libraries to Forge.

    Decomposition:
      R-OOP-06.1 New-style constructor (obj not in params) populates properties
      R-OOP-06.2 Method accepting arguments beyond self returns correct result
      R-OOP-06.3 Method constructs and returns a new instance of the same class
      R-OOP-06.4 Implicit default constructor sets property defaults
      R-OOP-06.5 Multiple methods on one class all dispatch correctly
      R-OOP-06.6 Old-style constructor (obj in params) populates properties
      R-OOP-06.7 Method modifies property and returns updated object

    Consistency argument: R-OOP-06.1 and R-OOP-06.6 cover both constructor
    conventions. R-OOP-06.4 covers the no-constructor case. R-OOP-06.2 tests
    argument passing beyond self. R-OOP-06.3 tests object-returning methods.
    R-OOP-06.5 tests multi-method dispatch. R-OOP-06.7 tests mutating methods
    that return the modified object (value semantics). Together these span the
    full surface of classdef method and constructor patterns used in practice.
    """

    def test_new_style_constructor(self, s):
        """R-OOP-06.1: New-style constructor (obj not in params) populates all properties."""
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
        """R-OOP-06.2: Method with extra argument beyond self returns correct sum."""
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
        """R-OOP-06.3: Method constructs and returns a new instance of the same class."""
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
        """R-OOP-06.4: Class with no explicit constructor uses defaults for all properties."""
        code = """classdef Simple
  properties
    val = 42
  end
end"""
        s.eval(code)
        s.eval("s = Simple()")
        assert float(s.eval("s.val")) == 42.0

    def test_multiple_methods(self, s):
        """R-OOP-06.5: Multiple methods on one class all dispatch and return correctly."""
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
        """R-OOP-06.6: Old-style constructor (obj in params) populates properties correctly."""
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
        """R-OOP-06.7: Mutating method modifies property and returns updated object."""
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
