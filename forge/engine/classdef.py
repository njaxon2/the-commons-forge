"""
Forge classdef support: basic OOP for Octave-compatible class definitions.

Provides ForgeClass, ForgeObject, Property, Method, and a CLASS_REGISTRY
for defining and instantiating classes with inheritance, property access,
method dispatch, and handle vs value semantics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CLASS_REGISTRY: Dict[str, "ForgeClass"] = {}


def register_class(cls: "ForgeClass") -> None:
    """Register a ForgeClass in the global registry."""
    CLASS_REGISTRY[cls.name] = cls


def get_class(name: str) -> "ForgeClass":
    """Look up a class by name. Raises KeyError if not found."""
    if name not in CLASS_REGISTRY:
        raise KeyError(f"Undefined class '{name}'")
    return CLASS_REGISTRY[name]


def class_exists(name: str) -> bool:
    """Return True if *name* is a registered class."""
    return name in CLASS_REGISTRY


def clear_registry() -> None:
    """Remove all registered classes (useful for tests)."""
    CLASS_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Property descriptor
# ---------------------------------------------------------------------------

@dataclass
class Property:
    """Describes a single class property."""

    name: str
    default_value: Any = None
    access: str = "public"          # 'public', 'private', 'protected'
    validation: Optional[Callable[[Any], Any]] = None
    constant: bool = False
    dependent: bool = False
    get_method: Optional[str] = None  # name of getter method for dependent props

    def validate(self, value: Any) -> Any:
        """Run the validation callback (if any) and return the value."""
        if self.validation is not None:
            return self.validation(value)
        return value


# ---------------------------------------------------------------------------
# Method descriptor
# ---------------------------------------------------------------------------

@dataclass
class Method:
    """Describes a single class method."""

    name: str
    function_def: Callable  # the actual Python callable
    is_static: bool = False
    access: str = "public"  # 'public', 'private', 'protected'


# ---------------------------------------------------------------------------
# ForgeClass
# ---------------------------------------------------------------------------

class ForgeClass:
    """
    Holds the *definition* of a class (properties, methods, supers).

    Parameters
    ----------
    name : str
        Class name (e.g. ``"MyFilter"``).
    superclasses : list[str]
        Names of parent classes (looked up in CLASS_REGISTRY at instantiation).
    properties : dict[str, Property]
        Declared properties keyed by name.
    methods : dict[str, Method]
        Declared methods keyed by name.
    is_handle : bool
        If True instances use *handle* (reference) semantics; otherwise they
        use *value* (copy-on-assign) semantics.  Default is False (value).
    """

    def __init__(
        self,
        name: str,
        superclasses: Optional[List[str]] = None,
        properties: Optional[Dict[str, Property]] = None,
        methods: Optional[Dict[str, Method]] = None,
        is_handle: bool = False,
    ) -> None:
        self.name = name
        self.superclasses: List[str] = superclasses or []
        self.properties: Dict[str, Property] = properties or {}
        self.methods: Dict[str, Method] = methods or {}
        self.is_handle = is_handle

    # ----- helpers --------------------------------------------------------

    def all_properties(self) -> Dict[str, Property]:
        """Return merged property dict walking the MRO (depth-first)."""
        merged: Dict[str, Property] = {}
        for parent_name in reversed(self.superclasses):
            parent = get_class(parent_name)
            merged.update(parent.all_properties())
        merged.update(self.properties)
        return merged

    def all_methods(self) -> Dict[str, Method]:
        """Return merged method dict walking the MRO (depth-first)."""
        merged: Dict[str, Method] = {}
        for parent_name in reversed(self.superclasses):
            parent = get_class(parent_name)
            merged.update(parent.all_methods())
        merged.update(self.methods)
        return merged

    def resolve_method(self, name: str) -> Optional[Method]:
        """Find a method by name, searching own methods then ancestors."""
        if name in self.methods:
            return self.methods[name]
        for parent_name in self.superclasses:
            parent = get_class(parent_name)
            m = parent.resolve_method(name)
            if m is not None:
                return m
        return None

    def resolve_property(self, name: str) -> Optional[Property]:
        """Find a property by name, searching own then ancestors."""
        if name in self.properties:
            return self.properties[name]
        for parent_name in self.superclasses:
            parent = get_class(parent_name)
            p = parent.resolve_property(name)
            if p is not None:
                return p
        return None

    def is_subclass_of(self, other_name: str) -> bool:
        """Return True if this class equals or inherits from *other_name*."""
        if self.name == other_name:
            return True
        for parent_name in self.superclasses:
            parent = get_class(parent_name)
            if parent.is_subclass_of(other_name):
                return True
        return False

    # ----- instantiation --------------------------------------------------

    def construct(self, *args: Any, **kwargs: Any) -> "ForgeObject":
        """
        Create a new ForgeObject.

        1. Initialise every property to its default value.
        2. If a constructor method exists (same name as the class), call it.
        """
        obj = ForgeObject(self)

        # Set defaults from full MRO
        for prop_name, prop in self.all_properties().items():
            obj._values[prop_name] = prop.default_value

        # Call constructor if defined
        ctor = self.resolve_method(self.name)
        if ctor is not None:
            result = ctor.function_def(obj, *args, **kwargs)
            # Octave constructors return the object
            if isinstance(result, ForgeObject):
                obj = result

        return obj

    def __repr__(self) -> str:
        supers = ", ".join(self.superclasses) if self.superclasses else "none"
        return (
            f"<ForgeClass '{self.name}' supers=[{supers}] "
            f"props={list(self.properties)} methods={list(self.methods)} "
            f"handle={self.is_handle}>"
        )


# ---------------------------------------------------------------------------
# ForgeObject
# ---------------------------------------------------------------------------

class ForgeObject:
    """
    A runtime instance of a :class:`ForgeClass`.

    Property values are stored in ``_values``.  Access control is enforced on
    ``get`` and ``set`` via the caller context (``_caller_context``), which
    the interpreter should set to ``'internal'`` when executing methods of
    the owning class.
    """

    def __init__(self, forge_class: ForgeClass) -> None:
        self._class: ForgeClass = forge_class
        self._values: Dict[str, Any] = {}
        # The interpreter sets this to bypass access checks inside methods
        self._caller_context: Optional[str] = None

    # ----- property access ------------------------------------------------

    def _check_read_access(self, prop: Property) -> None:
        if prop.access == "private" and self._caller_context != "internal":
            raise AttributeError(
                f"Cannot read private property '{prop.name}' of class "
                f"'{self._class.name}' from outside the class"
            )
        if prop.access == "protected" and self._caller_context != "internal":
            raise AttributeError(
                f"Cannot read protected property '{prop.name}' of class "
                f"'{self._class.name}' from outside the class hierarchy"
            )

    def _check_write_access(self, prop: Property) -> None:
        if prop.constant:
            raise AttributeError(
                f"Cannot modify constant property '{prop.name}' of class "
                f"'{self._class.name}'"
            )
        if prop.access == "private" and self._caller_context != "internal":
            raise AttributeError(
                f"Cannot set private property '{prop.name}' of class "
                f"'{self._class.name}' from outside the class"
            )
        if prop.access == "protected" and self._caller_context != "internal":
            raise AttributeError(
                f"Cannot set protected property '{prop.name}' of class "
                f"'{self._class.name}' from outside the class hierarchy"
            )

    def get(self, name: str) -> Any:
        """Read a property value (with access check)."""
        prop = self._class.resolve_property(name)
        if prop is None:
            raise AttributeError(
                f"'{self._class.name}' has no property '{name}'"
            )
        self._check_read_access(prop)
        if prop.dependent and prop.get_method:
            m = self._class.resolve_method(prop.get_method)
            if m is not None:
                return m.function_def(self)
        return self._values.get(name, prop.default_value)

    def set(self, name: str, value: Any) -> None:
        """Write a property value (with access + validation check)."""
        prop = self._class.resolve_property(name)
        if prop is None:
            raise AttributeError(
                f"'{self._class.name}' has no property '{name}'"
            )
        self._check_write_access(prop)
        value = prop.validate(value)
        self._values[name] = value

    # ----- method dispatch ------------------------------------------------

    def call_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a method on this object."""
        method = self._class.resolve_method(name)
        if method is None:
            raise AttributeError(
                f"'{self._class.name}' has no method '{name}'"
            )
        if method.access == "private" and self._caller_context != "internal":
            raise AttributeError(
                f"Cannot call private method '{name}' of class "
                f"'{self._class.name}' from outside the class"
            )
        if method.is_static:
            return method.function_def(*args, **kwargs)
        # Instance methods receive `self` (the ForgeObject) as first arg
        old_ctx = self._caller_context
        self._caller_context = "internal"
        try:
            return method.function_def(self, *args, **kwargs)
        finally:
            self._caller_context = old_ctx

    # ----- copy / value semantics -----------------------------------------

    def copy(self) -> "ForgeObject":
        """
        Return a deep copy of this object.

        For *value* classes the interpreter should call this on every
        assignment (``b = a`` copies).  For *handle* classes the interpreter
        simply copies the reference.
        """
        new = ForgeObject(self._class)
        new._values = copy.deepcopy(self._values)
        return new

    # ----- identity helpers -----------------------------------------------

    def isa(self, class_name: str) -> bool:
        """Return True if this object is an instance of *class_name* or a subclass."""
        return self._class.is_subclass_of(class_name)

    @property
    def class_name(self) -> str:
        return self._class.name

    # ----- dunder ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for internal attrs
        if name.startswith("_"):
            raise AttributeError(name)
        # Try property first
        prop = self._class.resolve_property(name)
        if prop is not None:
            return self.get(name)
        # Try method — return a bound callable
        method = self._class.resolve_method(name)
        if method is not None:
            if method.is_static:
                return method.function_def
            def _bound(*a: Any, **kw: Any) -> Any:
                return self.call_method(name, *a, **kw)
            return _bound
        raise AttributeError(
            f"'{self._class.name}' has no property or method '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        prop = self._class.resolve_property(name)
        if prop is not None:
            self.set(name, value)
        else:
            raise AttributeError(
                f"'{self._class.name}' has no property '{name}'"
            )

    def __repr__(self) -> str:
        props = ", ".join(
            f"{k}={v!r}" for k, v in self._values.items()
        )
        return f"<ForgeObject '{self._class.name}' {props}>"
