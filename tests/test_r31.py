"""Tests for R-31a: Common functions have multi-line help documentation."""
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOP_FUNCTIONS = [
    "plot", "disp", "fprintf", "linspace", "zeros", "ones", "eye",
    "rand", "size", "length", "sum", "max", "min", "abs",
    "sin", "cos", "sqrt", "exp", "log", "fft", "find", "sort",
    "floor", "ceil", "numel",
]


@pytest.fixture(scope="module")
def session():
    from forge.engine.session import ForgeSession
    return ForgeSession()


@pytest.mark.parametrize("fname", TOP_FUNCTIONS)
def test_function_has_multiline_doc(session, fname):
    """Each target function must have 3+ non-empty lines of documentation (R-31a)."""
    func = session._engine.functions.get(fname)
    assert func is not None, f"{fname} not registered"
    doc = func.__doc__ or ""
    lines = [ln for ln in doc.splitlines() if ln.strip()]
    assert len(lines) >= 3, (
        f"help({fname!r}) has {len(lines)} line(s), need >= 3. "
        f"Current: {doc[:120]!r}"
    )
