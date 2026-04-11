"""R-39: Error messages shall use Octave vocabulary without Python exception class prefixes.

Requirement: The error display pipeline shall strip Python exception class name
prefixes (NameError:, ValueError:, TypeError:, etc.) from all user-facing error
messages, and element-wise operator failures shall produce Octave-style
nonconformant arguments messages.

Model-user argument: The golden user is an engineer who thinks in MATLAB/Octave
terms, not Python. When they accidentally write [1 2 3] + [1 2], they need to
read "nonconformant arguments (op1 is 1x3, op2 is 1x2)" to immediately
understand what went wrong. Seeing "ValueError: operands could not be broadcast
together" adds cognitive friction and breaks the Octave-compatibility contract
precisely when feedback matters most.
"""
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    """Fresh session for each test."""
    return ForgeSession()


def _err(session, expr):
    """Evaluate and return the error string, or empty string if no error."""
    r = session.eval(expr)
    if isinstance(r, str) and r.startswith("error:"):
        return r
    return ""


class TestNoPythonPrefix:
    """R-39.1: Error messages shall NOT contain Python exception class name prefixes.

    Decomposition sub-requirement: every error message returned by session.eval()
    must start with 'error: ' followed directly by the error content, never with
    'error: NameError:', 'error: ValueError:', 'error: TypeError:', etc.
    """

    def test_r39_undefined_var_no_python_prefix(self, S):
        """R-39.1: Undefined variable error shall not include 'NameError:' prefix.

        When accessing an undefined variable, the message must not expose the
        Python NameError class name. The user only needs to know the variable
        is undefined.
        """
        r = _err(S, "undefined_var_xyz + 1")
        assert r, "Expected an error for undefined variable"
        assert "NameError" not in r, (
            f"Python exception class name leaked into error message: {r}"
        )
        assert "undefined" in r.lower(), f"Expected 'undefined' in message: {r}"

    def test_r39_dimension_mismatch_no_python_prefix(self, S):
        """R-39.1: Dimension mismatch error shall not include 'ValueError:' prefix.

        When adding vectors of incompatible lengths, the message must not expose
        the Python ValueError class name.
        """
        r = _err(S, "[1 2 3] + [1 2]")
        assert r, "Expected an error for dimension mismatch addition"
        assert "ValueError" not in r, (
            f"Python exception class name leaked into error message: {r}"
        )

    def test_r39_no_python_prefix_generic(self, S):
        """R-39.1: Generic errors shall not expose Python exception class names.

        A sweep across common exception types confirms the pipeline strips them all.
        """
        python_prefixes = ["NameError:", "ValueError:", "TypeError:", "IndexError:"]
        test_cases = [
            ("undef_abc()", "NameError"),
            ("[1 2 3] + [1 2]", "ValueError"),
        ]
        for expr, _ in test_cases:
            r = _err(S, expr)
            if r:
                for prefix in python_prefixes:
                    assert prefix not in r, (
                        f"Python prefix '{prefix}' found in error for {expr!r}: {r}"
                    )


class TestNonconformantMessages:
    """R-39.2: Element-wise operator failures shall use Octave nonconformant vocabulary.

    Decomposition sub-requirement: when + or - is applied to arrays of
    incompatible shapes, the error message must contain 'nonconformant arguments'
    with the actual operand shapes in (op1 is RxC, op2 is RxC) format.
    """

    def test_r39_add_row_vectors_nonconformant(self, S):
        """R-39.2: [1 2 3] + [1 2] shall report nonconformant arguments with shapes.

        The golden user typed shapes wrong. The message must tell them op1 is 1x3
        and op2 is 1x2 so they can immediately identify which operand to fix.
        """
        r = _err(S, "[1 2 3] + [1 2]")
        assert r, "Expected an error for incompatible addition"
        assert "nonconformant" in r.lower(), (
            f"Expected 'nonconformant' in message, got: {r}"
        )
        assert "1x3" in r or "1x2" in r, (
            f"Expected shapes in message, got: {r}"
        )

    def test_r39_sub_nonconformant(self, S):
        """R-39.2: [1 2] - [1 2 3] shall report nonconformant arguments with shapes.

        Subtraction with incompatible shapes must also produce the Octave message,
        not raw NumPy broadcast language.
        """
        r = _err(S, "[1 2] - [1 2 3]")
        assert r, "Expected an error for incompatible subtraction"
        assert "nonconformant" in r.lower(), (
            f"Expected 'nonconformant' in message, got: {r}"
        )

    def test_r39_add_matrix_nonconformant(self, S):
        """R-39.2: A (2x3) + B (3x2) shall report nonconformant arguments with shapes.

        Matrix addition with mismatched shapes must use Octave vocabulary.
        """
        r = _err(S, "[1 2 3; 4 5 6] + [1 2; 3 4; 5 6]")
        assert r, "Expected an error for incompatible matrix addition"
        assert "nonconformant" in r.lower(), (
            f"Expected 'nonconformant' in message, got: {r}"
        )
        assert "2x3" in r or "3x2" in r, (
            f"Expected shapes in message, got: {r}"
        )

    def test_r39_no_broadcast_language(self, S):
        """R-39.2: Error messages shall not contain NumPy broadcast vocabulary.

        'broadcast', 'operands could not be broadcast' etc. are NumPy internals
        that must never appear in user-facing messages.
        """
        r = _err(S, "[1 2 3] + [1 2]")
        assert r, "Expected an error"
        assert "broadcast" not in r.lower(), (
            f"NumPy broadcast language leaked into error message: {r}"
        )
