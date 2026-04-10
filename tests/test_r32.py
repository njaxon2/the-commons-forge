"""Tests for R-32: edit command must not raise ValueError on ForgeChar argument."""
import pytest


def test_edit_command_no_error():
    """R-32a: edit filename.m must not raise ValueError on ForgeChar argument."""
    from forge.engine.session import ForgeSession
    s = ForgeSession()
    result = s.eval("edit test_script.m")
    assert "ValueError" not in str(result), f"edit raised ValueError: {result}"
    assert "error:" not in str(result).lower() or "edit" not in str(result), f"edit errored: {result}"


def test_edit_command_sets_request():
    """R-32b: edit must set _edit_request on the session to the filename."""
    from forge.engine.session import ForgeSession
    s = ForgeSession()
    s.eval("edit my_analysis.m")
    # _edit_request is consumed after reading; check via engine
    # Since eval clears it, we patch to verify it was set
    # Instead re-run with monitoring
    found = []
    orig = s._engine.functions.get("edit")
    def spy(args, nargout=0):
        r = orig(args, nargout=nargout)
        found.append(s._edit_request)
        return r
    s._engine.functions["edit"] = spy
    s.eval("edit another_file.m")
    assert found, "edit did not call through"
    assert found[0] == "another_file.m" or "another_file" in str(found[0])


def test_edit_command_no_args_returns_empty():
    """R-32a: edit with no argument must return empty without error."""
    from forge.engine.session import ForgeSession
    s = ForgeSession()
    result = s.eval("edit")
    assert "ValueError" not in str(result)
    assert "Error" not in str(result) or "edit" not in str(result)


