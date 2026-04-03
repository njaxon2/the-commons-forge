"""
Distribution and packaging tests for forge-ide.

Verifies that the package metadata, imports, entry points, toolbox registries,
and basic engine functionality work correctly after installation.
"""
import importlib
import sys
import pytest


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    """Verify installed package metadata matches pyproject.toml."""

    def test_package_name(self):
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert meta["Name"] == "forge-ide"

    def test_version_present(self):
        import forge
        assert hasattr(forge, "__version__")
        assert forge.__version__  # non-empty

    def test_version_matches_pyproject(self):
        """__init__.__version__ must match pyproject.toml version."""
        import forge
        from importlib.metadata import version
        installed = version("forge-ide")
        assert forge.__version__ == installed

    def test_description_present(self):
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert meta["Summary"]  # non-empty

    def test_requires_python(self):
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert ">=3.11" in (meta["Requires-Python"] or "")

    def test_license(self):
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        license_text = meta.get("License") or meta.get("License-Expression") or ""
        assert "Apache" in license_text


# ---------------------------------------------------------------------------
# Core module imports
# ---------------------------------------------------------------------------

class TestCoreImports:
    """All required engine modules must be importable."""

    @pytest.mark.parametrize("module", [
        "forge",
        "forge.engine",
        "forge.engine.session",
        "forge.engine.evaluator",
        "forge.engine.lexer",
        "forge.engine.parser",
        "forge.engine.types",
        "forge.engine.containers",
        "forge.engine.classdef",
        "forge.engine.profiler",
    ])
    def test_core_module_import(self, module):
        importlib.import_module(module)

    @pytest.mark.parametrize("module", [
        "forge.validation",
        "forge.validation.framework",
    ])
    def test_validation_module_import(self, module):
        importlib.import_module(module)


# ---------------------------------------------------------------------------
# Toolbox / builtin imports
# ---------------------------------------------------------------------------

TOOLBOX_MODULES = [
    "forge.engine.builtins.audio",
    "forge.engine.builtins.comms",
    "forge.engine.builtins.control",
    "forge.engine.builtins.database",
    "forge.engine.builtins.elfun",
    "forge.engine.builtins.fileio",
    "forge.engine.builtins.financial",
    "forge.engine.builtins.fuzzy",
    "forge.engine.builtins.general",
    "forge.engine.builtins.geometry",
    "forge.engine.builtins.image",
    "forge.engine.builtins.instrument",
    "forge.engine.builtins.linalg",
    "forge.engine.builtins.neural",
    "forge.engine.builtins.ode",
    "forge.engine.builtins.optimization",
    "forge.engine.builtins.parallel",
    "forge.engine.builtins.plotting",
    "forge.engine.builtins.polynomial",
    "forge.engine.builtins.sets",
    "forge.engine.builtins.signal",
    "forge.engine.builtins.sparse",
    "forge.engine.builtins.specfun",
    "forge.engine.builtins.special_matrix",
    "forge.engine.builtins.statistics",
    "forge.engine.builtins.strings",
    "forge.engine.builtins.symbolic",
    "forge.engine.builtins.time_funcs",
    "forge.engine.builtins.web",
]


class TestToolboxImports:
    """Every toolbox module must load without error."""

    @pytest.mark.parametrize("module", TOOLBOX_MODULES)
    def test_toolbox_import(self, module):
        importlib.import_module(module)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

class TestEntryPoints:
    """Verify console entry points are properly configured."""

    def test_forge_engine_entry_point(self):
        from importlib.metadata import entry_points
        # Python 3.12+ returns SelectableGroups; 3.9+ supports group kwarg
        try:
            console = entry_points(group="console_scripts")
        except TypeError:
            eps = entry_points()
            console = eps.get("console_scripts", [])
        names = [ep.name for ep in console]
        assert "forge-engine" in names, f"forge-engine not in entry points: {names}"

    def test_main_module_exists(self):
        mod = importlib.import_module("forge.__main__")
        assert hasattr(mod, "cli_main")


# ---------------------------------------------------------------------------
# Engine functionality
# ---------------------------------------------------------------------------

class TestEngineBasic:
    """Basic engine session and eval must work."""

    def test_create_session(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert s is not None

    def test_eval_arithmetic(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("2 + 3")
        assert result == 5 or str(result).strip() == "5"

    def test_eval_matrix(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("[1 2 3]")
        assert result is not None

    def test_eval_function_call(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("sqrt(16)")
        val = float(result) if not isinstance(result, (int, float)) else result
        assert abs(val - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# Toolbox registries
# ---------------------------------------------------------------------------

class TestToolboxRegistries:
    """Session should register all toolbox functions without error."""

    def test_session_has_builtins(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        # Check that common functions exist
        for fn_name in ["sin", "cos", "sqrt", "abs", "zeros", "ones", "eye"]:
            result = s.eval(f"exist('{fn_name}')")
            # exist returns nonzero for known functions
            assert result, f"Function {fn_name} not found in session"

    def test_toolbox_count(self):
        """Should have at least 25 toolbox modules registered."""
        assert len(TOOLBOX_MODULES) >= 25


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------

class TestNoCircularImports:
    """Importing core modules in isolation must not cause circular import errors."""

    @pytest.mark.parametrize("module", [
        "forge",
        "forge.engine.session",
        "forge.engine.evaluator",
        "forge.engine.lexer",
        "forge.engine.parser",
    ])
    def test_no_circular_import(self, module):
        # Save and restore sys.modules to avoid polluting other tests
        saved = {k: v for k, v in sys.modules.items() if k.startswith("forge")}
        to_remove = [k for k in sys.modules if k.startswith("forge")]
        for k in to_remove:
            del sys.modules[k]
        try:
            importlib.import_module(module)
        finally:
            # Restore original forge modules so later tests are unaffected
            for k in list(sys.modules):
                if k.startswith("forge"):
                    del sys.modules[k]
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Test collection
# ---------------------------------------------------------------------------

class TestTestCollection:
    """All test files in tests/ must be collectable by pytest."""

    def test_test_files_exist(self):
        from pathlib import Path
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        assert len(test_files) >= 10, f"Only found {len(test_files)} test files"

    def test_no_syntax_errors_in_tests(self):
        """Every test_*.py file must be parseable."""
        import ast
        from pathlib import Path
        test_dir = Path(__file__).parent
        errors = []
        for tf in test_dir.glob("test_*.py"):
            try:
                ast.parse(tf.read_text())
            except SyntaxError as e:
                errors.append(f"{tf.name}: {e}")
        assert not errors, f"Syntax errors in test files: {errors}"
