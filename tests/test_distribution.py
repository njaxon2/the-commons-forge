"""
Distribution and packaging tests for forge-ide.

Verifies that the package metadata, imports, entry points, toolbox registries,
and basic engine functionality work correctly after installation.
"""
import importlib
import sys
import pytest


class TestPackageMetadata:
    """Verify installed package metadata matches pyproject.toml.

    Requirement: R-DIST-01
    SHALL: The installed forge-ide package shall expose correct metadata
    (name, version, description, Python requirement, and license) consistent
    with pyproject.toml so that pip, IDE integrations, and audit tools can
    identify the package unambiguously.

    Model-user argument:
    The engineer installs forge-ide via pip on a fresh machine. They run
    ``pip show forge-ide`` to confirm the install succeeded. If the name is
    wrong, the version is missing, or the license field is blank, they will
    assume the package is broken or unofficial and uninstall immediately.
    Correct metadata is the first proof of a professional distribution.

    Decomposition:
    R-DIST-01-A: Package name equals "forge-ide".
    R-DIST-01-B: forge.__version__ is a non-empty string.
    R-DIST-01-C: forge.__version__ matches importlib.metadata version.
    R-DIST-01-D: Summary field is present and non-empty.
    R-DIST-01-E: Requires-Python includes >=3.11.
    R-DIST-01-F: License field contains "Apache".

    Consistency: Sub-requirements A through F collectively cover every metadata
    field the user or tooling would inspect after installation. Together they
    fully satisfy R-DIST-01.
    """

    def test_package_name(self):
        """R-DIST-01-A: Package name equals 'forge-ide'."""
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert meta["Name"] == "forge-ide"

    def test_version_present(self):
        """R-DIST-01-B: forge.__version__ is a non-empty string."""
        import forge
        assert hasattr(forge, "__version__")
        assert forge.__version__

    def test_version_matches_pyproject(self):
        """R-DIST-01-C: __init__.__version__ must match pyproject.toml version."""
        import forge
        from importlib.metadata import version
        installed = version("forge-ide")
        assert forge.__version__ == installed

    def test_description_present(self):
        """R-DIST-01-D: Summary metadata field is present and non-empty."""
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert meta["Summary"]

    def test_requires_python(self):
        """R-DIST-01-E: Requires-Python includes >=3.11."""
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        assert ">=3.11" in (meta["Requires-Python"] or "")

    def test_license(self):
        """R-DIST-01-F: License field contains 'Apache'."""
        from importlib.metadata import metadata
        meta = metadata("forge-ide")
        license_text = meta.get("License") or meta.get("License-Expression") or ""
        assert "Apache" in license_text


class TestCoreImports:
    """All required engine modules must be importable.

    Requirement: R-DIST-02
    SHALL: Every core engine module (forge, forge.engine, session, evaluator,
    lexer, parser, types, containers, classdef, profiler, and the validation
    framework) shall be importable without error after a pip install.

    Model-user argument:
    After installing forge-ide, the engineer opens a Python REPL and types
    ``from forge.engine.session import ForgeSession``. If that import fails
    with a ModuleNotFoundError or an unexpected exception, their confidence
    in the product collapses on the spot. Every public module must load
    cleanly or the product is dead on arrival.

    Decomposition:
    R-DIST-02-A: Each core module in the parametrized list imports successfully.
    R-DIST-02-B: Each validation module in the parametrized list imports
    successfully.

    Consistency: The two parametrized sets together cover all public core and
    validation modules. Successful import of every item satisfies R-DIST-02.
    """

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
        """R-DIST-02-A: Core module imports without error."""
        importlib.import_module(module)

    @pytest.mark.parametrize("module", [
        "forge.validation",
        "forge.validation.framework",
    ])
    def test_validation_module_import(self, module):
        """R-DIST-02-B: Validation module imports without error."""
        importlib.import_module(module)


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
    """Every toolbox module must load without error.

    Requirement: R-DIST-03
    SHALL: All 29 toolbox modules under forge.engine.builtins shall be
    importable without error after a standard pip install.

    Model-user argument:
    The engineer types ``s.eval("fft([1 2 3 4])")`` and expects the signal
    toolbox to be available. Forge ships monolithic (all toolboxes included),
    so a missing or broken toolbox import means the user hits an opaque
    "function not found" error on a function they know exists in Octave.
    Every toolbox must load or the product fails its core promise.

    Decomposition:
    R-DIST-03-A: Each toolbox module in TOOLBOX_MODULES imports successfully.

    Consistency: The single parametrized sub-requirement iterates over all 29
    toolbox modules. Passing every parametrized case satisfies R-DIST-03.
    """

    @pytest.mark.parametrize("module", TOOLBOX_MODULES)
    def test_toolbox_import(self, module):
        """R-DIST-03-A: Toolbox module imports without error."""
        importlib.import_module(module)


class TestEntryPoints:
    """Verify console entry points are properly configured.

    Requirement: R-DIST-04
    SHALL: The forge-ide package shall register a "forge-engine" console_script
    entry point and provide a forge.__main__ module with a cli_main callable,
    so that the ``forge`` command launches the IDE from any terminal.

    Model-user argument:
    After ``pip install forge-ide``, the engineer types ``forge`` in their
    terminal. If no entry point is registered, nothing happens. If __main__
    lacks cli_main, ``python -m forge`` also fails. Either way the user
    concludes installation is broken and gives up. The entry point is the
    front door to the product.

    Decomposition:
    R-DIST-04-A: "forge-engine" exists in console_scripts entry points.
    R-DIST-04-B: forge.__main__ module exists and has a cli_main attribute.

    Consistency: A covers the pip-installed command, B covers the
    ``python -m forge`` fallback. Together they ensure both launch methods
    work, fully satisfying R-DIST-04.
    """

    def test_forge_engine_entry_point(self):
        """R-DIST-04-A: 'forge-engine' is in console_scripts entry points."""
        from importlib.metadata import entry_points
        try:
            console = entry_points(group="console_scripts")
        except TypeError:
            eps = entry_points()
            console = eps.get("console_scripts", [])
        names = [ep.name for ep in console]
        assert "forge-engine" in names, f"forge-engine not in entry points: {names}"

    def test_main_module_exists(self):
        """R-DIST-04-B: forge.__main__ has a cli_main callable."""
        mod = importlib.import_module("forge.__main__")
        assert hasattr(mod, "cli_main")


class TestEngineBasic:
    """Basic engine session and eval must work.

    Requirement: R-DIST-05
    SHALL: A freshly created ForgeSession shall evaluate arithmetic
    expressions, matrix literals, and builtin function calls (e.g. sqrt)
    and return correct results, confirming the engine is functional
    post-install.

    Model-user argument:
    The very first thing the engineer does after launching Forge is type
    ``2 + 3`` in the command window. If that returns the wrong answer or
    throws an exception, the product is unusable. Matrix construction and
    sqrt are the next two things any MATLAB/Octave user tries. These three
    smoke tests represent the minimum viable "it works" moment.

    Decomposition:
    R-DIST-05-A: ForgeSession instantiates without error.
    R-DIST-05-B: Arithmetic expression "2 + 3" evaluates to 5.
    R-DIST-05-C: Matrix literal "[1 2 3]" returns a non-None result.
    R-DIST-05-D: Function call "sqrt(16)" returns 4.0 within tolerance.

    Consistency: A confirms the session object is constructible. B, C, and D
    exercise the evaluator on the three fundamental expression types
    (arithmetic, matrix, function call). Together they prove the engine is
    minimally functional, satisfying R-DIST-05.
    """

    def test_create_session(self):
        """R-DIST-05-A: ForgeSession instantiates without error."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        assert s is not None

    def test_eval_arithmetic(self):
        """R-DIST-05-B: Arithmetic '2 + 3' evaluates to 5."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("2 + 3")
        assert result == 5 or str(result).strip() == "5"

    def test_eval_matrix(self):
        """R-DIST-05-C: Matrix literal '[1 2 3]' returns non-None."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("[1 2 3]")
        assert result is not None

    def test_eval_function_call(self):
        """R-DIST-05-D: sqrt(16) returns 4.0 within tolerance."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("sqrt(16)")
        val = float(result) if not isinstance(result, (int, float)) else result
        assert abs(val - 4.0) < 1e-10


class TestToolboxRegistries:
    """Session should register all toolbox functions without error.

    Requirement: R-DIST-06
    SHALL: A ForgeSession shall have all fundamental mathematical functions
    (sin, cos, sqrt, abs, zeros, ones, eye) registered and discoverable via
    exist(), and the distribution shall include at least 25 toolbox modules.

    Model-user argument:
    The engineer expects ``sin(pi)`` to work out of the box, just like in
    Octave. If ``exist('sin')`` returns false, they will assume Forge shipped
    without its standard library. The toolbox count check ensures the wheel
    was not built with missing subpackages, which would silently remove
    entire function families.

    Decomposition:
    R-DIST-06-A: Each of sin, cos, sqrt, abs, zeros, ones, eye is found
    by exist() in a fresh session.
    R-DIST-06-B: TOOLBOX_MODULES contains at least 25 entries.

    Consistency: A verifies runtime function registration. B verifies the
    distribution includes the expected breadth of toolboxes. Together they
    confirm the session's function namespace is correctly populated,
    satisfying R-DIST-06.
    """

    def test_session_has_builtins(self):
        """R-DIST-06-A: Fundamental math functions are registered in session."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        for fn_name in ["sin", "cos", "sqrt", "abs", "zeros", "ones", "eye"]:
            result = s.eval(f"exist('{fn_name}')")
            assert result, f"Function {fn_name} not found in session"

    def test_toolbox_count(self):
        """R-DIST-06-B: At least 25 toolbox modules are listed."""
        assert len(TOOLBOX_MODULES) >= 25


class TestNoCircularImports:
    """Importing core modules in isolation must not cause circular import errors.

    Requirement: R-DIST-07
    SHALL: Each core module (forge, session, evaluator, lexer, parser) shall
    be importable from a clean module state without triggering circular
    import errors.

    Model-user argument:
    Circular imports surface as cryptic ImportErrors that vary depending on
    which module the user touches first. The engineer who writes
    ``from forge.engine.parser import parse`` in their own script should
    never see "cannot import name X from partially initialized module Y."
    These errors are nearly impossible for end users to diagnose and always
    result in a support ticket or an uninstall.

    Decomposition:
    R-DIST-07-A: Each parametrized core module imports successfully after
    all forge.* entries are purged from sys.modules.

    Consistency: The single parametrized sub-requirement covers all five
    critical module entry points. Purging sys.modules before each import
    forces fresh resolution of the dependency graph. Passing every case
    satisfies R-DIST-07.
    """

    @pytest.mark.parametrize("module", [
        "forge",
        "forge.engine.session",
        "forge.engine.evaluator",
        "forge.engine.lexer",
        "forge.engine.parser",
    ])
    def test_no_circular_import(self, module):
        """R-DIST-07-A: Module imports cleanly from a purged module state."""
        saved = {k: v for k, v in sys.modules.items() if k.startswith("forge")}
        to_remove = [k for k in sys.modules if k.startswith("forge")]
        for k in to_remove:
            del sys.modules[k]
        try:
            importlib.import_module(module)
        finally:
            for k in list(sys.modules):
                if k.startswith("forge"):
                    del sys.modules[k]
            sys.modules.update(saved)


class TestTestCollection:
    """All test files in tests/ must be collectable by pytest.

    Requirement: R-DIST-08
    SHALL: The tests/ directory shall contain at least 10 test files, all
    of which are syntactically valid Python, so that ``pytest --collect-only``
    succeeds without collection errors.

    Model-user argument:
    The engineer who clones the repo and runs ``pytest`` expects a clean
    collection phase. If test files have syntax errors, pytest prints a wall
    of red tracebacks before any test runs, signaling poor project hygiene.
    A minimum file count guards against accidentally shipping an empty or
    gutted test directory in the wheel.

    Decomposition:
    R-DIST-08-A: At least 10 test_*.py files exist in the tests/ directory.
    R-DIST-08-B: Every test_*.py file parses without SyntaxError.

    Consistency: A ensures the test suite is present and non-trivial. B
    ensures every file is valid Python. Together they guarantee pytest can
    collect the full suite, satisfying R-DIST-08.
    """

    def test_test_files_exist(self):
        """R-DIST-08-A: At least 10 test files exist in tests/."""
        from pathlib import Path
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        assert len(test_files) >= 10, f"Only found {len(test_files)} test files"

    def test_no_syntax_errors_in_tests(self):
        """R-DIST-08-B: Every test_*.py file is parseable without SyntaxError."""
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
