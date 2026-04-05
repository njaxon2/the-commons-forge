# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""python -m forge  --  CLI entry point for forge-engine.

Provides a headless REPL for the Forge computation engine.
For the visual IDE, install forge-ide (thecommons.cc/forge).

Usage:
    forge                       Launch CLI REPL
    forge script.m              Run a script file
    forge --update              Check and update with validation
    forge --update --force      Skip validation
    forge --update --enable     Enable auto-updates
    forge --update --disable    Disable auto-updates
    forge --update --status     Show update settings and status
    forge --update --check-only Only check, do not apply
"""
import sys
import threading


def _background_update_check():
    """Run an auto-update check in a background thread (if enabled)."""
    try:
        from forge.update import check_and_update
        check_and_update(auto=True, on_progress=lambda m: None)
    except Exception:
        pass  # Never crash the main process for an update check


def _launch_cli():
    """Launch the interactive CLI REPL."""
    from forge.cli import main as cli_main
    cli_main()


def main():
    """Forge engine entry point: routes to CLI or update subsystem."""
    args = sys.argv[1:]

    # --update: delegate to update subsystem
    if "--update" in args:
        from forge.update import cli_update
        idx = args.index("--update")
        update_args = args[idx + 1:]
        sys.exit(cli_update(update_args))

    # Kick off background update check
    t = threading.Thread(target=_background_update_check, daemon=True)
    t.start()

    _launch_cli()


# Keep backward-compatible name used by pyproject.toml entry point
cli_main = main


if __name__ == "__main__":
    main()
