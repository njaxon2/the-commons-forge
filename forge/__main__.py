# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""python -m forge  --  CLI entry point for forge-engine.

Provides a headless REPL when the GUI (forge-ide) is not installed.

Update commands:
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


def cli_main():
    """Minimal interactive REPL for forge-engine (no GUI)."""
    # Handle --update flag before starting the REPL
    if "--update" in sys.argv:
        from forge.update import cli_update
        # Pass everything after --update to the update CLI
        idx = sys.argv.index("--update")
        update_args = sys.argv[idx + 1:]
        sys.exit(cli_update(update_args))

    from forge.engine.session import ForgeSession

    # Kick off a background auto-update check (non-blocking)
    t = threading.Thread(target=_background_update_check, daemon=True)
    t.start()

    session = ForgeSession()
    print(f"Forge Engine {session.__class__.__module__} — headless REPL")
    print("Type expressions or exit to quit.\n")

    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("exit", "quit"):
            break
        try:
            result = session.eval(line)
            if result is not None:
                print(result)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    cli_main()
