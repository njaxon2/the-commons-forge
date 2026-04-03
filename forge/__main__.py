# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""python -m forge  --  CLI entry point for forge-engine.

Provides a headless REPL when the GUI (forge-ide) is not installed,
or when no Pro/Academic/Enterprise license is active.

Usage:
    forge                       License check -> GUI (pro) or CLI (free)
    forge --cli                 Always launch CLI REPL
    forge --activate KEY        Activate license, then launch GUI if successful
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


def _launch_gui():
    """Launch the PySide6 GUI (forge-ide)."""
    try:
        from forge.gui.main_window import ForgeMainWindow
        from PySide6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = ForgeMainWindow()
        window.show()
        sys.exit(app.exec())
    except ImportError:
        print("error: GUI dependencies not available (PySide6 / forge-ide).")
        print("Install the full IDE:  pip install forge-ide")
        print("Falling back to CLI mode.\n")
        _launch_cli()


def _launch_cli():
    """Launch the interactive CLI REPL."""
    from forge.cli import main as cli_main
    cli_main()


def main():
    """Forge entry point -- routes to GUI, CLI, update, or activation."""
    args = sys.argv[1:]

    # --update: delegate to update subsystem
    if "--update" in args:
        from forge.update import cli_update
        idx = args.index("--update")
        update_args = args[idx + 1:]
        sys.exit(cli_update(update_args))

    # --activate KEY: activate license from command line
    if "--activate" in args:
        idx = args.index("--activate")
        if idx + 1 < len(args):
            key = args[idx + 1]
            from forge.license import get_license_manager
            lm = get_license_manager()
            result = lm.activate(key)
            if result.get("token"):
                print(f"License activated! Tier: {result.get('tier', 'pro')}")
                print("Launching Forge IDE...")
                _launch_gui()
            else:
                print(f"Activation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            print("Usage: forge --activate <license-key>")
            sys.exit(1)
        return

    # --cli: force CLI mode regardless of license
    if "--cli" in args:
        # Kick off background update check
        t = threading.Thread(target=_background_update_check, daemon=True)
        t.start()
        _launch_cli()
        return

    # Default: check license -> GUI (pro) or CLI (free)
    # Kick off background update check
    t = threading.Thread(target=_background_update_check, daemon=True)
    t.start()

    try:
        from forge.license import get_license_manager
        lm = get_license_manager()
        if lm.is_pro:
            _launch_gui()
            return
    except Exception:
        pass  # License check failed; fall through to CLI

    # No active pro license -- start CLI with upgrade message
    print("Forge -- Starting in CLI mode (no active license)")
    print("To activate the visual IDE: forge --activate <your-license-key>")
    print("Purchase a license at thecommons.cc/forge")
    print()
    _launch_cli()


# Keep backward-compatible name used by pyproject.toml entry point
cli_main = main


if __name__ == "__main__":
    main()
