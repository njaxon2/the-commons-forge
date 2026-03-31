# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""python -m forge  --  CLI entry point for forge-engine.

Provides a headless REPL when the GUI (forge-ide) is not installed.
"""
import sys


def cli_main():
    """Minimal interactive REPL for forge-engine (no GUI)."""
    from forge.engine.session import ForgeSession

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
