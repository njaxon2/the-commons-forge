# Copyright 2026 The Commons(TM)
# SPDX-License-Identifier: Apache-2.0
"""
Forge CLI -- Interactive command-line REPL for the Forge computation engine.
Full Octave-compatible engine at a >> prompt.
"""
import sys
import os
import atexit

# ANSI color codes
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _setup_readline(session):
    """Configure readline for history and tab completion if available."""
    try:
        import readline
    except ImportError:
        return  # Windows without pyreadline3, etc.

    # History file
    history_path = os.path.expanduser("~/.forge_history")
    try:
        readline.read_history_file(history_path)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(5000)
    atexit.register(readline.write_history_file, history_path)

    # Tab completion from session functions
    _func_names = sorted(session._engine.functions.keys())

    def completer(text, state):
        matches = [n for n in _func_names if n.startswith(text)]
        # Also complete workspace variable names
        for vname in session.workspace:
            if vname.startswith(text) and vname not in matches:
                matches.append(vname)
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n;,()[]{}+-*/=<>~!@#$%^&|")
    readline.parse_and_bind("tab: complete")


def _needs_continuation(line, buffer):
    """Return True if the input is incomplete and needs more lines."""
    # Explicit continuation: line ends with ...
    if line.rstrip().endswith("..."):
        return True
    # Check unclosed brackets/parens/braces in the accumulated buffer
    combined = buffer + "\n" + line if buffer else line
    opens = combined.count("(") - combined.count(")")
    opens += combined.count("[") - combined.count("]")
    opens += combined.count("{") - combined.count("}")
    return opens > 0


def _strip_continuation(line):
    """Remove trailing ... continuation marker."""
    s = line.rstrip()
    if s.endswith("..."):
        return s[:-3]
    return line


def main():
    """Run the Forge interactive CLI REPL."""
    from forge import __version__
    from forge.engine.session import ForgeSession

    # Determine license tier for banner
    tier = "free"
    try:
        from forge.license import get_license_manager
        lm = get_license_manager()
        tier = lm.tier
    except Exception:
        pass

    session = ForgeSession()
    _setup_readline(session)

    # Banner
    print(f"{_BOLD}Forge {__version__}{_RESET} -- Octave-compatible computation engine")
    print('Type "help" for help, "exit" to quit.')
    if tier == "free":
        print("To unlock the visual IDE, visit thecommons.cc/forge")
    print()

    # REPL loop
    while True:
        buf = ""
        prompt = ">> "
        try:
            while True:
                line = input(prompt)

                # Check for continuation
                if _needs_continuation(line, buf):
                    clean = _strip_continuation(line)
                    buf = (buf + "\n" + clean) if buf else clean
                    prompt = ".. "
                    continue

                # Complete line
                if buf:
                    code = buf + "\n" + line
                else:
                    code = line
                break

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            print("Use 'exit' to quit.")
            buf = ""
            continue

        code = code.strip()
        if not code:
            continue

        # Special commands -- check for exit/quit anywhere in the line
        if code in ("exit", "quit") or code.rstrip(";") in ("exit", "quit"):
            break

        if code == "help":
            _print_help()
            continue

        if code.startswith("help "):
            _print_func_help(session, code[5:].strip())
            continue

        # Evaluate
        suppress = code.rstrip().endswith(";")
        try:
            # Capture stdout from disp() etc.
            import io
            old_stdout = sys.stdout
            capture = io.StringIO()
            sys.stdout = capture

            result = session.eval(code)

            sys.stdout = old_stdout
            captured = capture.getvalue()

            # Print any captured output (from disp, fprintf, etc.)
            if captured:
                print(captured, end="")

            # Print result if not suppressed
            if result is not None and not suppress:
                print(result)

        except KeyboardInterrupt:
            sys.stdout = sys.__stdout__
            print()
            print("Interrupted.")
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"{_RED}error: {e}{_RESET}", file=sys.stderr)


def _print_help():
    """Print general help text."""
    print("Forge CLI -- Interactive Computation Engine")
    print()
    print("  Evaluate any Octave-compatible expression at the >> prompt.")
    print("  End a line with ;  to suppress output.")
    print("  End a line with ... to continue on the next line.")
    print()
    print("  Special commands:")
    print("    help            Show this help")
    print("    help <func>     Show help for a function")
    print("    who / whos      List workspace variables")
    print("    clear           Clear workspace")
    print("    exit / quit     Exit the REPL")
    print()


def _print_func_help(session, name):
    """Print docstring for a function."""
    func = session._engine.functions.get(name)
    if func is None:
        print(f"Function '{name}' not found.")
        return
    doc = getattr(func, "__doc__", None)
    if doc:
        print(doc)
    else:
        print(f"{name}: no documentation available.")


if __name__ == "__main__":
    main()
