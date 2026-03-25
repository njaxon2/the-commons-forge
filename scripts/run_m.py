#!/usr/bin/env python3
"""Run an .m script via ForgeSession and print output. No GUI needed."""
import sys, os
sys.path.insert(0, "/home/ubuntu/forge")
os.environ["DISPLAY"] = ":99"  # for matplotlib
os.chdir("/home/ubuntu/forge/ForgeHome/tiga")

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval("addpath(\"/home/ubuntu/forge/ForgeHome/tiga\")")

if len(sys.argv) > 1:
    script = sys.argv[1]
    try:
        out = s.eval(script)
        if out:
            print(out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
else:
    # Interactive mode - read from stdin
    import readline
    while True:
        try:
            cmd = input("forge>> ")
            if cmd.strip() in ("quit", "exit"):
                break
            out = s.eval(cmd)
            if out:
                print(out)
        except EOFError:
            break
        except Exception as e:
            print(f"error: {e}")
