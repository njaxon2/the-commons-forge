#!/usr/bin/env python3
"""Generate function reference documentation for Forge IDE.

Produces a Markdown file listing all registered functions by toolbox,
with signatures and docstrings.
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.engine.builtins import BUILTIN_REGISTRY

# Toolbox modules and their registries
TOOLBOXES = [
    ("elfun", "Elementary Functions"),
    ("general", "General Math"),
    ("specfun", "Special Functions"),
    ("linalg", "Linear Algebra"),
    ("polynomial", "Polynomials & Interpolation"),
    ("sets", "Set Operations"),
    ("special_matrix", "Special Matrices"),
    ("strings", "String Functions"),
    ("time_funcs", "Time & Date"),
    ("ode", "ODE Solvers"),
    ("optimization", "Optimization"),
    ("geometry", "Computational Geometry"),
    ("fileio", "File I/O"),
    ("sparse", "Sparse Matrices"),
    ("plotting", "Plotting"),
    ("signal", "Signal Processing"),
    ("image", "Image Processing"),
    ("statistics", "Statistics"),
    ("audio", "Audio"),
    ("web", "Web & FTP"),
    ("control", "Control Systems"),
    ("financial", "Financial"),
    ("comms", "Communications"),
    ("database", "Database"),
    ("parallel", "Parallel Computing"),
    ("fuzzy", "Fuzzy Logic"),
    ("neural", "Neural Networks"),
    ("instrument", "Instrument Control"),
    ("symbolic", "Symbolic Math"),
]


def get_registry(name):
    try:
        mod = __import__(f"forge.engine.builtins.{name}", fromlist=[""])
        for attr in dir(mod):
            val = getattr(mod, attr)
            if isinstance(val, dict) and attr.endswith("_REGISTRY"):
                return val
    except ImportError:
        pass
    return {}


def get_doc(func):
    if callable(func):
        doc = inspect.getdoc(func)
        if doc:
            return doc.split("\n")[0]
    return ""


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FUNCTION_REFERENCE.md")

    lines = [
        "# Forge IDE - Function Reference",
        "",
        f"**Total built-in functions: {len(BUILTIN_REGISTRY)}**",
        "",
    ]

    total = 0
    for mod_name, display_name in TOOLBOXES:
        reg = get_registry(mod_name)
        if not reg:
            continue

        lines.append(f"## {display_name} ({len(reg)} functions)")
        lines.append("")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")

        for fname in sorted(reg.keys()):
            entry = reg[fname]
            func = entry["func"] if isinstance(entry, dict) else entry
            doc = get_doc(func)
            lines.append(f"| `{fname}` | {doc} |")
            total += 1

        lines.append("")

    lines.append(f"---\n*Generated automatically. {total} functions documented.*")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {out_path} with {total} functions across {len(TOOLBOXES)} toolboxes")


if __name__ == "__main__":
    main()
