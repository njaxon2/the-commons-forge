#!/usr/bin/env python3
"""
Forge Benchmark Runner
======================
Runs .m benchmark files in both the Forge engine and GNU Octave (and optionally
MATLAB), compares numerical results, and produces a summary table + JSON report.

Usage:
    python3 bench_runner.py [--forge-only | --octave-only] [--runs N] [benchmark_name ...]

If no benchmark names are given, all bench_*.m files in the benchmarks/ directory
are run.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
RESULTS_FILE = BENCH_DIR / "results.json"

# Tolerance for floating-point comparison (relative)
# All benchmarks use deterministic inputs (linspace, ones, eye) so results
# should match within floating-point tolerance.
REL_TOL = 1e-4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_output(text: str):
    """Extract TIME= and RESULT= values from benchmark output."""
    t_match = re.search(r"TIME=([\d.eE+-]+)", text)
    r_match = re.search(r"RESULT=([\d.eE+-]+)", text)
    elapsed = float(t_match.group(1)) if t_match else None
    result = float(r_match.group(1)) if r_match else None
    return elapsed, result


def results_match(a, b, rel_tol=REL_TOL):
    """Check if two numerical results are close enough."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom < rel_tol


def format_time(t):
    """Format seconds as a readable string."""
    if t is None:
        return "ERROR"
    if t < 0.001:
        return f"{t*1e6:.0f} us"
    if t < 1.0:
        return f"{t*1000:.1f} ms"
    return f"{t:.3f} s"


def speedup_str(forge_t, other_t):
    """Return a speedup string (positive = Forge faster)."""
    if forge_t is None or other_t is None:
        return "N/A"
    if forge_t == 0:
        return "inf"
    ratio = other_t / forge_t
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

def run_forge(mfile: Path, runs: int = 1):
    """Run a benchmark through the Forge engine directly."""
    # Import Forge
    sys.path.insert(0, str(PROJECT_ROOT))
    from forge.engine.session import ForgeSession
    from forge.engine.types import _unwrap
    import numpy as np

    code = mfile.read_text()

    times = []
    result = None

    for i in range(runs):
        session = ForgeSession()
        # Capture stdout from fprintf
        import io
        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture

        wall_start = time.perf_counter()
        try:
            output = session.eval(code)
        except Exception as e:
            sys.stdout = old_stdout
            return None, None, str(e)
        wall_elapsed = time.perf_counter() - wall_start

        sys.stdout = old_stdout
        captured = capture.getvalue()

        # The eval output and captured stdout both may contain our markers
        full_output = (output or "") + "\n" + captured
        t_internal, res = parse_output(full_output)

        # Use internal tic/toc time if available, else wall time
        elapsed = t_internal if t_internal is not None else wall_elapsed
        times.append(elapsed)
        if res is not None:
            result = res

    avg_time = sum(times) / len(times) if times else None
    return avg_time, result, None


def run_octave(mfile: Path, runs: int = 1):
    """Run a benchmark through octave-cli subprocess."""
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if not octave:
        return None, None, "octave-cli not found"

    code = mfile.read_text()
    times = []
    result = None

    for i in range(runs):
        try:
            proc = subprocess.run(
                [octave, "--no-gui", "--silent", "--eval", code],
                capture_output=True, text=True, timeout=120,
                cwd=str(mfile.parent)
            )
            full_output = proc.stdout + "\n" + proc.stderr
            t_internal, res = parse_output(full_output)
            if t_internal is not None:
                times.append(t_internal)
            if res is not None:
                result = res
            if proc.returncode != 0 and t_internal is None:
                return None, None, f"Octave error (rc={proc.returncode}): {proc.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return None, None, "Octave timed out (120s)"
        except Exception as e:
            return None, None, str(e)

    avg_time = sum(times) / len(times) if times else None
    return avg_time, result, None


def run_matlab(mfile: Path, runs: int = 1):
    """Run a benchmark through MATLAB (if available)."""
    matlab = shutil.which("matlab")
    if not matlab:
        return None, None, "MATLAB not found"

    code = mfile.read_text()
    times = []
    result = None

    for i in range(runs):
        try:
            proc = subprocess.run(
                [matlab, "-batch", code],
                capture_output=True, text=True, timeout=120,
                cwd=str(mfile.parent)
            )
            full_output = proc.stdout + "\n" + proc.stderr
            t_internal, res = parse_output(full_output)
            if t_internal is not None:
                times.append(t_internal)
            if res is not None:
                result = res
            if proc.returncode != 0 and t_internal is None:
                return None, None, f"MATLAB error: {proc.stderr[:200]}"
        except subprocess.TimeoutExpired:
            return None, None, "MATLAB timed out (120s)"
        except Exception as e:
            return None, None, str(e)

    avg_time = sum(times) / len(times) if times else None
    return avg_time, result, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_benchmarks(names=None):
    """Find benchmark .m files."""
    pattern = str(BENCH_DIR / "bench_*.m")
    all_files = sorted(glob.glob(pattern))
    if not names:
        return [Path(f) for f in all_files]
    selected = []
    for name in names:
        # Allow "matmul" or "bench_matmul" or "bench_matmul.m"
        if not name.startswith("bench_"):
            name = "bench_" + name
        if not name.endswith(".m"):
            name = name + ".m"
        full = BENCH_DIR / name
        if full.exists():
            selected.append(full)
        else:
            print(f"WARNING: Benchmark {name} not found, skipping")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Forge Benchmark Runner")
    parser.add_argument("benchmarks", nargs="*", help="Specific benchmarks to run")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per benchmark (averaged)")
    parser.add_argument("--forge-only", action="store_true", help="Run only Forge")
    parser.add_argument("--octave-only", action="store_true", help="Run only Octave")
    args = parser.parse_args()

    benchmarks = discover_benchmarks(args.benchmarks)
    if not benchmarks:
        print("No benchmarks found!")
        sys.exit(1)

    run_forge_flag = not args.octave_only
    run_octave_flag = not args.forge_only
    run_matlab_flag = not args.forge_only and not args.octave_only and shutil.which("matlab") is not None

    engines = []
    if run_forge_flag:
        engines.append("Forge")
    if run_octave_flag:
        engines.append("Octave")
    if run_matlab_flag:
        engines.append("MATLAB")

    print(f"\n{'='*80}")
    print(f"  Forge Benchmark Suite")
    print(f"  Engines: {', '.join(engines)} | Runs per benchmark: {args.runs}")
    print(f"  Benchmarks: {len(benchmarks)}")
    print(f"{'='*80}\n")

    results = []

    for mfile in benchmarks:
        name = mfile.stem.replace("bench_", "")
        # Read the first comment line for description
        first_line = mfile.read_text().split("\n")[0]
        desc = first_line.lstrip("% ").strip() if first_line.startswith("%") else name

        entry = {"name": name, "description": desc, "file": mfile.name}
        print(f"  [{name}] {desc}")

        if run_forge_flag:
            print(f"    Forge ...", end=" ", flush=True)
            ft, fr, fe = run_forge(mfile, args.runs)
            entry["forge_time"] = ft
            entry["forge_result"] = fr
            entry["forge_error"] = fe
            if fe:
                print(f"ERROR: {fe}")
            else:
                print(f"{format_time(ft)} (result={fr:.4f})" if fr is not None else f"{format_time(ft)}")

        if run_octave_flag:
            print(f"    Octave ...", end=" ", flush=True)
            ot, orr, oe = run_octave(mfile, args.runs)
            entry["octave_time"] = ot
            entry["octave_result"] = orr
            entry["octave_error"] = oe
            if oe:
                print(f"ERROR: {oe}")
            else:
                print(f"{format_time(ot)} (result={orr:.4f})" if orr is not None else f"{format_time(ot)}")

        if run_matlab_flag:
            print(f"    MATLAB ...", end=" ", flush=True)
            mt, mr, me = run_matlab(mfile, args.runs)
            entry["matlab_time"] = mt
            entry["matlab_result"] = mr
            entry["matlab_error"] = me
            if me:
                print(f"ERROR: {me}")
            else:
                print(f"{format_time(mt)} (result={mr:.4f})" if mr is not None else f"{format_time(mt)}")

        # Check result match
        if run_forge_flag and run_octave_flag:
            match = results_match(entry.get("forge_result"), entry.get("octave_result"))
            entry["match_octave"] = match

        results.append(entry)
        print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*100}")

    if run_forge_flag and run_octave_flag:
        header = f"  {'Benchmark':<22} {'Forge':>12} {'Octave':>12} {'Match':>7} {'Comparison':>20}"
        print(header)
        print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*7} {'-'*20}")
        for r in results:
            name = r["name"]
            ft = format_time(r.get("forge_time"))
            ot = format_time(r.get("octave_time"))
            match = "YES" if r.get("match_octave") else "NO"
            if r.get("forge_error"):
                ft = "ERROR"
                match = "N/A"
            if r.get("octave_error"):
                ot = "ERROR"
                match = "N/A"
            spd = speedup_str(r.get("forge_time"), r.get("octave_time"))
            print(f"  {name:<22} {ft:>12} {ot:>12} {match:>7} {spd:>20}")
    elif run_forge_flag:
        header = f"  {'Benchmark':<22} {'Forge':>12}"
        print(header)
        print(f"  {'-'*22} {'-'*12}")
        for r in results:
            ft = format_time(r.get("forge_time")) if not r.get("forge_error") else "ERROR"
            print(f"  {r['name']:<22} {ft:>12}")
    elif run_octave_flag:
        header = f"  {'Benchmark':<22} {'Octave':>12}"
        print(header)
        print(f"  {'-'*22} {'-'*12}")
        for r in results:
            ot = format_time(r.get("octave_time")) if not r.get("octave_error") else "ERROR"
            print(f"  {r['name']:<22} {ot:>12}")

    print(f"{'='*100}")

    # Overall stats
    if run_forge_flag and run_octave_flag:
        matched = sum(1 for r in results if r.get("match_octave"))
        total = len(results)
        forge_errors = sum(1 for r in results if r.get("forge_error"))
        octave_errors = sum(1 for r in results if r.get("octave_error"))
        both_ok = [r for r in results if r.get("forge_time") and r.get("octave_time")]
        if both_ok:
            forge_faster = sum(1 for r in both_ok if r["forge_time"] < r["octave_time"])
            octave_faster = len(both_ok) - forge_faster
            forge_total = sum(r["forge_time"] for r in both_ok)
            octave_total = sum(r["octave_time"] for r in both_ok)
            print(f"\n  Results matched: {matched}/{total}")
            print(f"  Forge errors: {forge_errors} | Octave errors: {octave_errors}")
            print(f"  Forge faster: {forge_faster}/{len(both_ok)} | Octave faster: {octave_faster}/{len(both_ok)}")
            print(f"  Total time - Forge: {forge_total:.3f}s | Octave: {octave_total:.3f}s")
            if forge_total > 0:
                print(f"  Overall ratio: Octave/Forge = {octave_total/forge_total:.2f}x")

    # Save JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs_per_benchmark": args.runs,
            "engines": engines,
            "benchmarks": results
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
