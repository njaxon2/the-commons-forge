# Forge Model User Characterization (Vetted)

**Status**: Approved by human operator. This is the canonical characterization. Do not summarize, simplify, or paraphrase — read in full.

---

The MATLAB user is typically not a software engineer. They're an engineer, scientist, or applied mathematician. They think in terms of physical systems, signals, data, and equations — not abstractions, architectures, or design patterns.

## How they work differently:

**The command window is their REPL and their notebook.** A MATLAB user doesn't start by creating a file. They start by typing directly into the command window. `x = linspace(0, 2*pi, 100)` then `plot(x, sin(x))` — and a figure appears. They're exploring data interactively before they ever write a script. The command window is where thinking happens. In Python, you open a file or a notebook first. In MATLAB, you just start typing.

**The workspace is always visible and always matters.** A MATLAB user glances at the workspace panel constantly. "What did I name that matrix? How big is it? Is it complex?" They double-click a variable and a spreadsheet-like editor opens so they can inspect individual elements. This isn't a debugger — it's just how they look at data. Python developers use `print()` or a debugger; MATLAB users look at the workspace panel the way you'd look at a dashboard.

**Everything is a matrix.** A scalar is a 1x1 matrix. A string is a 1xN character array (in classic MATLAB). There are no "types" in the Python sense — there are dimensions. When something goes wrong, it's almost always a dimension mismatch, and the error message tells you "expected 3x3, got 3x1." The mental model is linear algebra, not object hierarchies.

**Scripts evolve from command history.** The typical workflow is: experiment in the command window, get something working, then highlight lines in the command history panel and click "Create Script." The script is a formalization of what they already discovered interactively. This is the opposite of the Python workflow where you write a module, import it, and test it.

**Plots are first-class, immediate, and interactive.** `plot(x, y)` opens a figure window right now. You can zoom, pan, add labels, change colors — all with mouse clicks on the figure toolbar. Then `hold on; plot(x, z, 'r--')` adds a second trace. Visualization isn't a library you import — it's a verb built into the language. Engineers live in plots. They're checking whether a transfer function looks right, whether a signal has the expected spectrum, whether a mesh converges.

**Toolboxes define domains, not libraries.** A controls engineer doesn't "pip install control-systems." They have the Control System Toolbox, and it gives them `bode()`, `step()`, `tf()`, `ss()`. A signal processing person has `fft()`, `filter()`, `spectrogram()`. These aren't third-party packages — they feel like extensions of the language itself. The toolbox functions appear in help, in tab-completion, alongside the core functions.

## The problems they solve are different too:

* "I have accelerometer data from a test rig, I need to find the resonant frequencies" — load CSV, plot the raw signal, take the FFT, find peaks
* "I need to design a Butterworth filter with a 500Hz cutoff" — `[b, a] = butter(4, 500/Fs);` apply it; plot before and after
* "Does this control law stabilize the plant?" — build the transfer function, compute the root locus, check the step response
* "I need to solve this PDE on a mesh" — finite element setup, sparse matrices, iterative solvers, contour plots of the result
* "My professor gave me a system of 5 equations" — type them in, `A\b`, done

## On the driving task:

The driving task has to be a large task — one that's intrinsically larger than you're physically able to get done in a few seconds. The TIGA example serves well both because it exercises a lot of the core of the platform, but also because it can't be realistically accomplished in less than 15 minutes, just because of the slow physical processes that have to happen in its completed form. Eventually it will push the computing platform to its limit; that limit must be assessed and documented in terms of what's physically possible within the time-tolerance of the user.
