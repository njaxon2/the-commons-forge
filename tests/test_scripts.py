# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for complex multi-function scripts and realistic workflows.

Requirement R-SCRIPT: The engine SHALL correctly execute multi-function
scripts, multi-output function calls, and parfor loops, propagating
variables into the caller's workspace and supporting all standard
decomposition output patterns.

Model-user argument: An engineer migrating from MATLAB writes .m files
containing multiple helper functions followed by a script body. The
engineer expects function definitions to be available in the script
scope, multi-output calls like [U,S,V]=svd(A) to populate all output
variables, and parfor to execute loop iterations (sequentially in Forge,
matching Octave's serial fallback). If any of these patterns fail, the
engineer cannot port existing MATLAB scripts to Forge.

Decomposition:
  R-SCRIPT-01..04: Multi-function script execution (stats, composition,
    varargin, eigenvalue verification)
  R-SCRIPT-05..11: Multi-output function calls (svd, lu, qr, unique,
    sort, size, meshgrid)
  R-SCRIPT-12..13: Parfor loop execution (scalar accumulation, array fill)

Consistency argument: Multi-function scripts (R-SCRIPT-01..04) validate
function definition, cross-function calls, variable arguments, and
mathematical verification patterns. Multi-output tests (R-SCRIPT-05..11)
validate the [out1, out2, ...] = func(args) calling convention for all
standard decomposition functions. Parfor tests (R-SCRIPT-12..13) validate
parallel-loop syntax execution. Together these cover the three script
execution patterns the engineer uses: define-and-call, multi-output
capture, and parallel iteration.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestMultiFunctionScripts:
    """R-SCRIPT-01..04: Multi-function scripts SHALL define functions
    visible to the script body, support cross-function calls, accept
    variable arguments, and produce mathematically verifiable results.

    Model-user argument: The engineer defines helper functions at the top
    of a script file and calls them from the script body below. This is
    the standard MATLAB script organization pattern. Functions must also
    call each other, accept varargin, and produce results that can be
    verified against known mathematical properties.

    Decomposition:
      R-SCRIPT-01: stats function returns mean and std of a vector
      R-SCRIPT-02: Function calls another function defined in the same eval
      R-SCRIPT-03: varargin accumulator sums variable number of arguments
      R-SCRIPT-04: Eigenvalues of a positive-definite matrix are all positive

    Consistency: R-SCRIPT-01 tests basic function definition and multi-output.
    R-SCRIPT-02 tests cross-function calls. R-SCRIPT-03 tests variable
    arguments. R-SCRIPT-04 tests a complete mathematical verification workflow.
    """

    def test_stats_function(self, s):
        """R-SCRIPT-01: stats function returns mean of [1..10] as 5.5."""
        code = "function [mu, sigma] = stats(x)\n  mu = mean(x);\n  sigma = std(x);\nend\n[m, sig] = stats([1 2 3 4 5 6 7 8 9 10]);"
        s.eval(code)
        assert abs(float(s.eval("m")) - 5.5) < 0.01

    def test_function_calling_function(self, s):
        """R-SCRIPT-02: sum_of_squares calls square, returns 30 for [1 2 3 4]."""
        code = "function r = square(x)\n  r = x .^ 2;\nend\nfunction r = sum_of_squares(v)\n  r = sum(square(v));\nend\nresult = sum_of_squares([1 2 3 4]);"
        s.eval(code)
        assert float(s.eval("result")) == 30.0

    def test_varargin_accumulator(self, s):
        """R-SCRIPT-03: varargin accumulator sums 1..5 to 15."""
        code = "function r = flex_add(varargin)\n  r = 0;\n  for k = 1:nargin\n    r = r + varargin{k};\n  end\nend\ntotal = flex_add(1, 2, 3, 4, 5);"
        s.eval(code)
        assert float(s.eval("total")) == 15.0

    def test_positive_definite_eigenvalues(self, s):
        """R-SCRIPT-04: Eigenvalues of a constructed positive-definite matrix are all positive."""
        code = "n = 5;\nA = rand(n);\nA = A + A';\nA = A + n * eye(n);\n[V, D] = eig(A);\neigenvalues = diag(D);\nall_pos = all(eigenvalues > 0);"
        s.eval(code)
        assert float(s.eval("all_pos")) == 1.0


class TestMultiOutput:
    """R-SCRIPT-05..11: Multi-output function calls SHALL populate all
    output variables with correctly shaped results.

    Model-user argument: The engineer uses [U,S,V]=svd(A), [L,U,P]=lu(A),
    [Q,R]=qr(A), [vals,idx]=unique(x), [sorted,idx]=sort(x),
    [m,n]=size(A), and [X,Y]=meshgrid(a,b) extensively. Each must
    assign all output variables with the correct shapes, or multi-step
    computations that depend on these outputs will fail.

    Decomposition:
      R-SCRIPT-05: svd returns U with shape (2,2)
      R-SCRIPT-06: lu returns L with shape (2,2)
      R-SCRIPT-07: qr returns Q with shape (2,2)
      R-SCRIPT-08: unique returns sorted unique values [1,2,3]
      R-SCRIPT-09: sort returns sorted values [1,1,3,4,5]
      R-SCRIPT-10: size returns row count 2 and column count 3
      R-SCRIPT-11: meshgrid returns X with shape (2,3)

    Consistency: These seven tests cover all standard multi-output
    decomposition and utility functions the engineer encounters.
    """

    def test_svd_3out(self, s):
        """R-SCRIPT-05: svd returns U with shape (2,2)."""
        s.eval("[U, S, V] = svd([1 2; 3 4])")
        U = s._engine.workspace.get("U")
        assert U.data.shape == (2, 2)

    def test_lu_3out(self, s):
        """R-SCRIPT-06: lu returns L with shape (2,2)."""
        s.eval("[L, U, P] = lu([1 2; 3 4])")
        L = s._engine.workspace.get("L")
        assert L.data.shape == (2, 2)

    def test_qr_2out(self, s):
        """R-SCRIPT-07: qr returns Q with shape (2,2)."""
        s.eval("[Q, R] = qr([1 2; 3 4])")
        Q = s._engine.workspace.get("Q")
        assert Q.data.shape == (2, 2)

    def test_unique_2out(self, s):
        """R-SCRIPT-08: unique returns sorted unique values [1,2,3]."""
        s.eval("[vals, idx] = unique([3 1 2 1 3])")
        vals = s._engine.workspace.get("vals")
        np.testing.assert_array_equal(vals.data.flatten(), [1, 2, 3])

    def test_sort_2out(self, s):
        """R-SCRIPT-09: sort returns sorted values [1,1,3,4,5]."""
        s.eval("[sorted_v, idx] = sort([3 1 4 1 5])")
        sv = s._engine.workspace.get("sorted_v")
        np.testing.assert_array_equal(sv.data.flatten(), [1, 1, 3, 4, 5])

    def test_size_2out(self, s):
        """R-SCRIPT-10: size returns row count 2 and column count 3."""
        s.eval("[m, n] = size([1 2 3; 4 5 6])")
        assert float(s.eval("m")) == 2.0
        assert float(s.eval("n")) == 3.0

    def test_meshgrid_2out(self, s):
        """R-SCRIPT-11: meshgrid returns X with shape (2,3)."""
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = s._engine.workspace.get("X")
        assert X.data.shape == (2, 3)


class TestParfor:
    """R-SCRIPT-12..13: parfor loops SHALL execute loop iterations and
    produce correct results (serial execution, matching Octave fallback).

    Model-user argument: The engineer uses parfor in MATLAB for
    embarrassingly parallel loops. In Octave (and Forge), parfor falls
    back to serial execution but must still produce correct results.
    The engineer's scripts should run without modification.

    Decomposition:
      R-SCRIPT-12: parfor scalar accumulation sums 1..5 to 15
      R-SCRIPT-13: parfor array fill assigns i^2 to each element

    Consistency: Scalar reduction (R-SCRIPT-12) and indexed assignment
    (R-SCRIPT-13) cover the two parfor usage patterns.
    """

    def test_parfor_basic(self, s):
        """R-SCRIPT-12: parfor accumulates sum of 1..5 to 15."""
        s.eval("r = 0; parfor i = 1:5; r = r + i; end")
        assert float(s.eval("r")) == 15.0

    def test_parfor_array(self, s):
        """R-SCRIPT-13: parfor fills array with i^2 for i=1..10."""
        s.eval("x = zeros(1, 10); parfor i = 1:10; x(i) = i^2; end")
        x = s._engine.workspace.get("x")
        np.testing.assert_array_equal(x.data.flatten(), [i**2 for i in range(1, 11)])
