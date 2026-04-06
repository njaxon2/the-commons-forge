# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for general toolbox (31 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001

V&V traceability backfill: R-GEN-01 through R-GEN-09.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.builtins.general import *


class TestCoordinateTransforms:
    """R-GEN-01: The general toolbox SHALL provide coordinate transform
    functions (cart2pol, pol2cart, cart2sph, sph2cart) that are numerically
    round-trip consistent within machine epsilon.

    Model-user argument: The engineer converts between coordinate systems
    when analyzing antenna patterns (spherical), radar tracks (polar), or
    robot kinematics (Cartesian). Round-trip conversion must preserve the
    original values; any drift means the transform pair is broken and
    spatial calculations will be wrong.

    Decomposition: cart2pol/pol2cart round-trip, cart2pol known value,
    cart2sph/sph2cart round-trip, and pol2cart vectorized tested.
    Consistency: These four tests verify both round-trip fidelity and
    known-value correctness for all coordinate-system pairs.
    """

    def test_cart2pol_roundtrip(self):
        """R-GEN-01.01: cart2pol then pol2cart recovers original (x, y)."""
        x, y = ForgeArray(3.0), ForgeArray(4.0)
        theta, r = forge_cart2pol(x, y)
        x2, y2 = forge_pol2cart(theta, r)
        assert abs(_unwrap(x2) - 3.0) < 1e-14
        assert abs(_unwrap(y2) - 4.0) < 1e-14

    def test_cart2pol_known(self):
        """R-GEN-01.02: cart2pol(1,1) returns r=sqrt(2), theta=pi/4."""
        theta, r = forge_cart2pol(ForgeArray(1.0), ForgeArray(1.0))
        assert abs(_unwrap(r) - np.sqrt(2)) < 1e-14
        assert abs(_unwrap(theta) - np.pi/4) < 1e-14

    def test_cart2sph_roundtrip(self):
        """R-GEN-01.03: cart2sph then sph2cart recovers original (x, y, z)."""
        x, y, z = ForgeArray(1.0), ForgeArray(2.0), ForgeArray(3.0)
        az, el, r = forge_cart2sph(x, y, z)
        x2, y2, z2 = forge_sph2cart(az, el, r)
        assert abs(_unwrap(x2) - 1.0) < 1e-14
        assert abs(_unwrap(y2) - 2.0) < 1e-14
        assert abs(_unwrap(z2) - 3.0) < 1e-14

    def test_pol2cart_vectorized(self):
        """R-GEN-01.04: pol2cart on angle vector produces correct (x, y) arrays."""
        theta = ForgeArray(np.array([0, np.pi/2, np.pi]))
        r = ForgeArray(np.array([1.0, 1.0, 1.0]))
        x, y = forge_pol2cart(theta, r)
        np.testing.assert_allclose(_unwrap(x).ravel(), [1, 0, -1], atol=1e-14)
        np.testing.assert_allclose(_unwrap(y).ravel(), [0, 1, 0], atol=1e-14)


class TestArrayManipulation:
    """R-GEN-02: The general toolbox SHALL provide array manipulation functions
    (circshift, flip, sortrows, repelem, postpad, prepad, rescale) with
    correct element ordering and value transformations.

    Model-user argument: The engineer shifts data for circular convolution
    (circshift), reverses signals for correlation (flip), sorts tabular data
    by columns (sortrows), replicates elements for upsampling (repelem),
    pads/truncates signals to fixed lengths (postpad/prepad), and normalizes
    data to specific ranges (rescale). Each operation must preserve or
    transform values exactly as documented.

    Decomposition: circshift right/left, flip vector/dim, sortrows,
    repelem, postpad, prepad, postpad truncate, rescale default, and
    rescale custom range tested. Consistency: These 11 sub-tests cover all
    array manipulation functions and their parameter variations.
    """

    def test_circshift_right(self):
        """R-GEN-02.01: circshift by +2 rotates elements right."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_circshift(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r.ravel(), [4, 5, 1, 2, 3])

    def test_circshift_left(self):
        """R-GEN-02.02: circshift by -1 rotates elements left."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_circshift(x, ForgeArray(-1.0)))
        np.testing.assert_array_equal(r.ravel(), [2, 3, 4, 5, 1])

    def test_flip_vector(self):
        """R-GEN-02.03: flip reverses element order."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_flip(x))
        np.testing.assert_array_equal(r.ravel(), [3, 2, 1])

    def test_flip_dim(self):
        """R-GEN-02.04: flip along dim 2 reverses columns."""
        x = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        r = _unwrap(forge_flip(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r, np.array([[2, 1], [4, 3]]))

    def test_sortrows(self):
        """R-GEN-02.05: sortrows orders rows by first column ascending."""
        x = ForgeArray(np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]))
        r = _unwrap(forge_sortrows(x))
        assert r[0, 0] == 1.0
        assert r[2, 0] == 3.0

    def test_repelem_scalar(self):
        """R-GEN-02.06: repelem repeats each element the specified number of times."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_repelem(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 1, 2, 2, 3, 3])

    def test_postpad(self):
        """R-GEN-02.07: postpad extends with zeros to the requested length."""
        x = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_postpad(x, ForgeArray(5.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 2, 0, 0, 0])

    def test_prepad(self):
        """R-GEN-02.08: prepad prepends zeros to the requested length."""
        x = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_prepad(x, ForgeArray(5.0)))
        np.testing.assert_array_equal(r.ravel(), [0, 0, 0, 1, 2])

    def test_postpad_truncate(self):
        """R-GEN-02.09: postpad truncates when target length is shorter."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_postpad(x, ForgeArray(3.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 2, 3])

    def test_rescale_default(self):
        """R-GEN-02.10: rescale maps min to 0 and max to 1 by default."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_rescale(x))
        assert abs(r.flat[0]) < 1e-15  # min -> 0
        assert abs(r.flat[-1] - 1.0) < 1e-15  # max -> 1

    def test_rescale_custom_range(self):
        """R-GEN-02.11: rescale maps to custom [-1, 1] range."""
        x = ForgeArray(np.array([0.0, 5.0, 10.0]))
        r = _unwrap(forge_rescale(x, ForgeArray(-1.0), ForgeArray(1.0))).ravel()
        np.testing.assert_allclose(r, [-1, 0, 1], atol=1e-14)


class TestNumericalCalculus:
    """R-GEN-03: The general toolbox SHALL provide numerical calculus functions
    (trapz, cumtrapz, gradient) that return correct integrals and
    derivatives for known analytical cases.

    Model-user argument: The engineer integrates measured signals (e.g.,
    acceleration to velocity with trapz) and computes numerical derivatives
    (e.g., velocity from position with gradient). These must agree with
    analytical results for simple functions, otherwise numerical analysis
    results are unreliable.

    Decomposition: trapz on a linear function, cumtrapz output length,
    gradient of a linear function, and gradient with explicit spacing tested.
    Consistency: These cover definite integration, cumulative integration
    shape, and gradient computation with default and explicit spacing.
    """

    def test_trapz_uniform(self):
        """R-GEN-03.01: Integral of f(x)=x from 0 to 1 equals 0.5."""
        x = ForgeArray(np.linspace(0, 1, 1001))
        y = x  # f(x) = x
        r = _unwrap(forge_trapz(y, x))
        assert abs(r - 0.5) < 1e-6

    def test_cumtrapz_length(self):
        """R-GEN-03.02: cumtrapz output has n-1 elements."""
        y = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        r = _unwrap(forge_cumtrapz(y))
        assert r.size == 3

    def test_gradient_linear(self):
        """R-GEN-03.03: Gradient of linear [1,2,3,4,5] is constant 1.0."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_gradient(x))
        np.testing.assert_allclose(r.ravel(), 1.0, atol=1e-14)

    def test_gradient_with_spacing(self):
        """R-GEN-03.04: Gradient of x^2 at x=2 is approximately 4.0."""
        x = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0, 16.0]))
        r = _unwrap(forge_gradient(x, ForgeArray(1.0)))
        # Central differences of x^2 at x=0,1,2,3,4
        assert abs(r.flat[2] - 4.0) < 1e-10  # 2*2


class TestInterpolation:
    """R-GEN-04: The general toolbox SHALL provide linear interpolation
    (interp1) that returns correct values at query points between known
    data samples.

    Model-user argument: The engineer interpolates measured data to align
    time series, upsample signals, or evaluate transfer functions at
    arbitrary frequencies. Linear interpolation at midpoints between known
    samples must return the arithmetic mean of the neighbors.

    Decomposition: Linear interpolation at three query points tested.
    Consistency: This verifies the default linear method at interior points.
    """

    def test_interp1_linear(self):
        """R-GEN-04.01: Linear interpolation at midpoints returns correct values."""
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        y = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0]))
        xq = ForgeArray(np.array([0.5, 1.5, 2.5]))
        r = _unwrap(forge_interp1(x, y, xq))
        assert abs(r.flat[0] - 0.5) < 1e-14  # Linear interp
        assert abs(r.flat[1] - 2.5) < 1e-14


class TestMathUtilities:
    """R-GEN-05: The general toolbox SHALL provide math utility functions
    (deg2rad, rad2deg, nextpow2, bincoeff, idivide, xor) with correct
    results for known inputs.

    Model-user argument: The engineer converts angles between degrees and
    radians for sensor data, computes FFT lengths with nextpow2, calculates
    combinatorial coefficients with bincoeff, performs integer division for
    index calculations, and applies XOR for bit-level logic. Each must
    return the documented result.

    Decomposition: deg2rad, rad2deg, nextpow2 (non-power, exact power),
    bincoeff, idivide (fix, floor modes), and xor tested. Consistency:
    These eight sub-tests cover all math utility functions in the toolbox.
    """

    def test_deg2rad(self):
        """R-GEN-05.01: 180 degrees converts to pi radians."""
        r = _unwrap(forge_deg2rad(ForgeArray(180.0)))
        assert abs(r - np.pi) < 1e-14

    def test_rad2deg(self):
        """R-GEN-05.02: pi radians converts to 180 degrees."""
        r = _unwrap(forge_rad2deg(ForgeArray(np.pi)))
        assert abs(r - 180.0) < 1e-12

    def test_nextpow2(self):
        """R-GEN-05.03: nextpow2(100) returns 7 (2^7 = 128)."""
        r = _unwrap(forge_nextpow2(ForgeArray(100.0)))
        assert int(r.flat[0]) == 7  # 2^7 = 128

    def test_nextpow2_exact(self):
        """R-GEN-05.04: nextpow2(64) returns 6 (2^6 = 64)."""
        r = _unwrap(forge_nextpow2(ForgeArray(64.0)))
        assert int(r.flat[0]) == 6

    def test_bincoeff(self):
        """R-GEN-05.05: bincoeff(10, 3) = 120."""
        r = _unwrap(forge_bincoeff(ForgeArray(10.0), ForgeArray(3.0)))
        assert abs(r - 120.0) < 1e-10

    def test_idivide_fix(self):
        """R-GEN-05.06: idivide(7, 2) with default truncation returns 3."""
        r = _unwrap(forge_idivide(ForgeArray(7.0), ForgeArray(2.0)))
        assert int(r.flat[0]) == 3

    def test_idivide_floor(self):
        """R-GEN-05.07: idivide(-7, 2, 'floor') returns -4."""
        r = _unwrap(forge_idivide(ForgeArray(-7.0), ForgeArray(2.0), ForgeChar("floor")))
        assert int(r.flat[0]) == -4

    def test_xor(self):
        """R-GEN-05.08: XOR returns true only where inputs differ."""
        a = ForgeArray(np.array([True, True, False, False]))
        b = ForgeArray(np.array([True, False, True, False]))
        r = _unwrap(forge_xor(a, b)).ravel()
        np.testing.assert_array_equal(r, [False, True, True, False])


class TestComparison:
    """R-GEN-06: The general toolbox SHALL provide deep equality comparison
    functions (isequal, isequaln) with correct NaN handling.

    Model-user argument: The engineer compares arrays for equality in test
    harnesses and validation scripts. isequal must return false when NaN is
    present (since NaN != NaN), while isequaln must treat NaN as equal to
    NaN for data-comparison purposes.

    Decomposition: isequal same, isequal different, isequaln with matching
    NaN, and isequaln with non-matching NaN tested. Consistency: These four
    tests cover the two comparison functions with their NaN-sensitive
    distinctions.
    """

    def test_isequal_same(self):
        """R-GEN-06.01: isequal returns true for identical arrays."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isequal(a, a)) == True

    def test_isequal_different(self):
        """R-GEN-06.02: isequal returns false for arrays with different values."""
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([1.0, 3.0]))
        assert _unwrap(forge_isequal(a, b)) == False

    def test_isequaln_nan(self):
        """R-GEN-06.03: isequaln treats NaN as equal to NaN."""
        a = ForgeArray(np.array([1.0, np.nan, 3.0]))
        b = ForgeArray(np.array([1.0, np.nan, 3.0]))
        assert _unwrap(forge_isequaln(a, b)) == True

    def test_isequaln_nan_different(self):
        """R-GEN-06.04: isequaln returns false when NaN vs non-NaN."""
        a = ForgeArray(np.array([1.0, np.nan]))
        b = ForgeArray(np.array([1.0, 2.0]))
        assert _unwrap(forge_isequaln(a, b)) == False


class TestMisc:
    """R-GEN-07: The general toolbox SHALL provide geometry (polyarea),
    accumulation (accumarray), logarithmic spacing (logspace), multi-output
    distribution (deal), and type formatting (int2str) functions.

    Model-user argument: The engineer computes polygon areas for land
    surveys or mesh quality metrics (polyarea), accumulates grouped data
    for histograms and sparse matrix construction (accumarray), creates
    logarithmically spaced frequency vectors for Bode plots (logspace),
    distributes values to multiple outputs (deal), and converts numbers to
    display strings (int2str).

    Decomposition: polyarea (square, triangle), accumarray, logspace, deal,
    and int2str tested. Consistency: These cover the remaining general
    toolbox functions not tested in earlier classes.
    """

    def test_polyarea_square(self):
        """R-GEN-07.01: Unit square area equals 1."""
        x = ForgeArray(np.array([0.0, 1.0, 1.0, 0.0]))
        y = ForgeArray(np.array([0.0, 0.0, 1.0, 1.0]))
        r = _unwrap(forge_polyarea(x, y))
        assert abs(r - 1.0) < 1e-14

    def test_polyarea_triangle(self):
        """R-GEN-07.02: Right triangle with legs 1 has area 0.5."""
        x = ForgeArray(np.array([0.0, 1.0, 0.0]))
        y = ForgeArray(np.array([0.0, 0.0, 1.0]))
        r = _unwrap(forge_polyarea(x, y))
        assert abs(r - 0.5) < 1e-14

    def test_accumarray(self):
        """R-GEN-07.03: accumarray sums values by subscript group."""
        subs = ForgeArray(np.array([1, 1, 2, 3, 3]))
        vals = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_accumarray(subs, vals)).ravel()
        np.testing.assert_array_equal(r, [3, 3, 9])

    def test_logspace(self):
        """R-GEN-07.04: logspace(0, 2, 3) produces [1, 10, 100]."""
        r = _unwrap(forge_logspace(ForgeArray(0.0), ForgeArray(2.0), ForgeArray(3.0))).ravel()
        np.testing.assert_allclose(r, [1, 10, 100], rtol=1e-14)

    def test_deal_single(self):
        """R-GEN-07.05: deal(5) returns 5."""
        r = forge_deal(ForgeArray(5.0))
        assert _unwrap(r) == 5.0

    def test_int2str(self):
        """R-GEN-07.06: int2str(42) returns '42' as a char string."""
        r = forge_int2str(ForgeArray(42.0))
        assert r.to_str() == "42"
