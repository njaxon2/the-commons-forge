"""R-POLISH-41: The .exe distribution SHALL use Intel MKL as its BLAS backend.

Model-user argument: The engineer running a convergence study with QR
decomposition in a loop, or computing SVD for a large dataset, experiences
the operation at roughly half the latency compared to OpenBLAS. MKL is the
industry standard for numerical linear algebra on Intel hardware; shipping
without it is a measurable performance regression that the user feels on
every matrix operation above a few hundred elements.

Decomposition:
    R-POLISH-41.1: numpy.show_config() reports "mkl" as the BLAS backend
    R-POLISH-41.2: SVD of a 500x500 matrix completes within 0.5s
    R-POLISH-41.3: mkl_rt.2.dll is loadable at runtime on Windows

Consistency: R-POLISH-41.1 confirms the compiled linkage. R-POLISH-41.2
confirms that the MKL acceleration is actually delivering expected
throughput (not just linked but unused). R-POLISH-41.3 confirms the
runtime DLL dependency is satisfied in the deployed environment.

Note: This test is only meaningful on Windows .exe builds. On the VPS
(Linux), numpy uses OpenBLAS and these tests are skipped.
"""
import sys
import pytest
import numpy as np


pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="MKL BLAS requirement applies to Windows .exe distribution only",
)


class TestMKLBlas:
    """R-POLISH-41: Verify Intel MKL is the active BLAS backend."""

    def test_blas_is_mkl(self):
        """R-POLISH-41.1: numpy BLAS backend SHALL be MKL."""
        # numpy 2.x exposes build config as a dict via __config__
        try:
            import numpy._core._multiarray_umath as _mu
            cfg = _mu.__cpu_features__  # exists but not what we need
        except Exception:
            pass

        # Check show_config output for "mkl"
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            np.show_config()
        config_text = buf.getvalue().lower()
        assert "mkl" in config_text, (
            "numpy BLAS backend is not MKL. "
            "Install the cgohlke MKL numpy wheel and ensure "
            "MSVCP140.dll + MKL DLLs are on PATH.\n"
            f"Config output: {config_text[:500]}"
        )

    def test_svd_performance(self):
        """R-POLISH-41.2: SVD 500x500 SHALL complete within 0.5s."""
        import time
        rng = np.random.default_rng(42)
        A = rng.standard_normal((500, 500))

        # Warm up
        np.linalg.svd(A[:50, :50], full_matrices=False)

        t0 = time.perf_counter()
        np.linalg.svd(A, full_matrices=False)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5, (
            f"SVD 500x500 took {elapsed:.3f}s (threshold: 0.5s). "
            "This suggests MKL is not active or hardware is very slow."
        )

    def test_mkl_dll_loadable(self):
        """R-POLISH-41.3: mkl_rt.2.dll SHALL be loadable at runtime."""
        import ctypes
        try:
            ctypes.CDLL("mkl_rt.2.dll")
        except OSError:
            pytest.fail(
                "mkl_rt.2.dll could not be loaded. "
                "Ensure the MKL pip package is installed and "
                "Library/bin is on the system PATH."
            )
