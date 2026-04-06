# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for OQE database and instrumentation system.

Requirement R-OQE-01: The Octave Query Equivalence (OQE) system SHALL record,
query, and analyze function call observations to detect numerical anomalies and
verify that Forge's output matches established reference implementations.

Model-user argument: An engineer migrating from Octave expects that every
function produces identical results in Forge. The OQE system is the
infrastructure that makes this verifiable: it records input/output hashes,
execution times, and anomaly flags for every instrumented function call. When
a function produces NaN or Inf where it should not, the anomaly is flagged
automatically. This gives the engineer (and the Forge development team)
continuous confidence that numerical equivalence is maintained.

Decomposition:
    R-OQE-01  The OQEDatabase SHALL correctly record, query, aggregate, and
              report function call observations with anomaly detection.
    R-OQE-02  The @oqe_instrument decorator SHALL transparently capture call
              data without altering function behavior.
    R-OQE-03  The helper functions (_hash_args, _hash_result, _detect_anomalies)
              SHALL produce deterministic hashes and detect NaN/Inf anomalies.

Consistency argument: R-OQE-01 tests the storage layer (CRUD, filtering,
aggregation, persistence). R-OQE-02 tests the instrumentation layer (decorator
transparency, numpy support, anomaly flagging). R-OQE-03 tests the utility
functions that underpin both layers (hashing determinism, anomaly detection).
Together these three layers cover the complete OQE pipeline from data capture
through storage to analysis.
"""
import os
import tempfile
import numpy as np
import pytest
from forge.validation.oqe import OQEDatabase, oqe_instrument, _hash_args, _hash_result, _detect_anomalies


@pytest.fixture
def db(tmp_path):
    """Fresh OQE database in temp directory."""
    db_path = tmp_path / "test_oqe.db"
    d = OQEDatabase(db_path=db_path)
    yield d
    d.close()


class TestOQEDatabase:
    """R-OQE-01: The OQEDatabase SHALL correctly store, retrieve, filter,
    aggregate, and persist function call observations.

    Model-user argument: The OQE database is the foundation of Forge's
    numerical equivalence guarantee. If it cannot reliably store and query
    observations, the entire verification pipeline is compromised. The
    engineer relies on the report and anomaly queries to confirm that their
    migrated code produces correct results.

    Decomposition:
        R-OQE-01.1  Record and query back a single observation.
        R-OQE-01.2  Query filters by function name.
        R-OQE-01.3  Query with anomalies_only flag filters correctly.
        R-OQE-01.4  get_stats aggregates call count and total duration.
        R-OQE-01.5  report returns correct summary statistics.
        R-OQE-01.6  Data persists across database connections.

    Consistency: These six cases cover the full CRUD lifecycle (create via
    record, read via query/stats/report), filtering (by function, by anomaly),
    aggregation (stats), and durability (persistence across connections).
    """

    def test_record_and_query(self, db):
        """R-OQE-01.1: Single observation round-trips through record and query."""
        db.record("test_func", input_hash="abc123", output_hash="def456",
                  duration_ms=1.5, anomaly_flag=0)
        results = db.query("test_func")
        assert len(results) == 1
        assert results[0]["function_name"] == "test_func"
        assert results[0]["input_hash"] == "abc123"
        assert results[0]["duration_ms"] == 1.5

    def test_query_by_function(self, db):
        """R-OQE-01.2: Query filters observations by function name."""
        db.record("func_a", duration_ms=1.0)
        db.record("func_b", duration_ms=2.0)
        db.record("func_a", duration_ms=3.0)
        results_a = db.query("func_a")
        results_b = db.query("func_b")
        assert len(results_a) == 2
        assert len(results_b) == 1

    def test_anomaly_detection_query(self, db):
        """R-OQE-01.3: anomalies_only flag returns only flagged observations."""
        db.record("func_x", anomaly_flag=0)
        db.record("func_x", anomaly_flag=1, notes="output_contains_nan")
        db.record("func_x", anomaly_flag=0)
        anomalies = db.query("func_x", anomalies_only=True)
        assert len(anomalies) == 1
        assert anomalies[0]["notes"] == "output_contains_nan"

    def test_function_stats(self, db):
        """R-OQE-01.4: get_stats aggregates call count and total duration."""
        db.record("my_func", duration_ms=10.0)
        db.record("my_func", duration_ms=20.0)
        db.record("my_func", duration_ms=30.0)
        stats = db.get_stats("my_func")
        assert len(stats) == 1
        assert stats[0]["call_count"] == 3
        assert stats[0]["total_duration_ms"] == 60.0

    def test_report(self, db):
        """R-OQE-01.5: Report returns correct summary statistics."""
        db.record("f1", anomaly_flag=0)
        db.record("f1", anomaly_flag=1)
        db.record("f2", anomaly_flag=0)
        report = db.report()
        assert report["total_observations"] == 3
        assert report["total_anomalies"] == 1
        assert report["unique_functions"] == 2
        assert abs(report["anomaly_rate"] - 1/3) < 1e-10

    def test_db_persistence(self, tmp_path):
        """R-OQE-01.6: Data persists across database connections."""
        db_path = tmp_path / "persist_test.db"
        db1 = OQEDatabase(db_path=db_path)
        db1.record("persist_func", duration_ms=42.0)
        db1.close()
        db2 = OQEDatabase(db_path=db_path)
        results = db2.query("persist_func")
        assert len(results) == 1
        assert results[0]["duration_ms"] == 42.0
        db2.close()


class TestOQEInstrumentation:
    """R-OQE-02: The @oqe_instrument decorator SHALL transparently capture
    function call data (inputs, outputs, timing, anomalies) without altering
    the decorated function's return value or raising exceptions.

    Model-user argument: The engineer does not interact with OQE directly; it
    runs behind the scenes on instrumented engine functions. The decorator must
    not change function behavior (return values must be identical) and must
    handle numpy arrays correctly (hashing by content, detecting NaN/Inf).

    Decomposition:
        R-OQE-02.1  Decorated function returns correct value and records observation.
        R-OQE-02.2  Decorator handles numpy array inputs and outputs.
        R-OQE-02.3  NaN in output triggers anomaly flag automatically.

    Consistency: These three cases cover basic decorator transparency (scalar),
    numpy-specific handling (array hashing and summaries), and the automatic
    anomaly detection path (NaN flagging). Together they ensure instrumentation
    is both transparent and effective.
    """

    def test_decorator_basic(self, tmp_path, monkeypatch):
        """R-OQE-02.1: Decorated function returns correct value and records call."""
        import forge.validation.oqe as oqe_mod
        db = OQEDatabase(db_path=tmp_path / "instr.db")
        monkeypatch.setattr(oqe_mod, "_db", db)

        @oqe_instrument
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5
        assert add._oqe_instrumented is True
        obs = db.query(add._oqe_name)
        assert len(obs) == 1
        assert obs[0]["duration_ms"] > 0
        db.close()

    def test_decorator_with_numpy(self, tmp_path, monkeypatch):
        """R-OQE-02.2: Decorator handles numpy array I/O with content summaries."""
        import forge.validation.oqe as oqe_mod
        db = OQEDatabase(db_path=tmp_path / "np_instr.db")
        monkeypatch.setattr(oqe_mod, "_db", db)

        @oqe_instrument(name="np_double")
        def double(x):
            return x * 2

        arr = np.array([1.0, 2.0, 3.0])
        result = double(arr)
        np.testing.assert_array_equal(result, arr * 2)
        obs = db.query("np_double")
        assert len(obs) == 1
        assert "ndarray" in obs[0]["input_summary"]
        assert "ndarray" in obs[0]["output_summary"]
        db.close()

    def test_anomaly_flagging(self, tmp_path, monkeypatch):
        """R-OQE-02.3: NaN in output triggers automatic anomaly flag."""
        import forge.validation.oqe as oqe_mod
        db = OQEDatabase(db_path=tmp_path / "anom.db")
        monkeypatch.setattr(oqe_mod, "_db", db)

        @oqe_instrument
        def bad_func(x):
            return np.array([1.0, float("nan"), 3.0])

        bad_func(1)
        obs = db.query(anomalies_only=True)
        assert len(obs) == 1
        assert obs[0]["anomaly_flag"] == 1
        assert "output_contains_nan" in obs[0]["notes"]
        db.close()


class TestHelpers:
    """R-OQE-03: The OQE helper functions SHALL produce deterministic hashes
    for identical inputs (including numpy arrays) and detect NaN/Inf anomalies
    in numeric outputs.

    Model-user argument: The hashing functions enable the OQE system to detect
    when a function produces different output for the same input (regression
    detection). The anomaly detector catches the most common numerical failure
    modes (NaN, Inf) that indicate a function implementation bug.

    Decomposition:
        R-OQE-03.1  _hash_args produces identical hash for identical arguments.
        R-OQE-03.2  _hash_args produces identical hash for identical numpy arrays.
        R-OQE-03.3  _hash_result produces a 16-character hash for numpy arrays.
        R-OQE-03.4  _detect_anomalies returns empty list for clean output.
        R-OQE-03.5  _detect_anomalies detects NaN in output.
        R-OQE-03.6  _detect_anomalies detects Inf in output.

    Consistency: R-OQE-03.1 and R-OQE-03.2 cover determinism for scalar and
    array inputs. R-OQE-03.3 covers result hashing format. R-OQE-03.4 through
    R-OQE-03.6 cover the three anomaly states (clean, NaN, Inf). Together
    these ensure the utility layer is reliable for both hashing and detection.
    """

    def test_hash_args_deterministic(self):
        """R-OQE-03.1: Same scalar args produce same hash."""
        h1 = _hash_args((1, "hello"), {"key": 42})
        h2 = _hash_args((1, "hello"), {"key": 42})
        assert h1 == h2

    def test_hash_args_numpy(self):
        """R-OQE-03.2: Identical numpy arrays produce same hash."""
        a = np.array([1.0, 2.0])
        h1 = _hash_args((a,), {})
        h2 = _hash_args((a.copy(),), {})
        assert h1 == h2

    def test_hash_result_numpy(self):
        """R-OQE-03.3: Result hash is 16 characters for numpy arrays."""
        r = np.array([1.0, 2.0, 3.0])
        h = _hash_result(r)
        assert len(h) == 16

    def test_detect_anomalies_clean(self):
        """R-OQE-03.4: Clean output returns empty anomaly list."""
        assert _detect_anomalies(np.array([1.0, 2.0])) == []

    def test_detect_anomalies_nan(self):
        """R-OQE-03.5: NaN in output detected as anomaly."""
        notes = _detect_anomalies(np.array([1.0, float("nan")]))
        assert "output_contains_nan" in notes

    def test_detect_anomalies_inf(self):
        """R-OQE-03.6: Inf in output detected as anomaly."""
        notes = _detect_anomalies(np.array([float("inf")]))
        assert "output_contains_inf" in notes
