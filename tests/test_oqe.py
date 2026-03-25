"""Tests for OQE database and instrumentation system."""
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
    def test_record_and_query(self, db):
        """Insert observation, query it back."""
        db.record("test_func", input_hash="abc123", output_hash="def456",
                  duration_ms=1.5, anomaly_flag=0)
        results = db.query("test_func")
        assert len(results) == 1
        assert results[0]["function_name"] == "test_func"
        assert results[0]["input_hash"] == "abc123"
        assert results[0]["duration_ms"] == 1.5

    def test_query_by_function(self, db):
        """Query filters by function name."""
        db.record("func_a", duration_ms=1.0)
        db.record("func_b", duration_ms=2.0)
        db.record("func_a", duration_ms=3.0)
        results_a = db.query("func_a")
        results_b = db.query("func_b")
        assert len(results_a) == 2
        assert len(results_b) == 1

    def test_anomaly_detection_query(self, db):
        """Query for anomalies only."""
        db.record("func_x", anomaly_flag=0)
        db.record("func_x", anomaly_flag=1, notes="output_contains_nan")
        db.record("func_x", anomaly_flag=0)
        anomalies = db.query("func_x", anomalies_only=True)
        assert len(anomalies) == 1
        assert anomalies[0]["notes"] == "output_contains_nan"

    def test_function_stats(self, db):
        """Stats aggregate correctly."""
        db.record("my_func", duration_ms=10.0)
        db.record("my_func", duration_ms=20.0)
        db.record("my_func", duration_ms=30.0)
        stats = db.get_stats("my_func")
        assert len(stats) == 1
        assert stats[0]["call_count"] == 3
        assert stats[0]["total_duration_ms"] == 60.0

    def test_report(self, db):
        """Report gives correct summary."""
        db.record("f1", anomaly_flag=0)
        db.record("f1", anomaly_flag=1)
        db.record("f2", anomaly_flag=0)
        report = db.report()
        assert report["total_observations"] == 3
        assert report["total_anomalies"] == 1
        assert report["unique_functions"] == 2
        assert abs(report["anomaly_rate"] - 1/3) < 1e-10

    def test_db_persistence(self, tmp_path):
        """Data persists across connections."""
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
    def test_decorator_basic(self, tmp_path, monkeypatch):
        """@oqe_instrument captures call data."""
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
        """Instrumentation handles numpy arrays."""
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
        """NaN in output triggers anomaly flag."""
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
    def test_hash_args_deterministic(self):
        """Same args produce same hash."""
        h1 = _hash_args((1, "hello"), {"key": 42})
        h2 = _hash_args((1, "hello"), {"key": 42})
        assert h1 == h2

    def test_hash_args_numpy(self):
        """Numpy arrays hash by content."""
        a = np.array([1.0, 2.0])
        h1 = _hash_args((a,), {})
        h2 = _hash_args((a.copy(),), {})
        assert h1 == h2

    def test_hash_result_numpy(self):
        """Result hashing works for arrays."""
        r = np.array([1.0, 2.0, 3.0])
        h = _hash_result(r)
        assert len(h) == 16

    def test_detect_anomalies_clean(self):
        """No anomalies for clean output."""
        assert _detect_anomalies(np.array([1.0, 2.0])) == []

    def test_detect_anomalies_nan(self):
        """NaN detected."""
        notes = _detect_anomalies(np.array([1.0, float("nan")]))
        assert "output_contains_nan" in notes

    def test_detect_anomalies_inf(self):
        """Inf detected."""
        notes = _detect_anomalies(np.array([float("inf")]))
        assert "output_contains_inf" in notes
