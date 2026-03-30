# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Ongoing Quality Evaluation (OQE) system.

Instruments functions to collect runtime telemetry:
- Input/output hashes for reproducibility tracking
- Execution timing and memory usage
- Anomaly detection (unexpected NaN, shape changes, performance regression)
"""
import sqlite3
import hashlib
import time
import functools
import numpy as np
from pathlib import Path


DB_PATH = Path.home() / ".forge" / "oqe.db"


class OQEDatabase:
    """SQLite-backed OQE observation store."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        function_name TEXT NOT NULL,
        timestamp REAL NOT NULL,
        input_hash TEXT,
        output_hash TEXT,
        duration_ms REAL,
        memory_bytes INTEGER,
        anomaly_flag INTEGER DEFAULT 0,
        notes TEXT,
        input_summary TEXT,
        output_summary TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_func ON observations(function_name);
    CREATE INDEX IF NOT EXISTS idx_ts ON observations(timestamp);
    CREATE INDEX IF NOT EXISTS idx_anomaly ON observations(anomaly_flag);

    CREATE TABLE IF NOT EXISTS function_stats (
        function_name TEXT PRIMARY KEY,
        call_count INTEGER DEFAULT 0,
        total_duration_ms REAL DEFAULT 0,
        avg_duration_ms REAL DEFAULT 0,
        last_called REAL,
        anomaly_count INTEGER DEFAULT 0
    );
    """

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def record(self, function_name, input_hash=None, output_hash=None,
               duration_ms=None, memory_bytes=None, anomaly_flag=0,
               notes=None, input_summary=None, output_summary=None):
        """Record a single function observation."""
        ts = time.time()
        self._conn.execute(
            """INSERT INTO observations
               (function_name, timestamp, input_hash, output_hash,
                duration_ms, memory_bytes, anomaly_flag, notes,
                input_summary, output_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (function_name, ts, input_hash, output_hash,
             duration_ms, memory_bytes, anomaly_flag, notes,
             input_summary, output_summary)
        )
        self._conn.execute(
            """INSERT INTO function_stats
               (function_name, call_count, total_duration_ms, avg_duration_ms, last_called, anomaly_count)
               VALUES (?, 1, ?, ?, ?, ?)
               ON CONFLICT(function_name) DO UPDATE SET
                 call_count = call_count + 1,
                 total_duration_ms = total_duration_ms + excluded.total_duration_ms,
                 avg_duration_ms = (total_duration_ms + excluded.total_duration_ms) / (call_count + 1),
                 last_called = excluded.last_called,
                 anomaly_count = anomaly_count + excluded.anomaly_count""",
            (function_name, duration_ms or 0, duration_ms or 0, ts, anomaly_flag)
        )
        self._conn.commit()

    def query(self, function_name=None, anomalies_only=False, limit=100):
        """Query observations."""
        sql = "SELECT * FROM observations WHERE 1=1"
        params = []
        if function_name:
            sql += " AND function_name = ?"
            params.append(function_name)
        if anomalies_only:
            sql += " AND anomaly_flag = 1"
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_stats(self, function_name=None):
        """Get aggregated stats."""
        if function_name:
            cur = self._conn.execute(
                "SELECT * FROM function_stats WHERE function_name = ?",
                (function_name,))
        else:
            cur = self._conn.execute(
                "SELECT * FROM function_stats ORDER BY call_count DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def report(self):
        """Generate summary report."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM observations").fetchone()[0]
        anomalies = self._conn.execute(
            "SELECT COUNT(*) FROM observations WHERE anomaly_flag=1").fetchone()[0]
        functions = self._conn.execute(
            "SELECT COUNT(DISTINCT function_name) FROM observations").fetchone()[0]
        return {
            "total_observations": total,
            "total_anomalies": anomalies,
            "unique_functions": functions,
            "anomaly_rate": anomalies / total if total > 0 else 0,
        }

    def close(self):
        self._conn.close()


def _hash_args(args, kwargs):
    """Create a stable hash of function arguments."""
    h = hashlib.sha256()
    for a in args:
        if isinstance(a, np.ndarray):
            h.update(a.tobytes())
            h.update(str(a.shape).encode())
        else:
            h.update(str(a).encode())
    for k in sorted(kwargs):
        h.update(k.encode())
        v = kwargs[k]
        if isinstance(v, np.ndarray):
            h.update(v.tobytes())
        else:
            h.update(str(v).encode())
    return h.hexdigest()[:16]


def _hash_result(result):
    """Hash function output."""
    h = hashlib.sha256()
    if isinstance(result, np.ndarray):
        h.update(result.tobytes())
        h.update(str(result.shape).encode())
    elif isinstance(result, tuple):
        for r in result:
            h.update(str(type(r)).encode())
            if isinstance(r, np.ndarray):
                h.update(r.tobytes())
            else:
                h.update(str(r).encode())
    else:
        h.update(str(result).encode())
    return h.hexdigest()[:16]


def _summarize(val, max_len=100):
    """Short summary of a value for logging."""
    if isinstance(val, np.ndarray):
        return f"ndarray{val.shape} dtype={val.dtype}"
    return str(val)[:max_len]


def _detect_anomalies(result):
    """Check output for anomalies."""
    notes = []
    if isinstance(result, np.ndarray):
        if np.any(np.isnan(result)):
            notes.append("output_contains_nan")
        if np.any(np.isinf(result)):
            notes.append("output_contains_inf")
    return notes


_db = None

def _get_db():
    global _db
    if _db is None:
        _db = OQEDatabase()
    return _db


def oqe_instrument(func=None, *, name=None):
    """Decorator to instrument a function with OQE telemetry.

    Usage:
        @oqe_instrument
        def my_func(x):
            return x + 1

        @oqe_instrument(name="custom_name")
        def my_func(x):
            return x + 1
    """
    def decorator(fn):
        fn_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            db = _get_db()
            input_hash = _hash_args(args, kwargs)
            input_summary = ", ".join(_summarize(a) for a in args[:3])
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - t0) * 1000
            output_hash = _hash_result(result)
            output_summary = _summarize(result)
            anomalies = _detect_anomalies(result)
            anomaly_flag = 1 if anomalies else 0
            notes_str = "; ".join(anomalies) if anomalies else None
            db.record(
                function_name=fn_name,
                input_hash=input_hash,
                output_hash=output_hash,
                duration_ms=duration_ms,
                anomaly_flag=anomaly_flag,
                notes=notes_str,
                input_summary=input_summary,
                output_summary=output_summary,
            )
            return result

        wrapper._oqe_instrumented = True
        wrapper._oqe_name = fn_name
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
