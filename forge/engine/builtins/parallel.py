# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Forge Parallel Computing Toolbox
==================================
Worker pools, parallel map, async evaluation, and distributed arrays
using Python's multiprocessing / concurrent.futures.

Backend: concurrent.futures, multiprocessing
"""

import os
import numpy as np
from numpy import ndarray
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Union, Optional, Dict, Any, List, Callable, Tuple


class ForgePool:
    """Wrapper around ProcessPoolExecutor with pool metadata."""

    def __init__(self, executor: ProcessPoolExecutor, n_workers: int):
        self.executor = executor
        self.n_workers = n_workers
        self.is_open = True

    def __repr__(self):
        status = "open" if self.is_open else "closed"
        return f"ForgePool(workers={self.n_workers}, {status})"


class ForgeFuture:
    """Wrapper around concurrent.futures.Future with output tracking."""

    def __init__(self, future: Future, n_outputs: int = 1):
        self.future = future
        self.n_outputs = n_outputs
        self._result = None
        self._done = False

    @property
    def done(self) -> bool:
        return self.future.done()

    def __repr__(self):
        state = "done" if self.done else "running"
        return f"ForgeFuture({state})"


class ForgeDistributed:
    """A distributed array split across pool workers.

    Stores chunks locally (conceptual distribution for the simplified model).
    """

    def __init__(self, chunks: List[ndarray], pool: ForgePool):
        self.chunks = chunks
        self.pool = pool

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def __repr__(self):
        total = sum(c.size for c in self.chunks)
        return f"ForgeDistributed(elements={total}, chunks={self.n_chunks})"


# Global reference to current pool
_current_pool: Optional[ForgePool] = None


# ---------------------------------------------------------------------------
# Pool Management
# ---------------------------------------------------------------------------

def forge_parpool(n: Optional[int] = None) -> ForgePool:
    """Create a worker pool.

    Parameters
    ----------
    n : number of workers (default: number of CPU cores)

    Returns
    -------
    ForgePool handle
    """
    global _current_pool
    if n is None:
        n = os.cpu_count() or 4
    n = int(n)
    executor = ProcessPoolExecutor(max_workers=n)
    pool = ForgePool(executor, n)
    _current_pool = pool
    return pool


def forge_delete_pool(pool: ForgePool) -> None:
    """Shut down and close a worker pool."""
    global _current_pool
    if pool.is_open:
        pool.executor.shutdown(wait=True)
        pool.is_open = False
    if _current_pool is pool:
        _current_pool = None


def forge_gcp() -> Optional[ForgePool]:
    """Get current pool (None if no pool is active)."""
    global _current_pool
    if _current_pool is not None and not _current_pool.is_open:
        _current_pool = None
    return _current_pool


# ---------------------------------------------------------------------------
# Parallel Execution
# ---------------------------------------------------------------------------

def forge_parfor_helper(func: Callable, items: list,
                        pool: Optional[ForgePool] = None) -> list:
    """Parallel map: apply func to each item in the list.

    Parameters
    ----------
    func  : callable that takes one argument
    items : list of arguments
    pool  : ForgePool (default: current pool via forge_gcp)

    Returns
    -------
    list of results in the same order as items
    """
    if pool is None:
        pool = forge_gcp()
    if pool is None or not pool.is_open:
        # Fallback to serial
        return [func(item) for item in items]

    futures = [pool.executor.submit(func, item) for item in items]
    return [f.result() for f in futures]


def forge_parfeval(pool: Optional[ForgePool], func: Callable,
                   nout: int = 1, *args) -> ForgeFuture:
    """Asynchronously evaluate a function on a pool worker.

    Parameters
    ----------
    pool  : ForgePool (None uses current pool)
    func  : callable
    nout  : number of expected outputs
    *args : arguments to func

    Returns
    -------
    ForgeFuture that can be polled or awaited
    """
    if pool is None:
        pool = forge_gcp()
    if pool is None or not pool.is_open:
        raise RuntimeError("No active pool. Create one with forge_parpool().")

    future = pool.executor.submit(func, *args)
    return ForgeFuture(future, nout)


def forge_fetchOutputs(ff: ForgeFuture):
    """Block until a ForgeFuture completes and return its result.

    Parameters
    ----------
    ff : ForgeFuture from forge_parfeval

    Returns
    -------
    The function's return value
    """
    result = ff.future.result()
    ff._result = result
    ff._done = True
    return result


# ---------------------------------------------------------------------------
# SPMD
# ---------------------------------------------------------------------------

def _spmd_worker(args):
    """Internal worker for SPMD execution."""
    func, worker_id, n_workers, extra_args = args
    return func(worker_id, n_workers, *extra_args)


def forge_spmd_helper(func: Callable, pool: Optional[ForgePool] = None,
                      *args) -> list:
    """Single-Program-Multiple-Data execution.

    The function signature must be: func(worker_id, n_workers, *args)
    where worker_id is 0-based.

    Parameters
    ----------
    func  : callable(worker_id, n_workers, *args)
    pool  : ForgePool (None uses current pool)
    *args : additional arguments passed to every worker

    Returns
    -------
    list of results, one per worker
    """
    if pool is None:
        pool = forge_gcp()
    if pool is None or not pool.is_open:
        raise RuntimeError("No active pool. Create one with forge_parpool().")

    n = pool.n_workers
    work_items = [(func, i, n, args) for i in range(n)]
    futures = [pool.executor.submit(_spmd_worker, item) for item in work_items]
    return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Distributed Arrays
# ---------------------------------------------------------------------------

def forge_distributed(data, pool: Optional[ForgePool] = None) -> ForgeDistributed:
    """Distribute an array across pool workers.

    Splits the first axis roughly evenly among workers.

    Parameters
    ----------
    data : array-like
    pool : ForgePool (None uses current pool)

    Returns
    -------
    ForgeDistributed object
    """
    if pool is None:
        pool = forge_gcp()
    if pool is None or not pool.is_open:
        raise RuntimeError("No active pool. Create one with forge_parpool().")

    arr = np.asarray(data)
    n = pool.n_workers
    chunks = np.array_split(arr, n, axis=0)
    return ForgeDistributed(chunks, pool)


def forge_gather(ddata: ForgeDistributed) -> ndarray:
    """Gather a distributed array back into a single array.

    Parameters
    ----------
    ddata : ForgeDistributed

    Returns
    -------
    numpy ndarray
    """
    return np.concatenate(ddata.chunks, axis=0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARALLEL_REGISTRY: Dict[str, Any] = {
    'forge_parpool': forge_parpool,
    'forge_delete_pool': forge_delete_pool,
    'forge_gcp': forge_gcp,
    'forge_parfor_helper': forge_parfor_helper,
    'forge_parfeval': forge_parfeval,
    'forge_fetchOutputs': forge_fetchOutputs,
    'forge_spmd_helper': forge_spmd_helper,
    'forge_distributed': forge_distributed,
    'forge_gather': forge_gather,
}
