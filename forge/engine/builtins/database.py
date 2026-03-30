# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Forge Database Toolbox
=======================
SQLite-backed database connectivity with a MATLAB-style interface.

Backend: sqlite3
"""

import sqlite3
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np


class ForgeDBConnection:
    """Wrapper around sqlite3 connection with metadata."""

    def __init__(self, conn: sqlite3.Connection, dbtype: str,
                 host: str, user: str, dbname: str):
        self.conn = conn
        self.dbtype = dbtype
        self.host = host
        self.user = user
        self.dbname = dbname
        self.is_open = True

    def __repr__(self):
        status = "open" if self.is_open else "closed"
        return f"ForgeDBConnection({self.dbtype}://{self.host}/{self.dbname}, {status})"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def forge_database(dbtype: str = 'sqlite', host: str = '',
                   user: str = '', pwd: str = '',
                   dbname: str = ':memory:') -> ForgeDBConnection:
    """Connect to a database.

    Parameters
    ----------
    dbtype : database type ('sqlite' supported; others reserved for future)
    host   : hostname (ignored for sqlite)
    user   : username (ignored for sqlite)
    pwd    : password (ignored for sqlite)
    dbname : database name / file path (default ':memory:' for in-memory)

    Returns
    -------
    ForgeDBConnection handle
    """
    if dbtype.lower() != 'sqlite':
        raise NotImplementedError(
            f"Database type '{dbtype}' not yet supported. Use 'sqlite'."
        )
    conn = sqlite3.connect(dbname)
    conn.row_factory = sqlite3.Row
    return ForgeDBConnection(conn, dbtype, host, user, dbname)


def forge_close_db(handle: ForgeDBConnection) -> None:
    """Close a database connection."""
    if handle.is_open:
        handle.conn.close()
        handle.is_open = False


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def forge_exec(handle: ForgeDBConnection, sql: str,
               params: Optional[tuple] = None) -> int:
    """Execute a SQL statement (INSERT, UPDATE, DELETE, CREATE, etc.).

    Parameters
    ----------
    handle : ForgeDBConnection
    sql    : SQL statement
    params : optional tuple of bind parameters

    Returns
    -------
    Number of rows affected
    """
    if not handle.is_open:
        raise RuntimeError("Connection is closed.")
    cur = handle.conn.cursor()
    if params:
        cur.execute(sql, params)
    else:
        cur.execute(sql)
    handle.conn.commit()
    return cur.rowcount


def forge_fetch(handle: ForgeDBConnection, sql: str,
                params: Optional[tuple] = None) -> Dict[str, Any]:
    """Fetch query results as a struct (dict of column arrays).

    Parameters
    ----------
    handle : ForgeDBConnection
    sql    : SELECT statement
    params : optional bind parameters

    Returns
    -------
    dict with column names as keys and lists as values,
    plus '_nrows' for row count.
    """
    if not handle.is_open:
        raise RuntimeError("Connection is closed.")
    cur = handle.conn.cursor()
    if params:
        cur.execute(sql, params)
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    if not rows:
        return {'_nrows': 0}

    col_names = [desc[0] for desc in cur.description]
    result: Dict[str, Any] = {'_nrows': len(rows)}
    for col in col_names:
        result[col] = []
    for row in rows:
        for i, col in enumerate(col_names):
            result[col].append(row[i])

    # Convert numeric columns to numpy arrays
    for col in col_names:
        vals = result[col]
        if vals and all(isinstance(v, (int, float)) for v in vals if v is not None):
            result[col] = np.array(vals, dtype=float)

    return result


# ---------------------------------------------------------------------------
# High-level insert / update
# ---------------------------------------------------------------------------

def forge_insert(handle: ForgeDBConnection, table: str,
                 data: Dict[str, list]) -> int:
    """Insert rows into a table.

    Parameters
    ----------
    handle : ForgeDBConnection
    table  : table name
    data   : dict of {column_name: list_of_values}
             All lists must have the same length.

    Returns
    -------
    Number of rows inserted
    """
    if not handle.is_open:
        raise RuntimeError("Connection is closed.")
    columns = list(data.keys())
    n_rows = len(data[columns[0]])
    placeholders = ', '.join(['?'] * len(columns))
    col_str = ', '.join(columns)
    sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"

    cur = handle.conn.cursor()
    rows_data = []
    for i in range(n_rows):
        row = tuple(data[col][i] for col in columns)
        rows_data.append(row)
    cur.executemany(sql, rows_data)
    handle.conn.commit()
    return cur.rowcount


def forge_update(handle: ForgeDBConnection, table: str,
                 data: Dict[str, Any], where: str,
                 params: Optional[tuple] = None) -> int:
    """Update rows in a table.

    Parameters
    ----------
    handle : ForgeDBConnection
    table  : table name
    data   : dict of {column_name: new_value}
    where  : WHERE clause (without the WHERE keyword)
    params : bind parameters for the WHERE clause

    Returns
    -------
    Number of rows updated
    """
    if not handle.is_open:
        raise RuntimeError("Connection is closed.")
    set_parts = []
    set_vals: List[Any] = []
    for col, val in data.items():
        set_parts.append(f"{col} = ?")
        set_vals.append(val)

    sql = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {where}"
    all_params = tuple(set_vals) + (params if params else ())

    cur = handle.conn.cursor()
    cur.execute(sql, all_params)
    handle.conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# High-level read / write
# ---------------------------------------------------------------------------

def forge_sqlread(handle: ForgeDBConnection, table: str,
                  columns: Optional[List[str]] = None,
                  where: Optional[str] = None,
                  params: Optional[tuple] = None) -> Dict[str, Any]:
    """Read an entire table (or subset) into a struct.

    Parameters
    ----------
    handle  : ForgeDBConnection
    table   : table name
    columns : list of column names (default: all)
    where   : optional WHERE clause
    params  : bind parameters for WHERE
    """
    col_str = ', '.join(columns) if columns else '*'
    sql = f"SELECT {col_str} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return forge_fetch(handle, sql, params)


def forge_sqlwrite(handle: ForgeDBConnection, table: str,
                   data: Dict[str, list],
                   create_table: bool = False) -> int:
    """Write data to a table (bulk insert).

    Parameters
    ----------
    handle       : ForgeDBConnection
    table        : table name
    data         : dict of {column_name: list_of_values}
    create_table : if True, create the table if it doesn't exist
                   (columns default to TEXT type)

    Returns
    -------
    Number of rows written
    """
    if create_table:
        columns = list(data.keys())
        col_defs = ', '.join(f"{c} TEXT" for c in columns)
        forge_exec(handle, f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")
    return forge_insert(handle, table, data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATABASE_REGISTRY: Dict[str, Any] = {
    'forge_database': forge_database,
    'forge_close_db': forge_close_db,
    'forge_exec': forge_exec,
    'forge_fetch': forge_fetch,
    'forge_insert': forge_insert,
    'forge_update': forge_update,
    'forge_sqlread': forge_sqlread,
    'forge_sqlwrite': forge_sqlwrite,
}
