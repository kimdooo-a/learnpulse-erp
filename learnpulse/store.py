"""Storage and the read-only ERP connector.

The prototype keeps the canonical ERP tables and LearnPulse's own tables
in one SQLite file. The connector reads the canonical tables through a
read-only URI, which is how a real adapter would treat a school ERP.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .generate import ERP_TABLES

DEFAULT_DB = Path("data") / "learnpulse.db"

DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    class TEXT NOT NULL,
    week INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('accept', 'edit', 'dismiss')),
    playbook_key TEXT,
    step TEXT,
    note TEXT,
    dismiss_reason TEXT,
    role TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT, action TEXT, detail TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def write_tables(db: Path, tables: dict[str, pd.DataFrame]) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False, chunksize=50_000)
        con.executescript(DECISIONS_DDL)


def read_erp(db: Path) -> dict[str, pd.DataFrame]:
    """Read-only connector: the canonical ERP tables."""
    uri = f"file:{Path(db).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        return {t: pd.read_sql_query(f"SELECT * FROM {t}", con) for t in ERP_TABLES}


def read_table(db: Path, name: str) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(f"SELECT * FROM {name}", con)


def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(DECISIONS_DDL)
    return con
