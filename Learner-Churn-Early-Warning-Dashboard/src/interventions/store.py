"""Local intervention tracking.

A single SQLite table demonstrating the *process design* for routing an
at-risk learner to a human reviewer. It stores no contact details and no
personal data beyond the pseudonymous OULAD identifier, which is already
pseudonymous in the source dataset.

This is deliberately not an outreach system. Nothing here sends anything.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS interventions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    code_module         TEXT    NOT NULL,
    code_presentation   TEXT    NOT NULL,
    id_student          INTEGER NOT NULL,
    cutoff_week         INTEGER NOT NULL,
    target_key          TEXT    NOT NULL,
    risk_probability    REAL,
    risk_band           TEXT,
    owner               TEXT,
    action              TEXT,
    due_date            TEXT,
    status              TEXT    NOT NULL DEFAULT 'New',
    notes               TEXT,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    UNIQUE (code_module, code_presentation, id_student, cutoff_week, target_key)
);
"""

_UPDATABLE = ("owner", "action", "due_date", "status", "notes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or config.INTERVENTION_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def upsert(
    code_module: str,
    code_presentation: str,
    id_student: int,
    cutoff_week: int,
    target_key: str,
    risk_probability: Optional[float] = None,
    risk_band: Optional[str] = None,
    owner: Optional[str] = None,
    action: Optional[str] = None,
    due_date: Optional[str] = None,
    status: str = "New",
    notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Create or update the intervention record for one learner."""
    if status not in config.INTERVENTION_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Allowed: {list(config.INTERVENTION_STATUSES)}"
        )
    if action is not None and action not in config.INTERVENTION_ACTIONS:
        raise ValueError(
            f"Invalid action '{action}'. Allowed: {list(config.INTERVENTION_ACTIONS)}"
        )

    now = _now()
    with _connect(db_path) as con:
        con.execute(
            """
            INSERT INTO interventions (
                code_module, code_presentation, id_student, cutoff_week, target_key,
                risk_probability, risk_band, owner, action, due_date, status, notes,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (code_module, code_presentation, id_student, cutoff_week, target_key)
            DO UPDATE SET
                risk_probability = COALESCE(excluded.risk_probability, risk_probability),
                risk_band        = COALESCE(excluded.risk_band, risk_band),
                owner            = COALESCE(excluded.owner, owner),
                action           = COALESCE(excluded.action, action),
                due_date         = COALESCE(excluded.due_date, due_date),
                status           = excluded.status,
                notes            = COALESCE(excluded.notes, notes),
                updated_at       = excluded.updated_at
            """,
            (
                code_module,
                code_presentation,
                int(id_student),
                int(cutoff_week),
                target_key,
                risk_probability,
                risk_band,
                owner,
                action,
                due_date,
                status,
                notes,
                now,
                now,
            ),
        )
        row = con.execute(
            """
            SELECT id FROM interventions
            WHERE code_module=? AND code_presentation=? AND id_student=?
              AND cutoff_week=? AND target_key=?
            """,
            (code_module, code_presentation, int(id_student), int(cutoff_week), target_key),
        ).fetchone()
    return int(row["id"]) if row else -1


def update(record_id: int, db_path: Optional[Path] = None, **fields: Any) -> None:
    """Update named fields on an existing record, stamping ``updated_at``."""
    unknown = set(fields) - set(_UPDATABLE)
    if unknown:
        raise ValueError(f"Cannot update unknown field(s): {sorted(unknown)}")
    if "status" in fields and fields["status"] not in config.INTERVENTION_STATUSES:
        raise ValueError(f"Invalid status '{fields['status']}'.")
    if "action" in fields and fields["action"] not in config.INTERVENTION_ACTIONS:
        raise ValueError(f"Invalid action '{fields['action']}'.")
    if not fields:
        return

    assignments = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [_now(), int(record_id)]
    with _connect(db_path) as con:
        con.execute(
            f"UPDATE interventions SET {assignments}, updated_at = ? WHERE id = ?", values
        )


def fetch(
    db_path: Optional[Path] = None,
    code_module: Optional[str] = None,
    code_presentation: Optional[str] = None,
    status: Optional[str] = None,
) -> pd.DataFrame:
    """Read the intervention table, optionally filtered."""
    clauses: List[str] = []
    params: List[Any] = []
    if code_module:
        clauses.append("code_module = ?")
        params.append(code_module)
    if code_presentation:
        clauses.append("code_presentation = ?")
        params.append(code_presentation)
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as con:
        rows = con.execute(
            f"SELECT * FROM interventions {where} ORDER BY updated_at DESC", params
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def delete(record_id: int, db_path: Optional[Path] = None) -> None:
    with _connect(db_path) as con:
        con.execute("DELETE FROM interventions WHERE id = ?", (int(record_id),))


def status_summary(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Counts per status, used for the coverage metric."""
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT status, COUNT(*) AS n FROM interventions GROUP BY status"
        ).fetchall()
    return {r["status"]: int(r["n"]) for r in rows}
