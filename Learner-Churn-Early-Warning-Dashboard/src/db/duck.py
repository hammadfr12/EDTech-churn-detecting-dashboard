"""DuckDB connection helpers and the SQL template renderer.

The SQL files are plain, readable SQL with ``{{PLACEHOLDER}}`` tokens. This
module renders those tokens and executes the files against an in-memory (or
file-backed) DuckDB connection. Nothing here knows about pandas or sklearn.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb

from src import config

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class MissingPlaceholderError(ValueError):
    """Raised when a SQL template references a placeholder that was not given."""


def render_sql(template: str, values: Dict[str, Any]) -> str:
    """Substitute ``{{NAME}}`` tokens, failing loudly on anything unresolved."""
    missing = sorted(
        {m.group(1) for m in _PLACEHOLDER_RE.finditer(template)} - set(values)
    )
    if missing:
        raise MissingPlaceholderError(
            "SQL template references undefined placeholder(s): " + ", ".join(missing)
        )

    def _sub(match: re.Match[str]) -> str:
        return str(values[match.group(1)])

    return _PLACEHOLDER_RE.sub(_sub, template)


def read_sql_file(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_statements(sql: str) -> List[str]:
    """Split a SQL file into executable statements.

    DuckDB can execute multi-statement strings directly, but splitting lets us
    report *which* statement failed, which matters when a 5-file SQL layer
    breaks somewhere in the middle.
    """
    statements: List[str] = []
    buffer: List[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            buffer.append(line)
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip().rstrip(";").strip()
    if tail:
        statements.append(tail)
    return statements


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with a sensible local configuration."""
    con = duckdb.connect(database=database)
    con.execute("SET preserve_insertion_order = false;")
    return con


def execute_file(
    con: duckdb.DuckDBPyConnection,
    path: Path | str,
    values: Optional[Dict[str, Any]] = None,
) -> None:
    """Render and run a SQL file, one statement at a time."""
    sql = render_sql(read_sql_file(path), values or {})
    for i, statement in enumerate(split_statements(sql), start=1):
        try:
            con.execute(statement)
        except Exception as exc:  # pragma: no cover - surfaced to the caller
            head = statement.splitlines()[0][:120]
            raise RuntimeError(
                f"{Path(path).name}: statement {i} failed ({head}...): {exc}"
            ) from exc


def staging_values(data_dir: Path | str | None = None) -> Dict[str, Any]:
    """Placeholders needed by 01_staging.sql.

    ``data_dir`` must be threaded through from the caller. Defaulting it here
    silently would make a caller that passed a different directory read the
    real dataset instead, which is exactly the kind of bug that makes a test
    suite pass while testing nothing.
    """
    raw = Path(data_dir if data_dir is not None else config.RAW_DIR).resolve()
    return {"RAW": str(raw).replace("\\", "/")}


def load_staging(
    con: duckdb.DuckDBPyConnection, data_dir: Path | str | None = None
) -> None:
    """Create every staging view from the raw CSVs in ``data_dir``."""
    execute_file(con, config.SQL_DIR / "01_staging.sql", staging_values(data_dir))


def load_engagement(con: duckdb.DuckDBPyConnection) -> None:
    execute_file(
        con,
        config.SQL_DIR / "02_learner_week_engagement.sql",
        {
            "DAYS_PER_WEEK": config.DAYS_PER_WEEK,
            "MAX_FEATURE_WEEK": config.MAX_FEATURE_WEEK,
        },
    )


def load_cohort_metrics(con: duckdb.DuckDBPyConnection) -> None:
    execute_file(con, config.SQL_DIR / "05_cohort_metrics.sql", {})


def _ols_slope_sql(clicks_terms: Iterable[str]) -> str:
    """Emit a closed-form OLS slope of clicks against week index.

    slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

    The x values are 1..n and are constant, so Σ(x - x̄)² is precomputed
    numerically. y values are the per-week click columns. Zero-click weeks
    inside the window are real zeros and are included.
    """
    terms = list(clicks_terms)
    n = len(terms)
    x_mean = (n + 1) / 2
    denom = sum((i + 1 - x_mean) ** 2 for i in range(n))
    # numerator = Σ x*y - n*x̄*ȳ
    y_sum = " + ".join(f"COALESCE({t}, 0)" for t in terms)
    x_y_sum = " + ".join(f"{i + 1} * COALESCE({t}, 0)" for i, t in enumerate(terms))
    return (
        f"(({x_y_sum}) - {n} * {x_mean} * (CAST(({y_sum}) AS DOUBLE) / {n})) "
        f"/ {denom}"
    )


def feature_values(cutoff_week: int) -> Dict[str, Any]:
    """Build the placeholder set for 03_ and 04_ at a given cutoff week."""
    if cutoff_week < 1:
        raise ValueError(f"cutoff_week must be >= 1, got {cutoff_week}")

    weeks = list(range(1, cutoff_week + 1))
    click_cols = [f"clicks_w{w}" for w in weeks]
    active_cols = [f"active_days_w{w}" for w in weeks]
    cutoff_day = cutoff_week * config.DAYS_PER_WEEK - 1

    click_sum = " + ".join(f"COALESCE({c}, 0)" for c in click_cols)
    active_sum = " + ".join(f"COALESCE({c}, 0)" for c in active_cols)

    # A learner-week with no clickstream row is a genuine zero, not missing
    # data: absence of activity means zero clicks. These columns are the only
    # features where that is true, so they are coalesced here and nowhere else.
    week_cols_sql = ", ".join(
        f"COALESCE(w.{c}, 0) AS {c}" for c in click_cols + active_cols
    )

    values: Dict[str, Any] = {
        "CUTOFF_WEEK": cutoff_week,
        "CUTOFF_DAY": cutoff_day,
        "WEEK_COLS": week_cols_sql,
        "TOTAL_CLICKS": click_sum,
        "TOTAL_ACTIVE_DAYS": active_sum,
        "CLICKS_PER_DAY": (
            f"CASE WHEN ({active_sum}) > 0 "
            f"THEN CAST(({click_sum}) AS DOUBLE) / ({active_sum}) END"
        ),
    }

    if cutoff_week == 1:
        # With a single week there is no "previous week" to compare against.
        values["CLICK_DELTA"] = "CAST(NULL AS BIGINT)"
        values["ACTIVE_DELTA"] = "CAST(NULL AS BIGINT)"
    else:
        last, prev = click_cols[-1], click_cols[-2]
        last_a, prev_a = active_cols[-1], active_cols[-2]
        values["CLICK_DELTA"] = (
            f"COALESCE({last}, 0) - COALESCE({prev}, 0)"
        )
        values["ACTIVE_DELTA"] = (
            f"COALESCE({last_a}, 0) - COALESCE({prev_a}, 0)"
        )

    values["CLICK_SLOPE"] = _ols_slope_sql(click_cols)
    return values


def build_feature_snapshot(
    con: duckdb.DuckDBPyConnection, cutoff_week: int
) -> None:
    """Materialise assessment features and the feature snapshot for a cutoff."""
    values = feature_values(cutoff_week)
    execute_file(con, config.SQL_DIR / "03_assessment_features.sql", values)
    execute_file(con, config.SQL_DIR / "04_learner_week_features.sql", values)


def query_df(con: duckdb.DuckDBPyConnection, sql: str):
    """Run a query and return a pandas DataFrame."""
    return con.execute(sql).fetchdf()
