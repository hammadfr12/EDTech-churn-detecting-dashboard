"""Feature build: run the SQL layer and materialise dashboard artifacts.

This is the single entry point that turns the raw CSVs into everything the
model and the dashboard consume. It builds the snapshot for the default
cutoff week *and* the comparison cutoff week, so the week-3 vs week-4
comparison is a real comparison rather than a re-run.

Usage
-----
    python -m src.features.build --data-dir data/raw --cutoff-week 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src import config
from src.db import duck
from src.data import schema


def build_connection(data_dir: Path):
    """Open DuckDB and create staging + engagement views over the raw CSVs."""
    if not (Path(data_dir) / "studentInfo.csv").exists():
        raise schema.DatasetError(
            f"No OULAD files found in {data_dir}. Run `python -m src.data.download` "
            "or see README for manual steps."
        )
    con = duck.connect()
    duck.load_staging(con, data_dir)
    duck.load_engagement(con)
    duck.load_cohort_metrics(con)
    return con


def snapshot_for(con, cutoff_week: int) -> pd.DataFrame:
    """Build the cutoff-safe feature snapshot for one cutoff week."""
    duck.build_feature_snapshot(con, cutoff_week)
    df = duck.query_df(con, "SELECT * FROM learner_feature_snapshot")
    return df


def grain_report(con) -> Dict[str, Any]:
    """Row counts and key uniqueness for every materialised relation."""
    checks = {
        "stg_student_info": ("code_module, code_presentation, id_student", "stg_student_info"),
        "learner_week_matrix": (
            "code_module, code_presentation, id_student",
            "learner_week_matrix",
        ),
        "cohort_outcomes": (
            "code_module, code_presentation, id_student",
            "cohort_outcomes",
        ),
        "weekly_cohort_activity": (
            "code_module, code_presentation, week",
            "weekly_cohort_activity",
        ),
    }
    report: Dict[str, Any] = {}
    for name, (key, relation) in checks.items():
        row = con.execute(
            f"""
            SELECT COUNT(*) AS rows,
                   COUNT(*) - COUNT(DISTINCT ({key})) AS duplicate_key_rows
            FROM {relation}
            """
        ).fetchone()
        report[name] = {
            "rows": int(row[0]),
            "duplicate_key_rows": int(row[1]),
            "key": key,
        }
    return report


def cohort_metrics(con) -> Dict[str, Any]:
    """Cohort-level facts, computed from SQL rather than re-derived in pandas."""
    total = con.execute("SELECT COUNT(*) FROM stg_student_info").fetchone()[0]
    outcomes = con.execute(
        """
        SELECT final_result, COUNT(*) AS n FROM stg_student_info GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    presentations = con.execute(
        """
        SELECT code_module, code_presentation, COUNT(*) AS learners
        FROM stg_student_info GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()

    # Global weekly activity across every cohort in the dataset.
    weekly = con.execute(
        """
        SELECT w.week,
               SUM(w.active_learners) AS active_learners,
               SUM(w.learners_enrolled) AS learners_enrolled
        FROM weekly_cohort_activity AS w
        WHERE w.week <= 25
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()

    weekly["weekly_active_rate"] = weekly["active_learners"] / weekly["learners_enrolled"]
    weekly["prev_active"] = weekly["active_learners"].shift(1)
    weekly["net_change"] = weekly["prev_active"] - weekly["active_learners"]
    weekly["dropoff_rate"] = weekly["net_change"] / weekly["prev_active"]
    weekly["dropoff_rate"] = weekly["dropoff_rate"].where(weekly["prev_active"] > 0)

    # Steepest declines, ranked from real rows only. Weeks 1-2 are excluded
    # because enrolment settles then; a "drop" there is registration timing,
    # not disengagement.
    ranked = (
        weekly[weekly["week"] >= 2]
        .dropna(subset=["dropoff_rate"])
        .sort_values("dropoff_rate", ascending=False)
        .head(5)
    )

    return {
        "learner_course_records": int(total),
        "outcome_counts": {label: int(n) for label, n in outcomes},
        "presentations": [
            {"code_module": m, "code_presentation": p, "learners": int(n)}
            for m, p, n in presentations
        ],
        "weekly_activity": weekly.to_dict(orient="records"),
        "steepest_dropoff_weeks": ranked.to_dict(orient="records"),
    }


def build(data_dir: Path, cutoff_weeks: List[int]) -> Dict[str, Path]:
    """Run the full SQL build and write artifacts. Returns written paths."""
    config.ensure_dirs()
    con = build_connection(data_dir)

    written: Dict[str, Path] = {}

    # --- engagement grain (needed by the learner-detail page) ---------------
    learner_week = duck.query_df(
        con,
        """
        SELECT code_module, code_presentation, id_student, week,
               clicks, active_days, first_active_day, last_active_day
        FROM learner_week_engagement
        """,
    )
    written["learner_week"] = config.artifact_path("learner_week")
    learner_week.to_parquet(written["learner_week"], index=False)

    # --- assessment events (for the learner timeline) -----------------------
    events = duck.query_df(
        con,
        """
        SELECT code_module, code_presentation, id_student, id_assessment,
               assessment_type, due_day, has_due_date, weight,
               submitted_day, score, lateness_days
        FROM stg_learner_assessment_events
        """,
    )
    written["assessment_events"] = config.artifact_path("assessment_events")
    events.to_parquet(written["assessment_events"], index=False)

    # --- cohort level views -------------------------------------------------
    outcomes = duck.query_df(con, "SELECT * FROM cohort_outcomes")
    written["cohort_outcomes"] = config.artifact_path("cohort_outcomes")
    outcomes.to_parquet(written["cohort_outcomes"], index=False)

    weekly = duck.query_df(con, "SELECT * FROM weekly_cohort_activity")
    written["weekly_cohort"] = config.artifact_path("weekly_cohort")
    weekly.to_parquet(written["weekly_cohort"], index=False)

    dropoff = duck.query_df(con, "SELECT * FROM activity_dropoff")
    written["dropoff"] = config.artifact_path("dropoff")
    dropoff.to_parquet(written["dropoff"], index=False)

    # --- feature snapshots at each cutoff week ------------------------------
    snapshots: Dict[str, pd.DataFrame] = {}
    for week in cutoff_weeks:
        snapshots[str(week)] = snapshot_for(con, week)
    # The default cutoff is what the dashboard loads as "the" snapshot.
    default_snapshot = snapshots[str(config.DEFAULT_CUTOFF_WEEK)]
    written["learner_snapshot"] = config.artifact_path("learner_snapshot")
    default_snapshot.to_parquet(written["learner_snapshot"], index=False)

    # --- metrics ------------------------------------------------------------
    metrics = {
        "generated_by": "src.features.build",
        "data_dir": str(Path(data_dir).resolve()),
        "cutoff_weeks_built": cutoff_weeks,
        "default_cutoff_week": config.DEFAULT_CUTOFF_WEEK,
        "week_mapping": (
            "week = floor(day / 7) + 1 for day >= 0; day 0 is week 1. "
            "Negative days are pre-course activity, reported separately and "
            "excluded from the weekly series."
        ),
        "grain": grain_report(con),
        "cohort": cohort_metrics(con),
        "snapshot_shapes": {
            week: {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
            for week, df in snapshots.items()
        },
        "snapshot_columns": {week: list(df.columns) for week, df in snapshots.items()},
    }
    written["cohort_metrics"] = config.artifact_path("cohort_metrics")
    written["cohort_metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Keep every cutoff snapshot on disk so week-3 vs week-4 is inspectable.
    for week, df in snapshots.items():
        path = config.PROCESSED_DIR / f"learner_feature_snapshot_w{week}.parquet"
        df.to_parquet(path, index=False)
        written[f"snapshot_w{week}"] = path

    con.close()
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SQL features and artifacts.")
    parser.add_argument("--data-dir", default=str(config.RAW_DIR))
    parser.add_argument(
        "--cutoff-week",
        type=int,
        default=config.DEFAULT_CUTOFF_WEEK,
        help="Default prediction cutoff week.",
    )
    args = parser.parse_args(argv)

    weeks = sorted({args.cutoff_week, config.COMPARISON_CUTOFF_WEEK, config.DEFAULT_CUTOFF_WEEK})
    written = build(Path(args.data_dir), weeks)

    metrics = json.loads(config.artifact_path("cohort_metrics").read_text())
    print("Feature build complete")
    print("=" * 62)
    print(f"  cutoff weeks built      : {weeks}")
    print(f"  learner-course records  : {metrics['cohort']['learner_course_records']:,}")
    for week, shape in sorted(metrics["snapshot_shapes"].items()):
        print(f"  snapshot week {week}         : {shape['rows']:,} rows x {shape['columns']} cols")
    print("  steepest activity drop-offs:")
    for row in metrics["cohort"]["steepest_dropoff_weeks"][:3]:
        print(
            f"      week {int(row['week']):>2}: {row['dropoff_rate'] * 100:5.1f}% "
            f"({int(row['prev_active']):,} -> {int(row['active_learners']):,} active learners)"
        )
    print("\n  artifacts:")
    for key, path in written.items():
        try:
            shown = path.relative_to(config.PROJECT_ROOT)
        except ValueError:
            shown = path
        print(f"      {key:<20} {shown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
