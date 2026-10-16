"""Data audit.

Reads the raw OULAD files, reports their real shape, and writes a machine
readable audit artifact. Every number the dashboard shows about the dataset
originates here, so that "what is the data" is answerable without re-reading
the CSVs.

Usage
-----
    python -m src.data.audit --data-dir data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src import config
from src.data import schema


def _null_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    return float(df[column].isna().mean())


def audit_table(filename: str, data_dir: Path) -> Dict[str, Any]:
    """Describe one raw table: shape, dtypes, nulls, duplicates, ranges."""
    df = schema.read_raw(filename, data_dir)
    spec = next((s for s in schema.TABLE_SPECS.values() if s.filename == filename), None)

    record: Dict[str, Any] = {
        "filename": filename,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_rates": {c: round(_null_rate(df, c), 6) for c in df.columns},
    }

    if spec and spec.unique_key:
        key = [k for k in spec.unique_key if k in df.columns]
        if key:
            record["unique_key"] = key
            record["duplicate_rows_on_key"] = int(df.duplicated(key).sum())
            record["unique_keys_observed"] = int(df.drop_duplicates(key).shape[0])

    # Date-like day-offset columns: report the observed range.
    for col in ("date", "date_registration", "date_unregistration", "date_submitted"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                record[f"{col}_min"] = int(numeric.min())
                record[f"{col}_max"] = int(numeric.max())
                record[f"{col}_negative_rows"] = int((numeric < 0).sum())

    return record


def run_audit(data_dir: Path | None = None) -> Dict[str, Any]:
    """Audit every table plus the cohort-level facts the model depends on."""
    data_dir = Path(data_dir or config.RAW_DIR)
    absent = schema.missing_files(data_dir)
    if absent:
        raise schema.DatasetError(
            f"Dataset incomplete in {data_dir}. Missing: {', '.join(absent)}. "
            "Run `python -m src.data.download` or follow the README download steps."
        )

    tables: List[Dict[str, Any]] = []
    for name, spec in schema.TABLE_SPECS.items():
        record = audit_table(spec.filename, data_dir)
        record["table"] = name
        record["description"] = spec.description
        record["declared_required_columns"] = list(spec.required_columns)
        record["missing_required_columns"] = schema.validate_columns(
            schema.read_raw(spec.filename, data_dir), spec
        )
        tables.append(record)

    info = schema.read_raw("studentInfo.csv", data_dir)
    courses = schema.read_raw("courses.csv", data_dir)
    assessments = schema.read_raw("assessments.csv", data_dir)

    outcome_counts = info["final_result"].value_counts().to_dict()
    total = int(len(info))

    assessment_due_available = int(assessments["date"].notna().sum())
    assessment_total = int(len(assessments))

    audit: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "source_url": config.OULAD_SOURCE_URL,
        "mirror_url": config.OULAD_MIRROR_URL,
        "tables": tables,
        "cohort": {
            "learner_course_records": total,
            "distinct_learners": int(info["id_student"].nunique()),
            "course_presentations": int(len(courses)),
            "distinct_modules": int(courses["code_module"].nunique()),
            "outcome_counts": {k: int(v) for k, v in outcome_counts.items()},
            "outcome_share": {
                k: round(v / total, 6) for k, v in outcome_counts.items()
            },
            "unexpected_outcome_labels": schema.validate_outcome_labels(
                info["final_result"]
            ),
            "duplicate_learner_course_rows": int(
                info.duplicated(["code_module", "code_presentation", "id_student"]).sum()
            ),
        },
        "assessment_calendar": {
            "total_assessments": assessment_total,
            "assessments_with_due_date": assessment_due_available,
            "assessments_without_due_date": assessment_total - assessment_due_available,
            "due_date_coverage": round(assessment_due_available / assessment_total, 6)
            if assessment_total
            else None,
            "types": {k: int(v) for k, v in assessments["assessment_type"].value_counts().items()},
            "due_date_by_type": {
                str(t): {
                    "total": int(grp.shape[0]),
                    "with_due_date": int(grp["date"].notna().sum()),
                }
                for t, grp in assessments.groupby("assessment_type")
            },
        },
        "warnings": [],
    }

    # Surface the things a reader would otherwise have to discover the hard way.
    if audit["cohort"]["duplicate_learner_course_rows"]:
        audit["warnings"].append(
            "studentInfo has duplicate learner-course rows; grain assumptions need review."
        )
    if audit["assessment_calendar"]["due_date_coverage"] < 1.0:
        pct = audit["assessment_calendar"]["due_date_coverage"] * 100
        audit["warnings"].append(
            f"Only {pct:.1f}% of assessments carry a due date. "
            "Lateness is computed on that subset only and is reported as unavailable "
            "elsewhere; no due dates are imputed."
        )
    if audit["cohort"]["unexpected_outcome_labels"]:
        audit["warnings"].append(
            f"Unexpected final_result values: {audit['cohort']['unexpected_outcome_labels']}"
        )

    return audit


def write_audit(audit: Dict[str, Any]) -> Path:
    config.ensure_dirs()
    out = config.artifact_path("data_audit")
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def print_summary(audit: Dict[str, Any]) -> None:
    cohort = audit["cohort"]
    cal = audit["assessment_calendar"]
    print("OULAD data audit")
    print("=" * 62)
    print(f"  learner-course records : {cohort['learner_course_records']:,}")
    print(f"  distinct learners      : {cohort['distinct_learners']:,}")
    print(f"  course presentations   : {cohort['course_presentations']}")
    print(f"  distinct modules       : {cohort['distinct_modules']}")
    print("  outcome mix            :")
    for label, count in sorted(
        cohort["outcome_counts"].items(), key=lambda kv: -kv[1]
    ):
        share = cohort["outcome_share"][label] * 100
        print(f"      {label:<12} {count:>7,}  ({share:5.1f}%)")
    print(
        f"  assessments with due date: {cal['assessments_with_due_date']}"
        f"/{cal['total_assessments']} ({cal['due_date_coverage'] * 100:.1f}%)"
    )
    print("\n  tables:")
    for table in audit["tables"]:
        flag = "  <-- MISSING COLUMNS" if table["missing_required_columns"] else ""
        print(f"      {table['filename']:<26} {table['rows']:>10,} rows{flag}")
    if audit["warnings"]:
        print("\n  warnings:")
        for warning in audit["warnings"]:
            print(f"      - {warning}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the raw OULAD dataset.")
    parser.add_argument("--data-dir", default=str(config.RAW_DIR))
    parser.add_argument("--no-write", action="store_true", help="Do not write the artifact.")
    args = parser.parse_args(argv)

    audit = run_audit(Path(args.data_dir))
    print_summary(audit)
    if not args.no_write:
        out = write_audit(audit)
        print(f"\n  audit written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
