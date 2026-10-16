"""Declared OULAD schemas and file-level validation.

The audit compares what is on disk against what this module says should be
there. Nothing downstream assumes a column exists: the audit reports the gap
and the pipeline refuses to build features from an invalid dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src import config


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or does not match the schema."""


@dataclass(frozen=True)
class TableSpec:
    """What we expect a single OULAD CSV to look like."""

    filename: str
    required_columns: Tuple[str, ...]
    #: Columns that must be unique in combination (the table's declared grain).
    unique_key: Tuple[str, ...] = ()
    #: Columns allowed to contain the OULAD missing-value marker '?'.
    nullable_columns: Tuple[str, ...] = ()
    description: str = ""


TABLE_SPECS: Dict[str, TableSpec] = {
    "courses": TableSpec(
        filename="courses.csv",
        required_columns=("code_module", "code_presentation", "module_presentation_length"),
        unique_key=("code_module", "code_presentation"),
        description="Course presentation metadata and length in days.",
    ),
    "assessments": TableSpec(
        filename="assessments.csv",
        required_columns=(
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ),
        unique_key=("id_assessment",),
        nullable_columns=("date",),
        description="Assessment calendar. `date` is the due day offset, '?' for most exams.",
    ),
    "vle": TableSpec(
        filename="vle.csv",
        required_columns=("id_site", "code_module", "code_presentation", "activity_type"),
        unique_key=("id_site",),
        nullable_columns=("week_from", "week_to"),
        description="VLE activity catalogue.",
    ),
    "studentInfo": TableSpec(
        filename="studentInfo.csv",
        required_columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "final_result",
        ),
        unique_key=("code_module", "code_presentation", "id_student"),
        nullable_columns=("imd_band",),
        description="Learner attributes, enrolment and final outcome. One row per learner-course.",
    ),
    "studentRegistration": TableSpec(
        filename="studentRegistration.csv",
        required_columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "date_registration",
            "date_unregistration",
        ),
        unique_key=("code_module", "code_presentation", "id_student"),
        nullable_columns=("date_registration", "date_unregistration"),
        description="Registration and unregistration day offsets.",
    ),
    "studentAssessment": TableSpec(
        filename="studentAssessment.csv",
        required_columns=("id_assessment", "id_student", "date_submitted", "is_banked", "score"),
        nullable_columns=("score",),
        description="One row per learner submission.",
    ),
    "studentVle": TableSpec(
        filename="studentVle.csv",
        required_columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ),
        nullable_columns=(),
        description="Daily learner interactions with VLE activities.",
    ),
}

OUTCOME_LABELS: Tuple[str, ...] = config.OUTCOME_LABELS


def read_raw(filename: str, data_dir: Path) -> pd.DataFrame:
    """Read a raw OULAD CSV, keeping '?' as a real missing marker.

    ``keep_default_na`` is left on so empty fields also become NaN, and '?' is
    added to the NA set so both encodings collapse to the same thing.
    """
    path = Path(data_dir) / filename
    if not path.exists():
        raise DatasetError(
            f"Missing required OULAD file: {path}. "
            f"Run `python -m src.data.download` or see README for manual steps."
        )
    return pd.read_csv(path, na_values=["?"], low_memory=False)


def missing_files(data_dir: Path) -> List[str]:
    """Return the filenames of required OULAD files that are not present."""
    data_dir = Path(data_dir)
    return [spec.filename for spec in TABLE_SPECS.values() if not (data_dir / spec.filename).exists()]


def validate_columns(df: pd.DataFrame, spec: TableSpec) -> List[str]:
    """Return a list of required columns that are absent."""
    return [c for c in spec.required_columns if c not in df.columns]


def validate_outcome_labels(series: pd.Series) -> List[str]:
    """Return any outcome labels that are not part of the documented set."""
    observed = set(series.dropna().astype(str).unique())
    return sorted(observed - set(OUTCOME_LABELS))


def check_dataset(data_dir: Path) -> None:
    """Raise :class:`DatasetError` unless the dataset is complete and sane."""
    absent = missing_files(data_dir)
    if absent:
        raise DatasetError(
            "Dataset incomplete. Missing: " + ", ".join(absent) + f" in {data_dir}"
        )

    problems: List[str] = []
    for spec in TABLE_SPECS.values():
        df = read_raw(spec.filename, data_dir)
        missing_cols = validate_columns(df, spec)
        if missing_cols:
            problems.append(f"{spec.filename}: missing columns {missing_cols}")
            continue
        if spec.unique_key:
            dupes = int(df.duplicated(list(spec.unique_key)).sum())
            if dupes:
                problems.append(
                    f"{spec.filename}: {dupes} duplicate rows on key {spec.unique_key}"
                )

    info = read_raw("studentInfo.csv", data_dir)
    # Only check labels if the column is actually present: the column check
    # above has already recorded the missing column, and reaching for it here
    # would replace a clear validation message with a KeyError.
    if "final_result" in info.columns:
        bad_labels = validate_outcome_labels(info["final_result"])
        if bad_labels:
            problems.append(f"studentInfo.csv: unexpected final_result values {bad_labels}")

    if problems:
        raise DatasetError("Dataset failed validation:\n  - " + "\n  - ".join(problems))
