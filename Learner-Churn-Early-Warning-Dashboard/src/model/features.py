"""Feature specification and the cutoff-safety guard.

Two jobs live here:

1. Declare exactly which columns become model features, and how each is
   treated (numeric, categorical, or excluded with a reason).
2. Provide :func:`assert_cutoff_safe`, the programmatic guard that refuses to
   build a feature matrix containing anything outcome-derived or post-cutoff.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

#: Numeric features. All are observed on or before the cutoff.
NUMERIC_FEATURES: Tuple[str, ...] = (
    # Engagement volume
    "total_clicks_to_cutoff",
    "total_active_days_to_cutoff",
    "clicks_per_active_day",
    "pre_course_clicks",
    "pre_course_active_days",
    # Per-week engagement
    "clicks_w1",
    "clicks_w2",
    "clicks_w3",
    "clicks_w4",
    "active_days_w1",
    "active_days_w2",
    "active_days_w3",
    "active_days_w4",
    # Trend
    "click_delta",
    "active_days_delta",
    "click_slope",
    # Recency
    "days_since_last_activity",
    # Assessment progress
    "assessments_due_by_cutoff",
    "submitted_by_cutoff",
    "submission_rate",
    "overdue_count",
    "mean_score",
    "min_score",
    "max_score",
    "first_score",
    "latest_score",
    "score_delta",
    "banked_submissions",
    "weight_due_by_cutoff",
    # Lateness (only where a real due date exists)
    "mean_lateness_days",
    "max_lateness_days",
    "late_submissions",
    "on_time_submissions",
    # Context
    "num_of_prev_attempts",
    "studied_credits",
    "module_presentation_length",
    # Missingness flags: tell the model that "no score" != "score of zero"
    "has_no_score_by_cutoff",
    "has_no_submission_by_cutoff",
    "lateness_unavailable",
)

#: Categorical features. `region` and `imd_band` are included because they are
#: available at prediction time, but see docs/model_card.md: the model card
#: flags them as fairness-sensitive and the dashboard does not use them for
#: intervention decisions.
CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
)

#: Columns that must never become features, with the reason. This list is
#: asserted against the built matrix, so a future edit that adds one of these
#: fails loudly rather than quietly leaking.
FORBIDDEN_FEATURES: Dict[str, str] = {
    "final_result": "the label itself",
    "target": "the label itself",
    "is_positive": "derived from the label",
    "is_negative": "derived from the label",
    "date_unregistration": "unregistration timing is a post-hoc outcome signal",
    "date_registration": "registration timing is a pre-course enrolment artifact",
    "last_active_week": "derived over the whole course, not cutoff-safe",
    "total_clicks": "derived over the whole course, not cutoff-safe",
    "last_active_day_in_window": "raw recency day; days_since_last_activity supersedes it",
    "last_active_week_in_window": "raw recency week; kept for display only",
    "last_submission_day": "raw day offset; superseded by submission features",
    "first_submission_day": "raw day offset; superseded by submission features",
    "mean_assessment_weight": "constant per module-presentation; adds no signal",
}

#: Columns carried for identity, display or joining. Not features.
IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "code_module",
    "code_presentation",
    "id_student",
)


class LeakageError(AssertionError):
    """Raised when the feature matrix contains a non-cutoff-safe column."""


def feature_columns() -> List[str]:
    """Every column the model is allowed to see."""
    return list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


def assert_cutoff_safe(df: pd.DataFrame) -> None:
    """Fail if the frame carries a forbidden or unexpected feature column.

    ``df`` is the assembled feature matrix (features only, no identifiers, no
    label). Any column not declared in :data:`NUMERIC_FEATURES` or
    :data:`CATEGORICAL_FEATURES` is rejected, which means a new SQL column
    cannot silently join the model.
    """
    present = set(df.columns)
    declared = set(feature_columns())

    leaked = sorted(present & set(FORBIDDEN_FEATURES))
    if leaked:
        reasons = "; ".join(f"{c} ({FORBIDDEN_FEATURES[c]})" for c in leaked)
        raise LeakageError(f"Non-cutoff-safe columns present in the feature matrix: {reasons}")

    undeclared = sorted(present - declared)
    if undeclared:
        raise LeakageError(
            "Undeclared columns in the feature matrix: "
            f"{undeclared}. Add them to src/model/features.py with a justification, "
            "or drop them."
        )


def build_feature_matrix(
    df: pd.DataFrame, cutoff_week: int | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a labelled snapshot into (X, y) and assert cutoff safety.

    Week-specific columns above the cutoff are dropped, so the same function
    works for week 3 and week 4 without special-casing.
    """
    cutoff = cutoff_week or config.DEFAULT_CUTOFF_WEEK

    wanted: List[str] = []
    for col in feature_columns():
        # Only keep per-week columns for weeks inside the window.
        if col.startswith(("clicks_w", "active_days_w")):
            week = int(col.rsplit("w", 1)[1])
            if week > cutoff:
                continue
        if col in df.columns:
            wanted.append(col)

    missing = sorted(set(wanted) - set(df.columns))
    if missing:  # pragma: no cover - guarded by tests
        raise KeyError(f"Snapshot is missing expected feature columns: {missing}")

    X = df[wanted].copy()
    assert_cutoff_safe(X)
    y = df["target"].astype(int).copy()
    return X, y
