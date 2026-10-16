"""Central configuration and shared constants.

Every tunable that changes an analytical result lives here, so the SQL layer,
the training pipeline, the tests and the dashboard all read the *same*
definition. If you change a cutoff or a target definition here, every layer
follows.

Environment overrides
---------------------
LCWD_DATA_DIR       root of the data directory (default: <project>/data)
LCWD_ARTIFACTS_DIR  root of the generated artifacts directory
LCWD_INTERVENTION_DB  path to the local interventions database
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("LCWD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("LCWD_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
SQL_DIR = PROJECT_ROOT / "sql"
DOCS_DIR = PROJECT_ROOT / "docs"

INTERVENTION_DB = Path(
    os.environ.get("LCWD_INTERVENTION_DB", PROCESSED_DIR / "interventions.sqlite")
)

# --------------------------------------------------------------------------
# OULAD source files
# --------------------------------------------------------------------------
OULAD_FILES: Tuple[str, ...] = (
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
)

OULAD_SOURCE_URL = "https://analyse.kmi.open.ac.uk/open_dataset"
OULAD_MIRROR_URL = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open+university+learning+analytics+dataset.zip"
)

# --------------------------------------------------------------------------
# Week mapping
# --------------------------------------------------------------------------
# OULAD `date` fields are day offsets from the start of the presentation.
# Day 0 is the first day of the presentation. We define week 1 as days 0-6,
# week 2 as days 7-13, and so on:
#
#     week = floor(day / 7) + 1     for day >= 0
#
# Negative day values are pre-course activity (registration, orientation,
# early access). They are NOT silently dropped: they are aggregated into a
# separate `pre_course_clicks` feature and excluded from the weekly series.
DAYS_PER_WEEK = 7

DEFAULT_CUTOFF_WEEK = 4
COMPARISON_CUTOFF_WEEK = 3
# The SQL wide matrix materialises weeks 1..MAX_FEATURE_WEEK. The snapshot for
# a given cutoff week only ever selects columns up to that cutoff.
MAX_FEATURE_WEEK = 8
SUPPORTED_CUTOFF_WEEKS: Tuple[int, ...] = (3, 4)

# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------
# OULAD `final_result` has four values. The primary target deliberately
# EXCLUDES `Fail` rather than folding it into the positive class: failing and
# withdrawing are different behaviours, and mixing them silently would make
# the headline number uninterpretable. A documented sensitivity target counts
# Fail as non-completion so the effect of that choice is visible.
OUTCOME_LABELS: Tuple[str, ...] = ("Pass", "Distinction", "Fail", "Withdrawn")

TARGETS: Dict[str, Dict[str, object]] = {
    "primary": {
        "key": "primary",
        "label": "Primary - Withdrawn vs Pass/Distinction",
        "description": (
            "Positive = Withdrawn. Negative = Pass or Distinction. "
            "Fail rows are excluded from the modelling cohort entirely."
        ),
        "positive_labels": ("Withdrawn",),
        "negative_labels": ("Pass", "Distinction"),
        "excluded_labels": ("Fail",),
    },
    "fail_as_churn": {
        "key": "fail_as_churn",
        "label": "Sensitivity - Withdrawn or Fail vs Pass/Distinction",
        "description": (
            "Positive = Withdrawn or Fail. Negative = Pass or Distinction. "
            "Shows how much the headline result depends on excluding Fail."
        ),
        "positive_labels": ("Withdrawn", "Fail"),
        "negative_labels": ("Pass", "Distinction"),
        "excluded_labels": (),
    },
}
DEFAULT_TARGET = "primary"

# --------------------------------------------------------------------------
# Risk bands and operating threshold
# --------------------------------------------------------------------------
# Bands are a presentation device for the queue, not a claim about the learner.
DEFAULT_RISK_BANDS: Dict[str, float] = {"low": 0.33, "high": 0.66}

# Fallback operating threshold if a trained model has not been scored yet.
DEFAULT_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
GROUP_KEY_COLUMNS: Tuple[str, ...] = ("code_module", "code_presentation", "id_student")
LEARNER_KEY_COLUMNS: Tuple[str, ...] = ("code_module", "code_presentation", "id_student")

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Artifact filenames (parquet / json / csv / joblib)
# --------------------------------------------------------------------------
ARTIFACTS: Dict[str, str] = {
    "data_audit": "data_audit.json",
    "cohort_metrics": "cohort_metrics.json",
    "model_metrics": "model_metrics.json",
    "build_report": "build_report.md",
    "model": "churn_model.joblib",
    "risk_queue": "risk_queue.parquet",
    "learner_week": "learner_week_engagement.parquet",
    "learner_snapshot": "learner_feature_snapshot.parquet",
    "assessment_events": "learner_assessment_events.parquet",
    "cohort_outcomes": "cohort_outcomes.parquet",
    "weekly_cohort": "weekly_cohort_activity.parquet",
    "dropoff": "activity_dropoff.parquet",
}


def artifact_path(key: str) -> Path:
    """Absolute path for a named artifact."""
    return ARTIFACTS_DIR / ARTIFACTS[key]


# --------------------------------------------------------------------------
# Intervention workflow vocabulary
# --------------------------------------------------------------------------
INTERVENTION_STATUSES: Tuple[str, ...] = (
    "New",
    "Reviewed",
    "Assigned",
    "Contact attempted",
    "Resolved",
    "No response",
    "Not appropriate",
)

INTERVENTION_ACTIONS: Tuple[str, ...] = (
    "Supportive check-in",
    "Clarify assessment requirements",
    "Share relevant learning resource",
    "Invite to office hours",
)


def ensure_dirs() -> None:
    """Create the directories the pipeline writes into."""
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
