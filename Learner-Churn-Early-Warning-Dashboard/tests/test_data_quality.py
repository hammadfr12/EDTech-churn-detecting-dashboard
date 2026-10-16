"""Data tests: required columns, valid labels, sane value ranges."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data import schema


def test_all_expected_files_exist_in_fixture(fixture_data_dir):
    assert schema.missing_files(fixture_data_dir) == []


def test_click_counts_are_non_negative(built_fixture):
    row = built_fixture.execute(
        "SELECT COUNT(*) FROM stg_student_vle WHERE sum_click < 0"
    ).fetchone()
    assert row[0] == 0, "negative click counts found"


def test_sum_click_has_no_nulls(built_fixture):
    row = built_fixture.execute(
        "SELECT COUNT(*) FROM stg_student_vle WHERE sum_click IS NULL"
    ).fetchone()
    assert row[0] == 0


def test_outcome_labels_are_from_the_documented_set(fixture_data_dir):
    info = schema.read_raw("studentInfo.csv", fixture_data_dir)
    assert schema.validate_outcome_labels(info["final_result"]) == []


def test_unknown_outcome_labels_are_detected():
    series = pd.Series(["Pass", "Deferred", "Withdrawn"])
    assert schema.validate_outcome_labels(series) == ["Deferred"]


def test_probabilities_are_bounded(built_fixture, fixture_config):
    """Every score the model produces must lie in [0, 1]."""
    from src.db import duck
    from src.model import train as train_module

    # Build the week-4 snapshot from the fixture, write it where training
    # expects to find it, then train and inspect the produced queue.
    duck.build_feature_snapshot(built_fixture, 4)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()
    snap.to_parquet(fixture_config["artifacts"] / "learner_feature_snapshot.parquet", index=False)

    metrics = train_module.train(cutoff_week=4, compare_tree=False)
    queue = pd.read_parquet(fixture_config["artifacts"] / "risk_queue.parquet")

    assert queue["risk_probability"].between(0.0, 1.0).all(), "probabilities outside [0, 1]"
    assert queue["risk_probability"].notna().all()
    assert 0.0 <= metrics["test_metrics"]["at_threshold"]["threshold"] <= 1.0


def test_risk_band_assignment_covers_every_score():
    from src.model.train import assign_risk_bands

    scores = pd.Series([0.0, 0.1, 0.33, 0.5, 0.66, 0.9, 1.0])
    bands = assign_risk_bands(scores, {"low": 0.33, "high": 0.66})
    assert set(bands) <= {"Low", "Medium", "High"}
    assert bands.iloc[0] == "Low"
    assert bands.iloc[-1] == "High"
    assert bands.iloc[3] == "Medium"


def test_risk_bands_are_monotonic_in_score():
    from src.model.train import assign_risk_bands

    scores = pd.Series([i / 100 for i in range(101)])
    bands = assign_risk_bands(scores, {"low": 0.33, "high": 0.66})
    rank = {"Low": 0, "Medium": 1, "High": 2}
    ranks = bands.map(rank).tolist()
    assert ranks == sorted(ranks), "band must not decrease as the score rises"


def test_negative_click_features_are_impossible(built_fixture):
    """Weekly click columns are counts and must never be negative."""
    from src.db import duck

    duck.build_feature_snapshot(built_fixture, 4)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()
    for col in [c for c in snap.columns if c.startswith("clicks_w")]:
        assert (snap[col].dropna() >= 0).all(), f"{col} contains negative values"


def test_week_columns_are_never_null(built_fixture):
    """A week with no activity is a genuine zero, so the column must be filled."""
    from src.db import duck

    duck.build_feature_snapshot(built_fixture, 4)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()
    for col in [c for c in snap.columns if c.startswith(("clicks_w", "active_days_w"))]:
        assert snap[col].notna().all(), f"{col} contains nulls; absent activity must be 0"


def test_missing_score_is_not_zero(built_fixture):
    """A learner with no submissions must have a null mean_score and a flag."""
    from src.db import duck

    duck.build_feature_snapshot(built_fixture, 4)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()

    no_subs = snap[snap["has_no_submission_by_cutoff"]]
    if len(no_subs):
        assert no_subs["mean_score"].isna().all(), (
            "learners with no submissions must not have a score of 0"
        )


def test_lateness_is_null_when_no_due_date_exists(built_fixture):
    """Exam assessments have no due date, so lateness must be unavailable there."""
    row = built_fixture.execute(
        """
        SELECT COUNT(*) FROM stg_learner_assessment_events
        WHERE has_due_date = FALSE AND lateness_days IS NOT NULL
        """
    ).fetchone()
    assert row[0] == 0, "lateness was computed for an assessment with no due date"


def test_lateness_is_computed_when_due_date_exists(built_fixture):
    row = built_fixture.execute(
        """
        SELECT COUNT(*) FROM stg_learner_assessment_events
        WHERE has_due_date = TRUE AND lateness_days IS NOT NULL
        """
    ).fetchone()
    assert row[0] > 0, "lateness should be computable for dated assessments in the fixture"


def test_unregistration_never_appears_in_features(built_fixture):
    from src.db import duck

    duck.build_feature_snapshot(built_fixture, 4)
    cols = built_fixture.execute("SELECT * FROM learner_feature_snapshot LIMIT 1").fetchdf().columns
    assert "date_unregistration" not in cols
