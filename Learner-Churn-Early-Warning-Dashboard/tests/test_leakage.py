"""Leakage tests — the most important tests in this project.

If any of these fail, every metric in the dashboard is suspect. They assert
the cutoff contract directly against the SQL layer:

    a week-4 snapshot must not move when week-5+ data changes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.db import duck
from src.model.features import (
    FORBIDDEN_FEATURES,
    LeakageError,
    assert_cutoff_safe,
    build_feature_matrix,
)
from src.model.target import label_frame


def _week4_vector(con, student: int) -> dict:
    duck.build_feature_snapshot(con, config.DEFAULT_CUTOFF_WEEK)
    row = con.execute(
        """
        SELECT total_clicks_to_cutoff, total_active_days_to_cutoff, click_delta,
               click_slope, days_since_last_activity, active_days_delta
        FROM learner_feature_snapshot WHERE id_student = ?
        """,
        [student],
    ).fetchdf()
    return row.iloc[0].to_dict()


def test_post_cutoff_activity_does_not_change_week4_features(
    fixture_data_dir, fixture_config
):
    """Inject heavy week-5..week-8 activity and prove week-4 features are identical.

    This is the core leakage assertion. If a future refactor lets post-cutoff
    clicks into the week-4 window, this test fails.
    """
    from src.features import build as feature_build

    con = feature_build.build_connection(fixture_data_dir)
    before = _week4_vector(con, 1002)
    con.close()

    # Append activity strictly AFTER the week-4 cutoff (day 28 onwards).
    vle_path = fixture_data_dir / "studentVle.csv"
    extra = pd.DataFrame(
        {
            "code_module": ["AAA"] * 4,
            "code_presentation": ["2013J"] * 4,
            "id_student": [1002] * 4,
            "id_site": [546943] * 4,
            "date": [28, 35, 42, 49],          # weeks 5, 6, 7, 8
            "sum_click": [9999, 9999, 9999, 9999],
        }
    )
    pd.concat([pd.read_csv(vle_path), extra], ignore_index=True).to_csv(
        vle_path, index=False
    )

    con2 = feature_build.build_connection(fixture_data_dir)
    after = _week4_vector(con2, 1002)
    con2.close()

    assert before == after, (
        "Week-4 features changed after adding post-cutoff activity — "
        f"leakage. before={before} after={after}"
    )


def test_post_cutoff_submission_does_not_change_week4_features(
    fixture_data_dir, fixture_config
):
    """A submission dated after the cutoff must not affect the week-4 snapshot."""
    from src.features import build as feature_build

    con = feature_build.build_connection(fixture_data_dir)
    duck.build_feature_snapshot(con, 4)
    before = con.execute(
        """
        SELECT submitted_by_cutoff, mean_score, overdue_count, submission_rate
        FROM learner_feature_snapshot WHERE id_student = 1002
        """
    ).fetchdf().iloc[0].to_dict()
    con.close()

    sa_path = fixture_data_dir / "studentAssessment.csv"
    extra = pd.DataFrame(
        {
            "id_assessment": [1753, 1753],
            "id_student": [1002, 1002],
            "date_submitted": [100, 200],   # far after the cutoff
            "is_banked": [0, 0],
            "score": [100, 100],
        }
    )
    pd.concat([pd.read_csv(sa_path), extra], ignore_index=True).to_csv(
        sa_path, index=False
    )

    con2 = feature_build.build_connection(fixture_data_dir)
    duck.build_feature_snapshot(con2, 4)
    after = con2.execute(
        """
        SELECT submitted_by_cutoff, mean_score, overdue_count, submission_rate
        FROM learner_feature_snapshot WHERE id_student = 1002
        """
    ).fetchdf().iloc[0].to_dict()
    con2.close()

    assert before == after, (
        f"Post-cutoff submissions changed week-4 assessment features: {before} -> {after}"
    )


def test_cutoff_week_3_uses_fewer_weeks_than_week_4(fixture_data_dir, fixture_config):
    """The week-3 snapshot must not carry week-4 columns."""
    from src.features import build as feature_build

    con = feature_build.build_connection(fixture_data_dir)
    duck.build_feature_snapshot(con, 3)
    cols3 = [
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'learner_feature_snapshot'"
        ).fetchall()
    ]
    con.close()
    assert "clicks_w3" in cols3
    assert "clicks_w4" not in cols3, "week-3 snapshot must not contain week-4 features"


def test_snapshot_never_contains_forbidden_columns(built_fixture):
    """No deny-listed column may reach the model feature matrix.

    The snapshot itself legitimately carries ``final_result`` (the label) and
    raw day/week columns for display and auditing. What matters is that none
    of them survive into the matrix the model is fitted on, so the assertion
    is made against ``build_feature_matrix`` rather than the raw snapshot.
    """
    duck.build_feature_snapshot(built_fixture, 4)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()
    labelled = label_frame(snap, "primary")
    X, _ = build_feature_matrix(labelled, cutoff_week=4)

    leaked = set(X.columns) & set(FORBIDDEN_FEATURES)
    assert not leaked, f"Feature matrix contains forbidden columns: {sorted(leaked)}"

    # And the guard itself must still refuse them if someone bypasses the
    # allow-list by hand.
    for column in ("final_result", "date_unregistration", "total_clicks"):
        probe = X.copy()
        probe[column] = 0
        with pytest.raises(LeakageError):
            assert_cutoff_safe(probe)


def test_assert_cutoff_safe_rejects_the_label():
    X = pd.DataFrame({"total_clicks_to_cutoff": [1.0], "final_result": ["Pass"]})
    with pytest.raises(LeakageError, match="final_result"):
        assert_cutoff_safe(X)


def test_assert_cutoff_safe_rejects_unregistration():
    X = pd.DataFrame({"total_clicks_to_cutoff": [1.0], "date_unregistration": [20]})
    with pytest.raises(LeakageError):
        assert_cutoff_safe(X)


def test_assert_cutoff_safe_rejects_undeclared_columns():
    """A new SQL column cannot silently join the model."""
    X = pd.DataFrame({"total_clicks_to_cutoff": [1.0], "mystery_future_signal": [1.0]})
    with pytest.raises(LeakageError, match="Undeclared"):
        assert_cutoff_safe(X)


def test_build_feature_matrix_drops_weeks_above_the_cutoff(built_fixture):
    """A week-3 matrix must contain clicks_w3 but never clicks_w4."""
    duck.build_feature_snapshot(built_fixture, 3)
    snap = built_fixture.execute("SELECT * FROM learner_feature_snapshot").fetchdf()
    labelled = label_frame(snap, "primary")
    X, y = build_feature_matrix(labelled, cutoff_week=3)

    assert "clicks_w3" in X.columns
    assert "clicks_w4" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()) <= {0, 1}


def test_group_split_never_shares_a_learning_instance():
    """Grouped split: no learner-course presentation appears on both sides."""
    from src.model import pipeline as pl

    df = pd.DataFrame(
        {
            "code_module": ["AAA"] * 20 + ["BBB"] * 20,
            "code_presentation": ["2013J"] * 40,
            "id_student": list(range(40)),
            "target": [0, 1] * 20,
        }
    )
    train, test = pl.group_train_test_split(df)
    train_keys = set(pl.group_key(train))
    test_keys = set(pl.group_key(test))
    assert not (train_keys & test_keys)
    assert len(train) + len(test) == len(df)
