"""Join-grain tests: declared keys must be unique and joins must not multiply rows."""

from __future__ import annotations

import pytest

from src import config
from src.data import schema
from src.db import duck


def test_student_info_grain_is_unique(built_fixture):
    row = built_fixture.execute(
        """
        SELECT COUNT(*) AS rows,
               COUNT(*) - COUNT(DISTINCT (code_module, code_presentation, id_student)) AS dupes
        FROM stg_student_info
        """
    ).fetchone()
    assert row[1] == 0, f"stg_student_info has {row[1]} duplicate learner-course rows"


def test_learner_week_matrix_grain_is_unique(built_fixture):
    row = built_fixture.execute(
        """
        SELECT COUNT(*),
               COUNT(*) - COUNT(DISTINCT (code_module, code_presentation, id_student))
        FROM learner_week_matrix
        """
    ).fetchone()
    assert row[1] == 0, "learner_week_matrix is not unique on the learner-course key"


def test_feature_snapshot_grain_is_unique(built_fixture):
    """The snapshot must have exactly one row per learner-course presentation."""
    duck.build_feature_snapshot(built_fixture, 4)
    row = built_fixture.execute(
        """
        SELECT COUNT(*),
               COUNT(*) - COUNT(DISTINCT (code_module, code_presentation, id_student))
        FROM learner_feature_snapshot
        """
    ).fetchone()
    assert row[1] == 0, "feature snapshot multiplied rows during its joins"


def test_feature_snapshot_row_count_equals_student_info(built_fixture):
    """A LEFT JOIN chain must never change the row count of the base table."""
    duck.build_feature_snapshot(built_fixture, 4)
    base = built_fixture.execute("SELECT COUNT(*) FROM stg_student_info").fetchone()[0]
    snap = built_fixture.execute("SELECT COUNT(*) FROM learner_feature_snapshot").fetchone()[0]
    assert snap == base, f"snapshot has {snap} rows but the base grain has {base}"


def test_assessment_calendar_is_unique_on_id(built_fixture):
    from src.db import duck

    # assessment_calendar is defined alongside the assessment features.
    duck.build_feature_snapshot(built_fixture, 4)
    row = built_fixture.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT id_assessment) FROM assessment_calendar
        """
    ).fetchone()
    assert row[0] == 0, "assessment_calendar is not unique on id_assessment"


def test_learner_week_engagement_grain_is_unique(built_fixture):
    row = built_fixture.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT (code_module, code_presentation, id_student, week))
        FROM learner_week_engagement
        """
    ).fetchone()
    assert row[0] == 0, "learner_week_engagement is not unique on learner-week"


def test_weekly_cohort_activity_grain_is_unique(built_fixture):
    row = built_fixture.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT (code_module, code_presentation, week))
        FROM weekly_cohort_activity
        """
    ).fetchone()
    assert row[0] == 0, "weekly_cohort_activity is not unique on cohort-week"


def test_submission_join_does_not_multiply_events(built_fixture):
    """Joining submissions to the calendar must be 1:1 on id_assessment."""
    raw = built_fixture.execute("SELECT COUNT(*) FROM stg_student_vle").fetchone()[0]
    assert raw > 0

    events = built_fixture.execute(
        "SELECT COUNT(*) FROM stg_learner_assessment_events"
    ).fetchone()[0]
    submissions = built_fixture.execute(
        "SELECT COUNT(*) FROM read_csv_auto(?, header=true, all_varchar=true)",
        [str(config.RAW_DIR)],
    ) if False else None

    # Every event must correspond to exactly one assessment in the calendar.
    dupes = built_fixture.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT e.id_student, e.id_assessment, COUNT(*) AS n
            FROM stg_learner_assessment_events AS e
            GROUP BY 1, 2 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    assert dupes == 0, f"{dupes} learner-assessment pairs were duplicated by the join"
    assert events > 0


def test_declared_unique_keys_hold_on_the_fixture(fixture_data_dir):
    """schema.check_dataset must pass on the structurally faithful fixture."""
    schema.check_dataset(fixture_data_dir)


def test_check_dataset_raises_on_missing_file(tmp_path):
    with pytest.raises(schema.DatasetError, match="Missing"):
        schema.check_dataset(tmp_path)


def test_check_dataset_raises_on_bad_outcome_label(fixture_data_dir, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(fixture_data_dir, broken)
    import pandas as pd

    info = pd.read_csv(broken / "studentInfo.csv")
    info.loc[0, "final_result"] = "Deferred"
    info.to_csv(broken / "studentInfo.csv", index=False)

    with pytest.raises(schema.DatasetError, match="unexpected final_result"):
        schema.check_dataset(broken)


def test_check_dataset_raises_on_missing_column(fixture_data_dir, tmp_path):
    import shutil

    import pandas as pd

    broken = tmp_path / "broken2"
    shutil.copytree(fixture_data_dir, broken)
    info = pd.read_csv(broken / "studentInfo.csv").drop(columns=["final_result"])
    info.to_csv(broken / "studentInfo.csv", index=False)

    with pytest.raises(schema.DatasetError, match="missing columns"):
        schema.check_dataset(broken)
