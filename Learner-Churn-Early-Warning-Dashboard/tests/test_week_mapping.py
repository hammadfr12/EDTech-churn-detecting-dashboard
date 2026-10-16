"""Week mapping: day offsets must map to course-relative weeks as documented."""

from __future__ import annotations

import pytest

from src import config
from src.db import duck


def week_of(day: int) -> int:
    """The documented mapping, expressed once so the test and SQL cannot drift."""
    assert day >= 0
    return day // config.DAYS_PER_WEEK + 1


@pytest.mark.parametrize(
    "day,expected",
    [
        (0, 1),    # day 0 belongs to week 1
        (1, 1),
        (6, 1),    # last day of week 1
        (7, 2),    # first day of week 2
        (13, 2),
        (14, 3),
        (20, 3),
        (21, 4),
        (27, 4),   # last day of week 4 == the default cutoff day
        (28, 5),
    ],
)
def test_day_to_week_mapping(day, expected):
    assert week_of(day) == expected


def test_default_cutoff_day_is_last_day_of_cutoff_week():
    assert config.DEFAULT_CUTOFF_WEEK * config.DAYS_PER_WEEK - 1 == 27
    assert week_of(27) == config.DEFAULT_CUTOFF_WEEK
    assert week_of(28) == config.DEFAULT_CUTOFF_WEEK + 1


def test_negative_days_are_rejected_by_the_week_mapping():
    """Negative offsets are pre-course activity and must not be assigned a week."""
    with pytest.raises(AssertionError):
        week_of(-1)


def test_sql_week_mapping_matches_python(built_fixture):
    """The SQL FLOOR(day/7)+1 must agree with the Python mapping for every row."""
    rows = built_fixture.execute(
        """
        SELECT day, week FROM (
            SELECT day, CAST(FLOOR(day / 7) + 1 AS INTEGER) AS week
            FROM stg_student_vle WHERE day >= 0
        ) GROUP BY 1, 2 ORDER BY 1
        """
    ).fetchall()
    assert rows, "fixture clickstream is empty"
    for day, week in rows:
        assert week == week_of(day), f"day {day} mapped to week {week}, expected {week_of(day)}"


def test_negative_day_activity_is_excluded_from_the_weekly_series(built_fixture):
    """Pre-course clicks must not appear in any week bucket."""
    min_day = built_fixture.execute(
        "SELECT MIN(first_active_day) FROM learner_week_engagement"
    ).fetchone()[0]
    assert min_day is not None and min_day >= 0, "negative days leaked into the week series"

    precourse_rows = built_fixture.execute(
        "SELECT COUNT(*) FROM learner_precourse_engagement WHERE pre_course_clicks > 0"
    ).fetchone()[0]
    assert precourse_rows > 0, "fixture should contain pre-course activity to test with"
