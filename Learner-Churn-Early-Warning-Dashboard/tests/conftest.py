"""Shared test fixtures.

Most tests run against a small synthetic fixture dataset rather than the real
451 MB OULAD download, so the suite is fast and runs on a fresh clone with no
data. The fixture is deliberately built to exercise the awkward cases:

* a learner with zero activity in some weeks (genuine zeros)
* a learner with no submissions at all (missing score, NOT zero)
* a learner who unregisters early (must never leak into features)
* an assessment with no due date (lateness must be unavailable, not invented)
* pre-course activity on negative day offsets
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def _make_fixture_dataset(data_dir: Path) -> Path:
    """Write a small but structurally faithful OULAD dataset to ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    courses = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "BBB"],
            "code_presentation": ["2013J", "2014J", "2013J"],
            "module_presentation_length": [268, 269, 240],
        }
    )
    courses.to_csv(data_dir / "courses.csv", index=False)

    # 12 learners across 3 presentations, all four outcome labels present.
    students = [
        # module, presentation, id, gender, region, education, imd, age, prev, credits, disability, result
        ("AAA", "2013J", 1001, "M", "London", "HE Qualification", "90-100%", "0-35", 0, 240, "N", "Pass"),
        ("AAA", "2013J", 1002, "F", "Scotland", "A Level", "20-30%", "35-55", 0, 60, "N", "Withdrawn"),
        ("AAA", "2013J", 1003, "M", "London", "HE Qualification", "50-60%", "0-35", 1, 120, "N", "Distinction"),
        ("AAA", "2013J", 1004, "F", "Wales", "Lower Than A Level", None, "55<=", 0, 60, "Y", "Fail"),
        ("AAA", "2014J", 1005, "M", "London", "HE Qualification", "70-80%", "0-35", 0, 240, "N", "Withdrawn"),
        ("AAA", "2014J", 1006, "F", "Scotland", "A Level", "30-40%", "35-55", 0, 60, "N", "Pass"),
        ("BBB", "2013J", 1007, "M", "London", "Post Graduate", "90-100%", "35-55", 0, 120, "N", "Distinction"),
        ("BBB", "2013J", 1008, "F", "Wales", "HE Qualification", "10-20%", "0-35", 2, 60, "N", "Withdrawn"),
        ("BBB", "2013J", 1009, "M", "Scotland", "A Level", "40-50%", "55<=", 0, 240, "N", "Pass"),
        ("BBB", "2013J", 1010, "F", "London", "HE Qualification", "80-90%", "0-35", 0, 60, "N", "Fail"),
        ("BBB", "2013J", 1011, "M", "Wales", "A Level", "60-70%", "35-55", 0, 120, "N", "Withdrawn"),
        ("BBB", "2013J", 1012, "F", "Scotland", "HE Qualification", "50-60%", "0-35", 1, 240, "N", "Pass"),
    ]
    student_info = pd.DataFrame(
        students,
        columns=[
            "code_module", "code_presentation", "id_student", "gender", "region",
            "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
            "studied_credits", "disability", "final_result",
        ],
    )
    student_info.to_csv(data_dir / "studentInfo.csv", index=False)

    # Learner 1002 unregisters at day 20 — inside the observation window but
    # still an outcome signal, so it must never become a feature.
    registration = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB", "BBB", "BBB"],
            "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2014J", "2014J", "2013J", "2013J", "2013J", "2013J", "2013J", "2013J"],
            "id_student": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012],
            "date_registration": [-40, -30, -25, -20, -35, -28, -22, -18, -26, -33, -19, -24],
            "date_unregistration": ["?", 20, "?", "?", 40, "?", "?", 15, "?", "?", 30, "?"],
        }
    )
    registration.to_csv(data_dir / "studentRegistration.csv", index=False)

    # One assessment with a real due date, one Exam with no due date.
    assessments = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J"],
            "id_assessment": [1752, 1753, 1754, 2001, 2002],
            "assessment_type": ["TMA", "TMA", "Exam", "TMA", "Exam"],
            "date": [19, 54, "?", 21, "?"],
            "weight": [10, 20, 70, 25, 75],
        }
    )
    assessments.to_csv(data_dir / "assessments.csv", index=False)

    # Submissions: some before the week-4 cutoff (day <= 27), some after.
    submissions = [
        (1752, 1001, 18, 0, 78), (1753, 1001, 50, 0, 82),
        (1752, 1002, 25, 0, 44),                      # late, then stops
        (1752, 1003, 15, 0, 91), (1753, 1003, 48, 0, 95),
        (1752, 1004, 19, 0, 55),
        (1752, 1005, 26, 0, 61),
        (1752, 1006, 17, 0, 74),
        (2001, 1007, 20, 0, 88), (2002, 1007, 60, 0, 90),
        (2001, 1008, 24, 0, 39),
        (2001, 1009, 19, 0, 71),
        (2001, 1010, 22, 0, 58),
        (2001, 1011, 27, 0, 46),
        (2001, 1012, 18, 0, 80),
    ]
    student_assessment = pd.DataFrame(
        submissions,
        columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    )
    student_assessment.to_csv(data_dir / "studentAssessment.csv", index=False)

    vle = pd.DataFrame(
        {
            "id_site": [546943, 546712, 546713],
            "code_module": ["AAA", "AAA", "BBB"],
            "code_presentation": ["2013J", "2013J", "2013J"],
            "activity_type": ["resource", "oucontent", "forumng"],
            "week_from": ["?", "?", "?"],
            "week_to": ["?", "?", "?"],
        }
    )
    vle.to_csv(data_dir / "vle.csv", index=False)

    # Clickstream: dense for engaged learners, sparse for withdrawn ones, plus
    # pre-course activity on negative days.
    rows = []
    activity = {
        1001: [30, 35, 40, 38, 42, 45, 40, 44],
        1002: [25, 10, 0, 0, 0, 0, 0, 0],
        1003: [40, 45, 50, 52, 48, 50, 55, 53],
        1004: [20, 18, 15, 12, 10, 8, 6, 5],
        1005: [22, 8, 0, 0, 0, 0, 0, 0],
        1006: [28, 30, 32, 30, 28, 31, 29, 33],
        1007: [35, 38, 42, 44, 46, 48, 45, 47],
        1008: [18, 6, 0, 0, 0, 0, 0, 0],
        1009: [26, 28, 27, 29, 30, 28, 31, 30],
        1010: [24, 20, 16, 14, 12, 10, 9, 8],
        1011: [21, 5, 0, 0, 0, 0, 0, 0],
        1012: [27, 29, 31, 33, 32, 34, 33, 35],
    }
    module_pres = {
        1001: ("AAA", "2013J"), 1002: ("AAA", "2013J"), 1003: ("AAA", "2013J"),
        1004: ("AAA", "2013J"), 1005: ("AAA", "2014J"), 1006: ("AAA", "2014J"),
        1007: ("BBB", "2013J"), 1008: ("BBB", "2013J"), 1009: ("BBB", "2013J"),
        1010: ("BBB", "2013J"), 1011: ("BBB", "2013J"), 1012: ("BBB", "2013J"),
    }
    for student, weekly in activity.items():
        module, pres = module_pres[student]
        for week_index, clicks in enumerate(weekly):
            if clicks == 0:
                continue  # no row at all == a genuine zero
            day = week_index * 7 + 2  # inside the week
            rows.append((module, pres, student, 546943, day, clicks))
            if clicks > 10:
                rows.append((module, pres, student, 546712, day + 1, clicks // 2))
        # Pre-course activity
        rows.append((module, pres, student, 546943, -5, 4))

    student_vle = pd.DataFrame(
        rows,
        columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
    )
    student_vle.to_csv(data_dir / "studentVle.csv", index=False)

    return data_dir


@pytest.fixture()
def fixture_data_dir(tmp_path_factory) -> Path:
    """A structurally faithful OULAD fixture in an isolated temp directory."""
    return _make_fixture_dataset(tmp_path_factory.mktemp("oulad_fixture"))


@pytest.fixture()
def fixture_config(tmp_path, monkeypatch) -> Dict[str, Path]:
    """Point config at temp artifact/processed dirs so tests never clobber
    the real generated artifacts."""
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "INTERVENTION_DB", processed / "interventions.sqlite")
    return {"artifacts": artifacts, "processed": processed}


@pytest.fixture()
def built_fixture(fixture_data_dir, fixture_config):
    """Run the SQL layer over the fixture and return a live connection."""
    from src.db import duck
    from src.features import build as feature_build

    con = feature_build.build_connection(fixture_data_dir)
    yield con
    con.close()
