"""Intervention store: create, update, validate, and reject bad input."""

from __future__ import annotations

import pytest

from src import config
from src.interventions import store


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "interventions.sqlite"


def test_upsert_creates_a_record(db):
    record_id = store.upsert(
        "AAA", "2013J", 1001, 4, "primary",
        risk_probability=0.81, risk_band="High",
        owner="mentor-a", action="Supportive check-in", status="Assigned",
        db_path=db,
    )
    assert record_id > 0
    records = store.fetch(db_path=db)
    assert len(records) == 1
    assert records.iloc[0]["owner"] == "mentor-a"
    assert records.iloc[0]["status"] == "Assigned"


def test_upsert_is_idempotent_on_the_natural_key(db):
    for score in (0.5, 0.9):
        store.upsert(
            "AAA", "2013J", 1001, 4, "primary",
            risk_probability=score, owner="mentor-a", db_path=db,
        )
    records = store.fetch(db_path=db)
    assert len(records) == 1, "the same learner/cutoff/target must not create a second row"
    assert records.iloc[0]["risk_probability"] == pytest.approx(0.9)


def test_different_targets_create_separate_records(db):
    store.upsert("AAA", "2013J", 1001, 4, "primary", db_path=db)
    store.upsert("AAA", "2013J", 1001, 4, "fail_as_churn", db_path=db)
    assert len(store.fetch(db_path=db)) == 2


def test_invalid_status_is_rejected(db):
    with pytest.raises(ValueError, match="Invalid status"):
        store.upsert("AAA", "2013J", 1001, 4, "primary", status="Escalated", db_path=db)


def test_invalid_action_is_rejected(db):
    with pytest.raises(ValueError, match="Invalid action"):
        store.upsert("AAA", "2013J", 1001, 4, "primary", action="Send warning email", db_path=db)


def test_update_changes_fields_and_stamps_timestamp(db):
    record_id = store.upsert("AAA", "2013J", 1001, 4, "primary", owner="mentor-a", db_path=db)
    before = store.fetch(db_path=db).iloc[0]["updated_at"]
    store.update(record_id, status="Resolved", notes="Reviewed with tutor.", db_path=db)
    after = store.fetch(db_path=db).iloc[0]
    assert after["status"] == "Resolved"
    assert after["notes"] == "Reviewed with tutor."
    assert after["updated_at"] >= before


def test_update_rejects_unknown_fields(db):
    record_id = store.upsert("AAA", "2013J", 1001, 4, "primary", db_path=db)
    with pytest.raises(ValueError, match="unknown field"):
        store.update(record_id, email="someone@example.com", db_path=db)


def test_update_rejects_invalid_status(db):
    record_id = store.upsert("AAA", "2013J", 1001, 4, "primary", db_path=db)
    with pytest.raises(ValueError):
        store.update(record_id, status="Not a real status", db_path=db)


def test_all_documented_statuses_are_accepted(db):
    for i, status in enumerate(config.INTERVENTION_STATUSES):
        store.upsert("AAA", "2013J", 2000 + i, 4, "primary", status=status, db_path=db)
    assert len(store.fetch(db_path=db)) == len(config.INTERVENTION_STATUSES)


def test_all_documented_actions_are_accepted(db):
    for i, action in enumerate(config.INTERVENTION_ACTIONS):
        store.upsert("AAA", "2013J", 3000 + i, 4, "primary", action=action, db_path=db)
    assert len(store.fetch(db_path=db)) == len(config.INTERVENTION_ACTIONS)


def test_fetch_filters_by_cohort_and_status(db):
    store.upsert("AAA", "2013J", 1, 4, "primary", status="New", db_path=db)
    store.upsert("BBB", "2013J", 2, 4, "primary", status="Resolved", db_path=db)
    store.upsert("BBB", "2013J", 3, 4, "primary", status="New", db_path=db)

    assert len(store.fetch(db_path=db, code_module="BBB")) == 2
    assert len(store.fetch(db_path=db, status="New")) == 2
    assert len(store.fetch(db_path=db, code_module="BBB", status="New")) == 1


def test_delete_removes_the_record(db):
    record_id = store.upsert("AAA", "2013J", 1001, 4, "primary", db_path=db)
    store.delete(record_id, db_path=db)
    assert store.fetch(db_path=db).empty


def test_status_summary_counts_per_status(db):
    store.upsert("AAA", "2013J", 1, 4, "primary", status="New", db_path=db)
    store.upsert("AAA", "2013J", 2, 4, "primary", status="New", db_path=db)
    store.upsert("AAA", "2013J", 3, 4, "primary", status="Resolved", db_path=db)
    summary = store.status_summary(db_path=db)
    assert summary["New"] == 2
    assert summary["Resolved"] == 1


def test_no_contact_fields_exist_in_the_schema(db):
    """The schema must not be able to store contact details."""
    store.upsert("AAA", "2013J", 1001, 4, "primary", db_path=db)
    columns = set(store.fetch(db_path=db).columns)
    forbidden = {"email", "phone", "mobile", "address", "name", "full_name"}
    assert not (columns & forbidden), f"schema exposes personal-data fields: {columns & forbidden}"
