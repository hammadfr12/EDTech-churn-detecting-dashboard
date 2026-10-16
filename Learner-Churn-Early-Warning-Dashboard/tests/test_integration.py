"""End-to-end integration over the fixture dataset.

Proves the documented run path works: fixture CSVs -> SQL feature build ->
model training -> artifacts -> dashboard loaders. If this passes, a fresh
clone with real data in ``data/raw`` will run the same way.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.features import build as feature_build


@pytest.fixture()
def pipeline_artifacts(fixture_data_dir, fixture_config):
    """Run the real feature build and training over the fixture."""
    from src.model import train as train_module

    feature_build.build(fixture_data_dir, [3, 4])
    metrics = train_module.train(cutoff_week=4, compare_tree=True)
    return metrics


def test_feature_build_writes_every_artifact(pipeline_artifacts, fixture_config):
    for key in (
        "learner_week",
        "assessment_events",
        "cohort_outcomes",
        "weekly_cohort",
        "dropoff",
        "learner_snapshot",
        "cohort_metrics",
    ):
        path = config.artifact_path(key)
        assert path.exists(), f"artifact '{key}' was not written to {path}"


def test_snapshot_has_one_row_per_learner_course(fixture_data_dir, pipeline_artifacts):
    info = pd.read_csv(fixture_data_dir / "studentInfo.csv")
    snap = pd.read_parquet(config.artifact_path("learner_snapshot"))
    assert len(snap) == len(info)
    assert not snap.duplicated(["code_module", "code_presentation", "id_student"]).any()


def test_cohort_metrics_json_is_valid_and_complete(pipeline_artifacts):
    data = json.loads(config.artifact_path("cohort_metrics").read_text())
    assert data["cohort"]["learner_course_records"] == 12
    assert data["default_cutoff_week"] == config.DEFAULT_CUTOFF_WEEK
    assert set(data["snapshot_shapes"]) == {"3", "4"}
    for relation, info in data["grain"].items():
        assert info["duplicate_key_rows"] == 0, f"{relation} has duplicate keys"


def test_model_metrics_json_has_every_documented_field(pipeline_artifacts):
    m = json.loads(config.artifact_path("model_metrics").read_text())
    for key in (
        "cutoff_week", "target", "dataset", "splits", "threshold",
        "validation_metrics", "test_metrics", "cross_validation",
        "feature_indicators", "warning", "limitations",
    ):
        assert key in m, f"model_metrics is missing '{key}'"

    assert m["cutoff_week"] == 4
    assert m["target"]["key"] == "primary"
    assert "Fail" in m["target"]["excluded_labels"]
    assert m["warning"]


def test_metrics_are_computed_on_held_out_data(pipeline_artifacts):
    m = json.loads(config.artifact_path("model_metrics").read_text())
    test_n = m["splits"]["test"]["n"]
    train_n = m["splits"]["train"]["n"]
    assert test_n > 0 and train_n > 0
    assert test_n + m["splits"]["validation"]["n"] + train_n == m["dataset"]["rows_in_modelling_cohort"]

    # The test confusion matrix must sum to the test set size.
    at = m["test_metrics"]["at_threshold"]
    total = at["true_positive"] + at["false_positive"] + at["true_negative"] + at["false_negative"]
    assert total == test_n


def test_risk_queue_is_sorted_and_bounded(pipeline_artifacts):
    queue = pd.read_parquet(config.artifact_path("risk_queue"))
    assert queue["risk_probability"].between(0, 1).all()
    assert queue["risk_probability"].is_monotonic_decreasing
    assert set(queue["risk_band"]) <= {"Low", "Medium", "High"}
    assert "final_result" in queue.columns  # kept for auditing, never a feature


def test_model_artifact_round_trips(pipeline_artifacts):
    import joblib

    bundle = joblib.load(config.artifact_path("model"))
    assert bundle["cutoff_week"] == 4
    assert bundle["target_key"] == "primary"
    assert 0 <= bundle["threshold"] <= 1
    assert bundle["feature_columns"]

    # The saved pipeline must actually predict.
    queue = pd.read_parquet(config.artifact_path("risk_queue"))
    assert len(queue) > 0


def test_dashboard_loaders_read_the_generated_artifacts(pipeline_artifacts):
    """The loaders the dashboard uses must find and parse everything."""
    from src.analytics import loader

    for fn in (
        loader.cohort_metrics,
        loader.model_metrics,
        loader.learner_snapshot,
        loader.risk_queue,
        loader.cohort_outcomes,
        loader.weekly_cohort,
        loader.activity_dropoff,
        loader.learner_week,
        loader.assessment_events,
        loader.model_bundle,
    ):
        result = fn()
        assert result.ok, f"{fn.__name__} failed to load: {result.reason}"


def test_loaders_report_a_reason_when_artifacts_are_absent(tmp_path, monkeypatch):
    """A missing artifact must produce a reason, never a silent empty value."""
    from src.analytics import loader

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    loader.cohort_metrics.clear()
    result = loader.cohort_metrics()
    assert not result.ok
    assert result.reason and "cohort_metrics.json" in result.reason


def test_weekly_activity_dropoff_uses_a_real_denominator(pipeline_artifacts):
    dropoff = pd.read_parquet(config.artifact_path("dropoff"))
    assert len(dropoff) > 0
    # Rows where the previous week had zero active learners must be excluded,
    # never reported as a 100% drop.
    assert dropoff["prev_active_learners"].gt(0).all()
    assert dropoff["dropoff_rate"].between(-1, 1).all()


def test_pipeline_is_rerunnable(fixture_data_dir, pipeline_artifacts):
    """Running the build twice must produce the same row counts."""
    before = pd.read_parquet(config.artifact_path("learner_snapshot")).shape
    feature_build.build(fixture_data_dir, [3, 4])
    after = pd.read_parquet(config.artifact_path("learner_snapshot")).shape
    assert before == after
