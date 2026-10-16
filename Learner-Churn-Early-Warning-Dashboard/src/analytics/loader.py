"""Artifact loading for the dashboard.

The dashboard must never invent a number. Every loader returns ``None`` (with
a reason) when an artifact is absent, so pages can render an honest empty
state such as "Load OULAD data to calculate" instead of a placeholder metric.

Results are cached so a page re-render does not re-read 10M-row artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src import config


@dataclass(frozen=True)
class LoadResult:
    """Either a value or a stated reason why there isn't one."""

    value: Any
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def _read_parquet(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult(None, f"`{path.name}` has not been generated yet.")
    try:
        return LoadResult(pd.read_parquet(path))
    except Exception as exc:  # pragma: no cover - corrupt artifact
        return LoadResult(None, f"`{path.name}` could not be read: {exc}")


def _read_json(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult(None, f"`{path.name}` has not been generated yet.")
    try:
        return LoadResult(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # pragma: no cover
        return LoadResult(None, f"`{path.name}` could not be parsed: {exc}")


@st.cache_data(show_spinner=False)
def data_audit() -> LoadResult:
    return _read_json(config.artifact_path("data_audit"))


@st.cache_data(show_spinner=False)
def cohort_metrics() -> LoadResult:
    return _read_json(config.artifact_path("cohort_metrics"))


@st.cache_data(show_spinner=False)
def model_metrics() -> LoadResult:
    return _read_json(config.artifact_path("model_metrics"))


@st.cache_data(show_spinner=False)
def learner_snapshot() -> LoadResult:
    return _read_parquet(config.artifact_path("learner_snapshot"))


@st.cache_data(show_spinner=False)
def risk_queue() -> LoadResult:
    return _read_parquet(config.artifact_path("risk_queue"))


@st.cache_data(show_spinner=False)
def cohort_outcomes() -> LoadResult:
    return _read_parquet(config.artifact_path("cohort_outcomes"))


@st.cache_data(show_spinner=False)
def weekly_cohort() -> LoadResult:
    return _read_parquet(config.artifact_path("weekly_cohort"))


@st.cache_data(show_spinner=False)
def activity_dropoff() -> LoadResult:
    return _read_parquet(config.artifact_path("dropoff"))


@st.cache_data(show_spinner=False)
def learner_week() -> LoadResult:
    return _read_parquet(config.artifact_path("learner_week"))


@st.cache_data(show_spinner=False)
def assessment_events() -> LoadResult:
    return _read_parquet(config.artifact_path("assessment_events"))


@st.cache_data(show_spinner=False)
def model_bundle() -> LoadResult:
    """The fitted pipeline and its metadata, loaded lazily."""
    path = config.artifact_path("model")
    if not path.exists():
        return LoadResult(None, "`churn_model.joblib` has not been generated yet.")
    try:
        import joblib

        return LoadResult(joblib.load(path))
    except Exception as exc:  # pragma: no cover
        return LoadResult(None, f"Model could not be loaded: {exc}")


def pipeline_status() -> Dict[str, Any]:
    """What exists on disk, for the sidebar status block."""
    checks = {
        "raw dataset": config.RAW_DIR / "studentInfo.csv",
        "data audit": config.artifact_path("data_audit"),
        "cohort metrics": config.artifact_path("cohort_metrics"),
        "feature snapshot": config.artifact_path("learner_snapshot"),
        "trained model": config.artifact_path("model"),
        "model metrics": config.artifact_path("model_metrics"),
        "risk queue": config.artifact_path("risk_queue"),
    }
    return {name: path.exists() for name, path in checks.items()}


def available_cohorts(snapshot: pd.DataFrame) -> Tuple[list, list]:
    """Course and presentation options present in the data."""
    courses = sorted(snapshot["code_module"].dropna().unique().tolist())
    presentations = sorted(snapshot["code_presentation"].dropna().unique().tolist())
    return courses, presentations
