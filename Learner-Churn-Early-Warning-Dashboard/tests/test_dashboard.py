"""Dashboard render tests using Streamlit's own AppTest harness.

These execute each page for real — running the page body, the filters and the
artifact loaders — and assert that no exception is raised and that expected
content appears. They run headlessly, so they work in CI and on a fresh clone.

The pages are run against the artifacts produced by ``test_integration``'s
fixture pipeline, so the tests cover the real render path rather than mocks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.features import build as feature_build
from src.model import train as train_module

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    ROOT / "app" / "Home.py",
    ROOT / "app" / "pages" / "1_Engagement.py",
    ROOT / "app" / "pages" / "2_Risk_Queue.py",
    ROOT / "app" / "pages" / "3_Learner_Detail.py",
    ROOT / "app" / "pages" / "4_Interventions.py",
    ROOT / "app" / "pages" / "5_Model_and_Data_Quality.py",
]


@pytest.fixture(scope="module")
def rendered_app(request):
    """Build artifacts from the fixture once, then render every page."""
    # Reuse the session-scoped fixture data by building in a module temp dir.
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="lcwd_dash_"))

    # Recreate the fixture dataset in this temp location.
    from tests.conftest import _make_fixture_dataset

    data_dir = _make_fixture_dataset(tmp / "raw")

    artifacts = tmp / "artifacts"
    processed = tmp / "processed"
    artifacts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # Snapshot config so this module's temporary redirect cannot leak into
    # other test modules running in the same session.
    original = {
        name: getattr(config, name)
        for name in ("ARTIFACTS_DIR", "PROCESSED_DIR", "DATA_DIR", "INTERVENTION_DB")
    }
    config.ARTIFACTS_DIR = artifacts
    config.PROCESSED_DIR = processed
    config.DATA_DIR = tmp
    config.INTERVENTION_DB = processed / "interventions.sqlite"

    def _clear_caches():
        from src.analytics import loader

        for fn in (
            loader.data_audit, loader.cohort_metrics, loader.model_metrics,
            loader.learner_snapshot, loader.risk_queue, loader.cohort_outcomes,
            loader.weekly_cohort, loader.activity_dropoff, loader.learner_week,
            loader.assessment_events, loader.model_bundle,
        ):
            fn.clear()

    try:
        feature_build.build(data_dir, [3, 4])
        train_module.train(cutoff_week=4, compare_tree=True)
        _clear_caches()
        yield tmp
    finally:
        for name, value in original.items():
            setattr(config, name, value)
        _clear_caches()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(page, rendered_app):
    """Every page must render end to end against real artifacts."""
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()

    assert not at.exception, (
        f"{page} raised during render:\n"
        + "\n".join(str(e.value) for e in at.exception)
    )


@pytest.mark.parametrize("page", PAGES)
def test_page_produces_content(page, rendered_app):
    """A rendered page must actually produce markdown, not a blank screen."""
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    body = " ".join(m.value for m in at.markdown)
    assert len(body) > 200, f"{page} rendered almost no content"


def test_overview_shows_real_counts(rendered_app):
    """The overview must show the fixture's real record count, not a placeholder."""
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "12" in body, "overview should show the 12 fixture learner-course records"
    assert "Load OULAD data" not in body


def test_risk_queue_renders_a_table(rendered_app):
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "2_Risk_Queue.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1, "risk queue should render at least one dataframe"


def test_model_page_reports_metrics(rendered_app):
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "5_Model_and_Data_Quality.py"), default_timeout=120)
    at.run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    # The definitions must be present before any metric.
    assert "Target definition" in body or "target" in body.lower()
    assert "leakage" in body.lower()


def test_pages_degrade_gracefully_without_artifacts(tmp_path, monkeypatch):
    """With no artifacts, pages must show an empty state, not crash.

    The loader cache is cleared on the way in *and* on the way out: leaving an
    empty snapshot cached would make every later test in the session render the
    no-data state, which is a test-isolation failure rather than a real one.
    """
    from src.analytics import loader

    cached = (
        loader.data_audit, loader.cohort_metrics, loader.model_metrics,
        loader.learner_snapshot, loader.risk_queue, loader.cohort_outcomes,
        loader.weekly_cohort, loader.activity_dropoff, loader.learner_week,
        loader.assessment_events, loader.model_bundle,
    )

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "empty")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "empty")
    (tmp_path / "empty").mkdir()

    for fn in cached:
        fn.clear()

    try:
        at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120)
        at.run()

        assert not at.exception, (
            "the overview must show a no-data state rather than raising:\n"
            + "\n".join(str(e.value) for e in at.exception)
        )
        body = " ".join(m.value for m in at.markdown)
        assert "No feature snapshot" in body or "not been run" in body
    finally:
        for fn in cached:
            fn.clear()


def test_no_page_renders_a_raw_nan(rendered_app):
    """Optional numeric fields must render as an em-dash, never as 'nan'.

    A learner with no recorded activity has a null last-active week. Printing
    the raw value reads as a bug, so this asserts it never reaches the screen.
    """
    for page in PAGES:
        at = AppTest.from_file(str(page), default_timeout=120)
        at.run()
        assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"

        rendered = " ".join(m.value for m in at.markdown)
        rendered += " ".join(str(c.value) for c in at.caption)
        # 'nan' as a standalone token, not a substring of a real word.
        assert "`nan`" not in rendered, f"{page} rendered a raw `nan`"
        assert "&nbsp; `nan" not in rendered, f"{page} rendered a raw nan value"


def test_sidebar_filters_change_the_rendered_cohort(rendered_app):
    """Selecting one course must change the counts, not just the label."""
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception

    baseline = " ".join(m.value for m in at.markdown)
    assert "All cohorts" in baseline

    # The course selector is the first selectbox in the sidebar.
    course_select = at.sidebar.selectbox[0]
    options = list(course_select.options)
    assert len(options) > 1, "fixture should expose more than one course"

    course_select.select(options[1]).run()
    assert not at.exception
    filtered = " ".join(m.value for m in at.markdown)

    assert filtered != baseline, "changing the course filter did not change the page"
    assert options[1] in filtered, "the selected course should appear in the scope label"


def _arrow_serialisable(df: pd.DataFrame) -> tuple[bool, str]:
    """Streamlit renders dataframes through Arrow, so every frame a page emits
    must serialise. A mixed str/number column raises at render time."""
    import pyarrow as pa

    try:
        pa.Table.from_pandas(df, preserve_index=False)
        return True, ""
    except Exception as exc:  # pragma: no cover - surfaced in the assert
        return False, str(exc)


@pytest.mark.parametrize("page", PAGES)
def test_every_rendered_dataframe_serialises_to_arrow(page, rendered_app):
    """Catch mixed-type columns before they reach the user.

    A dataframe column holding numbers *and* a placeholder string (an em-dash,
    say) renders in some contexts but raises ArrowInvalid in Streamlit's Arrow
    path. Asserting serialisation here makes that a test failure rather than a
    runtime error on a page nobody opened.
    """
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"

    for i, element in enumerate(at.dataframe):
        df = element.value
        if df is None or len(df.columns) == 0:
            continue
        ok, message = _arrow_serialisable(df)
        assert ok, (
            f"{page}: dataframe #{i} failed Arrow serialisation, which Streamlit "
            f"raises at render time.\nColumns: {list(df.columns)}\nDtypes: "
            f"{dict(df.dtypes.astype(str))}\nError: {message}"
        )
