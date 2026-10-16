"""Learner detail: one learner's observed timeline against the cutoff.

Shows what was actually observed up to the cutoff week, what the model made
of it, and — kept visually and explicitly separate — what happened afterwards.
The separation is the whole point of the page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

# Streamlit puts only the running script's directory on sys.path, so the
# project root (which holds `src` and `app`) has to be added explicitly.
# Walk up until we find the directory that contains `src/config.py`.
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "src" / "config.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from app._shared import (
    COLORS,
    OUTCOME_COLORS,
    RISK_COLORS,
    empty_state,
    fmt_int,
    fmt_num,
    fmt_pct,
    global_filters,
    kpi,
    kpi_row,
    load_snapshot_or_stop,
    note,
    risk_band_series,
    section,
    setup_page,
    sidebar_status,
    style_fig,
)
from src import config
from src.analytics import loader

setup_page(
    "Learner detail",
    "One learner's engagement and assessment timeline, their modelled score, "
    "and the cutoff that separates observation from outcome.",
    tags=["Cutoff-safe", "No causal claims"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)
bands = filters["risk_bands"]
cutoff = filters["cutoff_week"]

weeks_result = loader.learner_week()
events_result = loader.assessment_events()
queue_result = loader.risk_queue()
metrics_result = loader.model_metrics()

if not weeks_result.ok or not events_result.ok:
    empty_state(
        "Learner timelines are missing",
        "Run the feature build to generate engagement and assessment events.",
        "python -m src.features.build --data-dir data/raw",
    )
    st.stop()

snap = snapshot
queue = queue_result.value if queue_result.ok else None
threshold = (
    float(metrics_result.value.get("threshold", {}).get("threshold", config.DEFAULT_THRESHOLD))
    if metrics_result.ok
    else config.DEFAULT_THRESHOLD
)

# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
section("Select a learner")

c1, c2, c3 = st.columns([1.2, 1, 1.4])
with c1:
    courses = ["All courses"] + sorted(snap["code_module"].unique().tolist())
    course = st.selectbox("Course", courses, index=0)
with c2:
    if course == "All courses":
        pool = snap
    else:
        pool = snap[snap["code_module"] == course]
    presentations = ["All presentations"] + sorted(
        pool["code_presentation"].unique().tolist()
    )
    presentation = st.selectbox("Presentation", presentations, index=0)
with c3:
    pool = pool if presentation == "All presentations" else pool[
        pool["code_presentation"] == presentation
    ]
    ids = sorted(pool["id_student"].unique().tolist())
    if not ids:
        empty_state("No learners", "No learners match this course and presentation.")
        st.stop()
    learner_id = st.selectbox(
        "Learner ID (pseudonymous OULAD identifier)", ids, index=0
    )

row = pool[pool["id_student"] == learner_id].iloc[0]
module = row["code_module"]
pres = row["code_presentation"]

# ---------------------------------------------------------------------------
# Score + headline
# ---------------------------------------------------------------------------
score = None
band = "—"
if queue is not None:
    match = queue[
        (queue["id_student"] == learner_id)
        & (queue["code_module"] == module)
        & (queue["code_presentation"] == pres)
    ]
    if len(match):
        score = float(match["risk_probability"].iloc[0])
        band = risk_band_series(pd.Series([score]), bands).iloc[0]

observed_outcome = str(row["final_result"])

section(f"Learner {learner_id} · {module} {pres}")
kpi_row(
    [
        kpi(
            "Modelled risk",
            f"{score:.3f}" if score is not None else "not scored",
            f"band: {band} · threshold {threshold:.2f}",
            accent=RISK_COLORS.get(band, COLORS["primary"]),
            bar_fraction=score,
        ),
        kpi(
            f"Clicks by week {cutoff}",
            fmt_int(row.get("total_clicks_to_cutoff", 0)),
            f"{fmt_num(row.get('clicks_per_active_day'), 1)} per active day",
            accent=COLORS["accent"],
        ),
        kpi(
            "Days since activity",
            fmt_int(row.get("days_since_last_activity", 0)),
            f"last active week {fmt_int(row.get('last_active_week_in_window', 0))}",
        ),
        kpi(
            "Outcome (historical)",
            observed_outcome,
            "recorded after the course — never a feature",
            accent=OUTCOME_COLORS.get(observed_outcome, COLORS["muted"]),
        ),
    ]
)

if score is not None:
    note(
        f"The model saw this learner's behaviour up to the end of "
        f"<strong>week {cutoff}</strong> and assigned a risk score of "
        f"<strong>{score:.3f}</strong>."
    )
else:
    note(
        f"The model saw this learner's behaviour up to the end of "
        f"<strong>week {cutoff}</strong>. This learner is not in the scored queue "
        "(they may have been excluded by the target definition)."
    )
note(
    "The outcome shown above is the <em>historical</em> OULAD record for this learner. "
    "It was never available to the model, and its presence here is what makes the "
    "early-warning claim testable."
)

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
section("Observed timeline")
lw = weeks_result.value
learner_weeks = lw[
    (lw["id_student"] == learner_id)
    & (lw["code_module"] == module)
    & (lw["code_presentation"] == pres)
].sort_values("week")

fig = go.Figure()
if len(learner_weeks):
    fig.add_trace(
        go.Bar(
            x=learner_weeks["week"], y=learner_weeks["clicks"], name="Clicks",
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            hovertemplate="Week %{x}<br>%{y:,} clicks<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=learner_weeks["week"], y=learner_weeks["active_days"], name="Active days",
            mode="lines+markers", line=dict(color=COLORS["accent"], width=2.4),
            marker=dict(size=6), yaxis="y2",
            hovertemplate="Week %{x}<br>%{y} active days<extra></extra>",
        )
    )
fig.add_vrect(
    x0=0.5, x1=cutoff + 0.5,
    fillcolor="rgba(47,69,197,0.07)", line_width=0,
    annotation_text="observed by the model", annotation_position="top left",
    annotation_font=dict(size=10, color=COLORS["primary"]),
)
fig.add_vline(
    x=cutoff + 0.5, line=dict(color=COLORS["faint"], width=1.6, dash="dash"),
)
fig.update_layout(
    yaxis=dict(title="Clicks"),
    yaxis2=dict(title="Active days", overlaying="y", side="right", showgrid=False),
    bargap=0.3,
)
fig.update_xaxes(title="Course-relative week", dtick=2)
st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------
section("Assessment progress")
ev = events_result.value
learner_events = ev[
    (ev["id_student"] == learner_id)
    & (ev["code_module"] == module)
    & (ev["code_presentation"] == pres)
].copy()
learner_events = learner_events.sort_values("submitted_day")

if len(learner_events):
    learner_events["before_cutoff"] = learner_events["submitted_day"] <= cutoff * 7 - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=learner_events["submitted_day"],
            y=learner_events["score"],
            mode="markers",
            name="Submissions",
            marker=dict(
                size=11,
                color=[
                    COLORS["primary"] if b else "rgba(139,147,167,0.45)"
                    for b in learner_events["before_cutoff"]
                ],
                line=dict(width=1, color="#FFFFFF"),
                symbol=[
                    "circle" if b else "circle-open" for b in learner_events["before_cutoff"]
                ],
            ),
            customdata=np.stack(
                [
                    learner_events["assessment_type"].astype(str),
                    learner_events["lateness_days"].fillna(-999),
                    learner_events["before_cutoff"].map({True: "yes", False: "no"}),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "day %{x} · %{customdata[0]}<br>score %{y}<br>"
                "lateness %{customdata[1]} d<br>before cutoff: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=cutoff * 7 - 1, line=dict(color=COLORS["faint"], width=1.6, dash="dash"),
        annotation_text=f"cutoff · day {cutoff * 7 - 1}", annotation_position="top",
        annotation_font=dict(size=10, color=COLORS["faint"]),
    )
    fig.update_xaxes(title="Day offset from presentation start")
    fig.update_yaxes(title="Score", range=[0, 100])
    st.plotly_chart(style_fig(fig, height=320, legend=False), use_container_width=True)

    before = learner_events[learner_events["before_cutoff"]]
    after = learner_events[~learner_events["before_cutoff"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Submissions before cutoff", f"{len(before):,}")
    c2.metric("Submissions after cutoff", f"{len(after):,}")
    c3.metric(
        "Mean score before cutoff",
        f"{before['score'].mean():.1f}" if len(before) else "—",
    )
    c4.metric(
        "Late submissions before cutoff",
        f"{int((before['lateness_days'] > 0).sum()):,}"
        if len(before) and before["lateness_days"].notna().any()
        else "not available",
    )
    st.caption(
        "Hollow grey markers are submissions recorded after the cutoff. They are shown "
        "for context only and are excluded from every feature. Lateness is only defined "
        "for assessments that carry a real due date in OULAD; it is never imputed."
    )
else:
    st.caption("No assessment submissions recorded for this learner.")

# ---------------------------------------------------------------------------
# Feature contributions
# ---------------------------------------------------------------------------
section("What contributed to this score")
contrib_cols = [
    ("overdue_count", "Assessments due but not submitted"),
    ("submission_rate", "Submission rate against what was due"),
    ("mean_score", "Mean score so far"),
    ("click_slope", "Click trend across the window"),
    ("days_since_last_activity", "Days since last activity"),
    ("total_clicks_to_cutoff", "Total clicks by cutoff"),
    ("total_active_days_to_cutoff", "Active days by cutoff"),
    ("active_days_delta", "Change in active days, last week vs previous"),
]
rows = []
for col, label in contrib_cols:
    if col in snap.columns:
        value = row.get(col)
        median = snap[col].median()
        rows.append(
            {
                "Indicator": label,
                # Both columns stay numeric so the table serialises; a missing
                # value is genuinely absent rather than a placeholder string.
                "This learner": None if pd.isna(value) else round(float(value), 3),
                "Cohort median": None if pd.isna(median) else round(float(median), 3),
            }
        )
contrib = pd.DataFrame(rows)
if len(contrib):
    contrib["This learner"] = pd.to_numeric(contrib["This learner"], errors="coerce")
    contrib["Cohort median"] = pd.to_numeric(contrib["Cohort median"], errors="coerce")
    st.dataframe(
        contrib,
        hide_index=True,
        use_container_width=True,
        column_config={
            "This learner": st.column_config.NumberColumn("This learner", format="%.3f"),
            "Cohort median": st.column_config.NumberColumn("Cohort median", format="%.3f"),
        },
    )
    st.caption(
        "These are the features the model uses, shown alongside the cohort median so the "
        "learner's position is readable. A blank cell means the value was never observed "
        "for this learner — it is not zero. These describe association in historical data, "
        "<strong>not</strong> reasons. Do not present them to a learner as an explanation."
    )

st.divider()
st.caption(
    "Human review is required before any intervention. This page describes observed "
    "behaviour; it does not establish intent, ability or cause."
)
