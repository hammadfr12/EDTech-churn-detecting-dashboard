"""Overview: the cohort at a glance.

Headline counts, the outcome mix, weekly active learners and the cohort
funnel. Every figure traces back to a generated artifact; nothing here is a
placeholder.
"""

from __future__ import annotations

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

from app._shared import (  # noqa: E402
    COLORS,
    OUTCOME_COLORS,
    RISK_COLORS,
    apply_cohort_filter,
    empty_state,
    fmt_int,
    fmt_pct,
    global_filters,
    kpi,
    kpi_row,
    load_snapshot_or_stop,
    note,
    section,
    setup_page,
    sidebar_status,
    style_fig,
)
from src import config
from src.analytics import loader

setup_page(
    "Learner Churn Early-Warning",
    "Cohort outcomes, engagement drop-off and an explainable at-risk queue "
    "built on the Open University Learning Analytics Dataset.",
    tags=["OULAD", f"Cutoff week {config.DEFAULT_CUTOFF_WEEK}", "Research prototype"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)

outcomes = loader.cohort_outcomes()
weekly = loader.weekly_cohort()
audit = loader.data_audit()

if not outcomes.ok:
    empty_state(
        "Cohort artifacts are missing",
        "Run the feature build to generate cohort metrics.",
        "python -m src.features.build --data-dir data/raw",
    )
    st.stop()

cohort = apply_cohort_filter(outcomes.value, filters["course"], filters["presentation"])
scope = (
    f"{filters['course']} · {filters['presentation']}"
    if filters["course"] != "All courses" or filters["presentation"] != "All presentations"
    else "All cohorts"
)

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
enrolled = len(cohort)
n_withdrawn = int((cohort["final_result"] == "Withdrawn").sum())
n_fail = int((cohort["final_result"] == "Fail").sum())
n_pass = int((cohort["final_result"] == "Pass").sum())
n_dist = int((cohort["final_result"] == "Distinction").sum())
completed = n_pass + n_dist

section(f"Overview · {scope}")
kpi_row(
    [
        kpi("Learners enrolled", fmt_int(enrolled), "distinct learner-course presentations"),
        kpi(
            "Withdrawal rate",
            fmt_pct(n_withdrawn / enrolled if enrolled else None),
            f"{fmt_int(n_withdrawn)} withdrawn",
            accent=OUTCOME_COLORS["Withdrawn"],
            bar_fraction=(n_withdrawn / enrolled) if enrolled else None,
        ),
        kpi(
            "Completion rate",
            fmt_pct(completed / enrolled if enrolled else None),
            f"{fmt_int(n_pass)} pass · {fmt_int(n_dist)} distinction",
            accent=OUTCOME_COLORS["Pass"],
            bar_fraction=(completed / enrolled) if enrolled else None,
        ),
        kpi(
            "Failure rate",
            fmt_pct(n_fail / enrolled if enrolled else None),
            "reported separately from withdrawal",
            accent=OUTCOME_COLORS["Fail"],
            bar_fraction=(n_fail / enrolled) if enrolled else None,
        ),
    ]
)

note(
    "Withdrawal and failure are different outcomes and are never combined here. "
    "The model's primary target treats <strong>Withdrawn</strong> as the positive class "
    "and <strong>excludes Fail</strong>; a sensitivity target that counts Fail as "
    "non-completion is available in the sidebar so the effect of that choice is visible."
)

left, right = st.columns([1, 1.25], gap="medium")

# ---------------------------------------------------------------------------
# Outcome mix
# ---------------------------------------------------------------------------
with left:
    section("Outcome mix")
    mix = (
        cohort["final_result"].value_counts().reindex(config.OUTCOME_LABELS).dropna().astype(int)
    )
    order = ["Distinction", "Pass", "Fail", "Withdrawn"]
    mix = mix.reindex([o for o in order if o in mix.index])
    fig = go.Figure(
        go.Bar(
            x=mix.values,
            y=mix.index,
            orientation="h",
            marker=dict(color=[OUTCOME_COLORS[i] for i in mix.index], line=dict(width=0)),
            text=[f"{v:,}  ({v / enrolled * 100:.1f}%)" for v in mix.values],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["muted"]),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[0, mix.max() * 1.35 if len(mix) else 1])
    st.plotly_chart(style_fig(fig, height=260, legend=False), use_container_width=True)

# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------
with right:
    section("Cohort funnel")
    ever_active = int(cohort["last_active_week"].notna().sum()) if "last_active_week" in cohort else 0
    active_by_w4 = 0
    if weekly.ok:
        w = apply_cohort_filter(weekly.value, filters["course"], filters["presentation"])
        row = w[(w["week"] == filters["cutoff_week"])]
        active_by_w4 = int(row["active_learners"].sum()) if len(row) else 0

    stages = [
        ("Enrolled", enrolled, COLORS["primary"]),
        ("Ever active in VLE", ever_active, COLORS["accent"]),
        (f"Active in week {filters['cutoff_week']}", active_by_w4, "#7A5AF8"),
        ("Completed (Pass + Distinction)", completed, OUTCOME_COLORS["Pass"]),
    ]
    fig = go.Figure(
        go.Bar(
            x=[s[1] for s in stages],
            y=[s[0] for s in stages],
            orientation="h",
            marker=dict(color=[s[2] for s in stages], line=dict(width=0)),
            text=[
                f"{s[1]:,}  ({s[1] / enrolled * 100:.0f}%)" if enrolled else "—"
                for s in stages
            ],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["muted"]),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[0, max([s[1] for s in stages] + [1]) * 1.35])
    st.plotly_chart(style_fig(fig, height=260, legend=False), use_container_width=True)

# ---------------------------------------------------------------------------
# Weekly activity
# ---------------------------------------------------------------------------
if weekly.ok:
    section("Weekly active learners")
    w = apply_cohort_filter(weekly.value, filters["course"], filters["presentation"])
    agg = (
        w[w["week"] <= 25]
        .groupby("week", as_index=False)
        .agg(active_learners=("active_learners", "sum"), enrolled=("learners_enrolled", "sum"))
    )
    agg["rate"] = agg["active_learners"] / agg["enrolled"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["week"],
            y=agg["active_learners"],
            mode="lines+markers",
            name="Active learners",
            line=dict(color=COLORS["primary"], width=2.4),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(47,69,197,0.08)",
            hovertemplate="Week %{x}<br>%{y:,} active learners<extra></extra>",
        )
    )
    fig.add_vline(
        x=filters["cutoff_week"] + 0.5,
        line=dict(color=COLORS["faint"], width=1.4, dash="dash"),
        annotation_text=f"cutoff · week {filters['cutoff_week']}",
        annotation_position="top",
        annotation_font=dict(size=10, color=COLORS["faint"]),
    )
    fig.update_xaxes(title="Course-relative week", dtick=2)
    fig.update_yaxes(title="Active learners")
    st.plotly_chart(style_fig(fig, height=330, legend=False), use_container_width=True)

    if audit.ok:
        st.caption(
            "Active = at least one recorded VLE interaction in that week. "
            "Pre-course activity (negative day offsets) is excluded from this series "
            "and reported separately on the Engagement page."
        )

# ---------------------------------------------------------------------------
# Steepest decline + provenance
# ---------------------------------------------------------------------------
metrics = loader.cohort_metrics()
col_a, col_b = st.columns([1.15, 1], gap="medium")

with col_a:
    section("Steepest measured activity decline")
    if metrics.ok:
        rows = metrics.value["cohort"]["steepest_dropoff_weeks"]
        table = pd.DataFrame(rows)[
            ["week", "prev_active", "active_learners", "net_change", "dropoff_rate"]
        ].copy()
        table.columns = ["Week", "Active (prev)", "Active (now)", "Net change", "Drop-off"]
        table["Drop-off"] = table["Drop-off"].map(lambda v: f"{v * 100:.1f}%")
        table["Week"] = table["Week"].astype(int)
        st.dataframe(table, hide_index=True, use_container_width=True)
        note(
            "These are <strong>activity</strong> declines across all cohorts in the dataset, "
            "measured as (active last week − active this week) / active last week. "
            "A fall in clicks is not evidence of withdrawal; the two are reported "
            "separately throughout this dashboard.",
            kind="warn",
        )
    else:
        st.caption(metrics.reason)

with col_b:
    section("Data provenance")
    if audit.ok:
        c = audit.value["cohort"]
        cal = audit.value["assessment_calendar"]
        st.markdown(
            f"""
- **Source**: Open University Learning Analytics Dataset (OULAD), public research data
- **Learner-course records**: {c['learner_course_records']:,}
- **Distinct learners**: {c['distinct_learners']:,} (pseudonymous IDs)
- **Course presentations**: {c['course_presentations']} across {c['distinct_modules']} modules
- **Assessments with a real due date**: {cal['assessments_with_due_date']}/{cal['total_assessments']} ({cal['due_date_coverage'] * 100:.1f}%)
"""
        )
        for warning in audit.value.get("warnings", []):
            st.caption(f"⚠️ {warning}")
    else:
        st.caption(audit.reason)

st.divider()
st.caption(
    "Retrospective research dataset — not a live operational feed. "
    "Risk scores are not judgements about learners and require human review before "
    "any hypothetical contact. See the Model & data quality page for limitations."
)
