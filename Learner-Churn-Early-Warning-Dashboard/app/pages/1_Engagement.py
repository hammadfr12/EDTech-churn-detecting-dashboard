"""Engagement: how activity changes week by week, and what learners do.

Keeps activity drop-off explicitly distinct from withdrawal. The page never
says a learner "dropped out" because clicks fell.
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
    SERIES_SEQUENCE,
    apply_cohort_filter,
    empty_state,
    fmt_int,
    fmt_num,
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
from src.analytics import loader

setup_page(
    "Engagement",
    "Weekly active rates, click volume, activity decay and assessment participation.",
    tags=["Behaviour", "Cutoff-safe"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)

weekly = loader.weekly_cohort()
dropoff = loader.activity_dropoff()
events = loader.assessment_events()
weeks = loader.learner_week()

if not weekly.ok:
    empty_state(
        "Weekly activity is missing",
        "Run the feature build to generate engagement artifacts.",
        "python -m src.features.build --data-dir data/raw",
    )
    st.stop()

w = apply_cohort_filter(weekly.value, filters["course"], filters["presentation"])
cohort = apply_cohort_filter(
    snapshot, filters["course"], filters["presentation"]
)
scope = (
    f"{filters['course']} · {filters['presentation']}"
    if filters["course"] != "All courses" or filters["presentation"] != "All presentations"
    else "All cohorts"
)

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
cutoff = filters["cutoff_week"]
agg = (
    w[w["week"] <= 25]
    .groupby("week", as_index=False)
    .agg(active=("active_learners", "sum"), enrolled=("learners_enrolled", "sum"),
         clicks=("total_clicks", "sum"))
)
agg["rate"] = agg["active"] / agg["enrolled"]

enrolled = int(cohort.shape[0])
wk = agg[agg["week"] == cutoff]
active_cutoff = int(wk["active"].iloc[0]) if len(wk) else 0
rate_cutoff = (wk["rate"].iloc[0]) if len(wk) else None

clicks_cutoff = int(cohort["total_clicks_to_cutoff"].sum()) if "total_clicks_to_cutoff" in cohort else 0
per_learner = clicks_cutoff / enrolled if enrolled else None

inactive = int((cohort["total_clicks_to_cutoff"] == 0).sum()) if "total_clicks_to_cutoff" in cohort else 0

section(f"Engagement at the week-{cutoff} cutoff · {scope}")
kpi_row(
    [
        kpi("Active in cutoff week", fmt_int(active_cutoff),
            f"{fmt_pct(rate_cutoff)} of {fmt_int(enrolled)} enrolled"),
        kpi("Clicks by cutoff", fmt_int(clicks_cutoff),
            f"{fmt_num(per_learner, 1)} per enrolled learner", accent=COLORS["accent"]),
        kpi("No activity by cutoff", fmt_int(inactive),
            f"{fmt_pct(inactive / enrolled if enrolled else None)} of the cohort",
            accent="#D64550", bar_fraction=(inactive / enrolled) if enrolled else None),
        kpi("Weekly active rate", fmt_pct(rate_cutoff),
            f"week {cutoff} of the presentation"),
    ]
)

note(
    "Every figure on this page uses only activity observed on or before the last day of "
    f"<strong>week {cutoff}</strong>. Clicks after the cutoff are excluded from features by "
    "construction and verified by an automated leakage test."
)

# ---------------------------------------------------------------------------
# Weekly active rate
# ---------------------------------------------------------------------------
section("Weekly active rate and click volume")
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=agg["week"], y=agg["clicks"], name="Total clicks",
        marker=dict(color="rgba(14,159,142,0.35)", line=dict(width=0)),
        hovertemplate="Week %{x}<br>%{y:,} clicks<extra></extra>",
        yaxis="y2",
    )
)
fig.add_trace(
    go.Scatter(
        x=agg["week"], y=agg["rate"], name="Weekly active rate",
        mode="lines+markers", line=dict(color=COLORS["primary"], width=2.6),
        marker=dict(size=6),
        hovertemplate="Week %{x}<br>%{y:.1%} active<extra></extra>",
        yaxis="y",
    )
)
fig.add_vline(
    x=cutoff + 0.5, line=dict(color=COLORS["faint"], width=1.4, dash="dash"),
    annotation_text=f"cutoff · week {cutoff}", annotation_position="top",
    annotation_font=dict(size=10, color=COLORS["faint"]),
)
fig.update_layout(
    yaxis=dict(title="Weekly active rate", tickformat=".0%"),
    yaxis2=dict(title="Total clicks", overlaying="y", side="right", showgrid=False),
    bargap=0.25,
)
fig.update_xaxes(title="Course-relative week", dtick=2)
st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

# ---------------------------------------------------------------------------
# Drop-off
# ---------------------------------------------------------------------------
section("Week-over-week activity drop-off")
if dropoff.ok:
    d = apply_cohort_filter(dropoff.value, filters["course"], filters["presentation"])
    d = d[d["week"] <= 25]
    if len(d):
        dagg = (
            d.groupby("week", as_index=False)
            .agg(prev=("prev_active_learners", "sum"), now=("active_learners", "sum"))
        )
        dagg["dropoff"] = (dagg["prev"] - dagg["now"]) / dagg["prev"].replace(0, np.nan)
        dagg = dagg.dropna(subset=["dropoff"])

        colors = [
            COLORS["primary"] if v <= 0 else ("#E0A100" if v < 0.10 else "#D64550")
            for v in dagg["dropoff"]
        ]
        fig = go.Figure(
            go.Bar(
                x=dagg["week"], y=dagg["dropoff"],
                marker=dict(color=colors, line=dict(width=0)),
                customdata=np.stack([dagg["prev"], dagg["now"]], axis=-1),
                hovertemplate=(
                    "Week %{x}<br>drop-off %{y:.1%}<br>"
                    "%{customdata[0]:,} → %{customdata[1]:,} active learners<extra></extra>"
                ),
            )
        )
        fig.update_yaxes(title="Drop-off vs previous week", tickformat=".0%")
        fig.update_xaxes(title="Course-relative week", dtick=2)
        st.plotly_chart(style_fig(fig, height=310, legend=False), use_container_width=True)

        steepest = dagg.sort_values("dropoff", ascending=False).head(3)
        bullets = " · ".join(
            f"week {int(r.week)}: <strong>{r.dropoff * 100:.1f}%</strong> "
            f"({int(r.prev):,} → {int(r.now):,})"
            for r in steepest.itertuples()
        )
        note(
            f"Steepest measured declines in this selection — {bullets}. "
            "Denominator is the previous week's active learners; weeks where nothing "
            "was active are excluded rather than treated as a 100% drop. "
            "<strong>This is activity decline, not withdrawal.</strong>",
            kind="warn",
        )
    else:
        st.caption("Not enough weeks in this selection to compute drop-off.")
else:
    st.caption(dropoff.reason)

# ---------------------------------------------------------------------------
# Distribution + assessment trend
# ---------------------------------------------------------------------------
left, right = st.columns(2, gap="medium")

with left:
    section("Activity distribution by cutoff")
    if "total_clicks_to_cutoff" in cohort:
        vals = cohort["total_clicks_to_cutoff"].astype(float)
        fig = go.Figure(
            go.Histogram(
                x=np.log1p(vals), nbinsx=44,
                marker=dict(color=COLORS["primary"], line=dict(width=0)),
                hovertemplate="log(1+clicks) ≈ %{x:.2f}<br>%{y} learners<extra></extra>",
            )
        )
        zero_share = (vals == 0).mean()
        fig.update_xaxes(title="log(1 + clicks by cutoff)")
        fig.update_yaxes(title="Learners")
        st.plotly_chart(style_fig(fig, height=300, legend=False), use_container_width=True)
        st.caption(
            f"Log scale: click volume is heavily right-skewed. "
            f"{zero_share * 100:.1f}% of this cohort recorded no activity at all by week {cutoff}."
        )
    else:
        st.caption("No click features in this selection.")

with right:
    section("Assessment participation and scores")
    if events.ok and "mean_score" in cohort:
        sub = apply_cohort_filter(events.value, filters["course"], filters["presentation"])
        sub = sub[(sub["submitted_day"].notna()) & (sub["submitted_day"] <= cutoff * 7 - 1)]
        if len(sub):
            binned = sub.assign(week=(sub["submitted_day"] // 7 + 1).astype(int))
            trend = binned.groupby("week", as_index=False).agg(
                submissions=("score", "size"), mean_score=("score", "mean")
            )
            trend = trend[trend["week"] <= cutoff]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=trend["week"], y=trend["submissions"], name="Submissions",
                    marker=dict(color="rgba(47,69,197,0.28)", line=dict(width=0)),
                    yaxis="y2",
                    hovertemplate="Week %{x}<br>%{y:,} submissions<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend["week"], y=trend["mean_score"], name="Mean score",
                    mode="lines+markers", line=dict(color=COLORS["accent"], width=2.6),
                    marker=dict(size=6),
                    hovertemplate="Week %{x}<br>mean score %{y:.1f}<extra></extra>",
                )
            )
            fig.update_layout(
                yaxis=dict(title="Mean score", range=[0, 100]),
                yaxis2=dict(title="Submissions", overlaying="y", side="right", showgrid=False),
                bargap=0.35,
            )
            fig.update_xaxes(title="Course-relative week", dtick=1)
            st.plotly_chart(style_fig(fig, height=300), use_container_width=True)
            st.caption(
                "Submissions with a recorded day offset on or before the cutoff. "
                "Assessments without a due date are excluded from lateness statistics "
                "but still counted as submissions."
            )
        else:
            st.caption("No submissions recorded before the cutoff in this selection.")
    else:
        st.caption("Assessment events artifact unavailable.")

# ---------------------------------------------------------------------------
# Lateness — only where genuinely supported
# ---------------------------------------------------------------------------
section("Assessment lateness (only where a due date exists)")
if events.ok:
    sub = apply_cohort_filter(events.value, filters["course"], filters["presentation"])
    dated = sub[sub["has_due_date"] & sub["lateness_days"].notna()]
    undated = sub[~sub["has_due_date"]]
    if len(dated):
        late = (dated["lateness_days"] > 0).mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Submissions with a due date", fmt_int(len(dated)))
        c2.metric("Submitted late", fmt_pct(late))
        c3.metric("Median lateness (days)", fmt_num(dated["lateness_days"].median(), 0))
        c4.metric("Without a usable due date", fmt_int(len(undated)))

        fig = go.Figure(
            go.Histogram(
                x=dated["lateness_days"].clip(-21, 42), nbinsx=48,
                marker=dict(color=COLORS["accent"], line=dict(width=0)),
                hovertemplate="%{x} days<br>%{y} submissions<extra></extra>",
            )
        )
        fig.add_vline(x=0, line=dict(color=COLORS["ink"], width=1.4))
        fig.update_xaxes(title="Days relative to due date (clipped to [-21, 42])")
        fig.update_yaxes(title="Submissions")
        st.plotly_chart(style_fig(fig, height=280, legend=False), use_container_width=True)
        st.caption(
            f"{len(undated):,} submissions in this selection belong to assessments whose "
            "due date is recorded as '?' in OULAD. No due date is ever imputed, so those "
            "submissions contribute to participation counts but never to lateness."
        )
    else:
        st.caption("No assessments with a usable due date in this selection.")
else:
    st.caption(events.reason)
