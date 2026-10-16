"""Risk queue: the operational surface.

A sortable, filterable, exportable queue of learners ranked by modelled risk,
with interpretable indicators and a link through to the learner timeline.

The queue is a triage list for human review. It is not an outreach list, and
the page says so.
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
    RISK_COLORS,
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
    risk_band_series,
    section,
    setup_page,
    sidebar_status,
    style_fig,
)
from src import config
from src.analytics import loader

setup_page(
    "Risk queue",
    "Learners ranked by modelled non-completion risk at the prediction cutoff, "
    "with the indicators that drove the score.",
    tags=["Triage", "Human review required"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)

queue_result = loader.risk_queue()
metrics_result = loader.model_metrics()

if not queue_result.ok:
    empty_state(
        "No risk queue yet",
        "The model has not been trained, so there is nothing to rank.",
        "python -m src.model.train --cutoff-week 4",
    )
    st.stop()

queue = queue_result.value
bands = filters["risk_bands"]
metrics = metrics_result.value if metrics_result.ok else {}

trained_cutoff = metrics.get("cutoff_week")
trained_target = metrics.get("target", {}).get("key")
if trained_cutoff and trained_cutoff != filters["cutoff_week"]:
    note(
        f"The trained artifact was built for <strong>week {trained_cutoff}</strong>, "
        f"but the sidebar is set to week {filters['cutoff_week']}. Scores below are the "
        f"week-{trained_cutoff} scores. Re-run "
        f"<code>python -m src.model.train --cutoff-week {filters['cutoff_week']}</code> "
        "to score the other cutoff.",
        kind="warn",
    )
if trained_target and trained_target != filters["target_key"]:
    note(
        f"The trained artifact uses target <strong>{trained_target}</strong>, but the "
        f"sidebar is set to <strong>{filters['target_key']}</strong>. Scores below are "
        "for the trained target. Re-run training with "
        f"<code>--target {filters['target_key']}</code> to switch.",
        kind="warn",
    )

q = apply_cohort_filter(queue, filters["course"], filters["presentation"]).copy()
q["risk_band"] = risk_band_series(q["risk_probability"], bands)
threshold = float(metrics.get("threshold", {}).get("threshold", config.DEFAULT_THRESHOLD))

scope = (
    f"{filters['course']} · {filters['presentation']}"
    if filters["course"] != "All courses" or filters["presentation"] != "All presentations"
    else "All cohorts"
)

# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
total = len(q)
high = int((q["risk_band"] == "High").sum())
medium = int((q["risk_band"] == "Medium").sum())
low = int((q["risk_band"] == "Low").sum())
flagged = int((q["risk_probability"] >= threshold).sum())

section(f"Queue summary · {scope}")
kpi_row(
    [
        kpi("Learners scored", fmt_int(total), "modelling cohort in this selection"),
        kpi("High risk", fmt_int(high), f"{fmt_pct(high / total if total else None)} of the queue",
            accent=RISK_COLORS["High"], bar_fraction=(high / total) if total else None),
        kpi("Medium risk", fmt_int(medium), f"{fmt_pct(medium / total if total else None)} of the queue",
            accent=RISK_COLORS["Medium"], bar_fraction=(medium / total) if total else None),
        kpi("Flagged at threshold", fmt_int(flagged),
            f"score ≥ {threshold:.3f}", accent=COLORS["primary"],
            bar_fraction=(flagged / total) if total else None),
    ]
)

note(
    "A risk score is a <strong>rank-ordering of observed behaviour</strong>, not a judgement "
    "about a learner, their ability or their intent. This queue exists to focus limited "
    "mentor attention. Every entry requires human review before any action, and nothing "
    "in this application contacts a learner."
)

# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------
left, right = st.columns([1.4, 1], gap="medium")

with left:
    section("Risk score distribution")
    fig = go.Figure(
        go.Histogram(
            x=q["risk_probability"], nbinsx=50,
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            hovertemplate="score ≈ %{x:.2f}<br>%{y} learners<extra></extra>",
        )
    )
    for value, colour, label in (
        (bands["low"], RISK_COLORS["Medium"], "low/medium"),
        (bands["high"], RISK_COLORS["High"], "medium/high"),
    ):
        fig.add_vline(x=value, line=dict(color=colour, width=1.4, dash="dot"))
    fig.add_vline(
        x=threshold, line=dict(color=COLORS["ink"], width=1.8),
        annotation_text=f"threshold {threshold:.2f}", annotation_position="top",
        annotation_font=dict(size=10, color=COLORS["ink"]),
    )
    fig.update_xaxes(title="Modelled risk probability", range=[0, 1])
    fig.update_yaxes(title="Learners")
    st.plotly_chart(style_fig(fig, height=300, legend=False), use_container_width=True)

with right:
    section("Band composition")
    counts = pd.Series(
        {"High": high, "Medium": medium, "Low": low}
    ).reindex(["High", "Medium", "Low"]).dropna()
    fig = go.Figure(
        go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker=dict(color=[RISK_COLORS[i] for i in counts.index], line=dict(width=0)),
            text=[f"{int(v):,}" for v in counts.values], textposition="outside",
            textfont=dict(size=11, color=COLORS["muted"]),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[0, max(counts.max() * 1.3, 1) if len(counts) else 1])
    st.plotly_chart(style_fig(fig, height=300, legend=False), use_container_width=True)

# ---------------------------------------------------------------------------
# The queue table
# ---------------------------------------------------------------------------
section("At-risk learner queue")

f1, f2, f3 = st.columns([1, 1, 1.4])
with f1:
    band_filter = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
with f2:
    only_flagged = st.checkbox("Only score ≥ threshold", value=False)
with f3:
    search = st.text_input("Search learner ID or course", "")

view = q[q["risk_band"].isin(band_filter)] if band_filter else q
if only_flagged:
    view = view[view["risk_probability"] >= threshold]
if search.strip():
    needle = search.strip().lower()
    view = view[
        view["id_student"].astype(str).str.contains(needle, na=False)
        | view["code_module"].str.lower().str.contains(needle, na=False)
        | view["code_presentation"].str.lower().str.contains(needle, na=False)
    ]

display_cols = {
    "id_student": "Learner ID",
    "code_module": "Course",
    "code_presentation": "Presentation",
    "risk_probability": "Risk score",
    "risk_band": "Band",
    "last_active_week_in_window": "Last active week",
    "days_since_last_activity": "Days since activity",
    "total_clicks_to_cutoff": "Clicks to cutoff",
    "submission_rate": "Submission rate",
    "mean_score": "Mean score",
    "click_slope": "Click trend (slope)",
}
available = {k: v for k, v in display_cols.items() if k in view.columns}
table = view[list(available)].rename(columns=available).copy()
table["Risk score"] = table["Risk score"].round(4)
if "Submission rate" in table:
    table["Submission rate"] = table["Submission rate"].round(3)
if "Mean score" in table:
    table["Mean score"] = table["Mean score"].round(1)
if "Click trend (slope)" in table:
    table["Click trend (slope)"] = table["Click trend (slope)"].round(2)

st.caption(
    f"{len(table):,} learners shown, sorted by risk score. "
    "Indicators are shown for review context; they are associations learned from "
    "historical data, not causes."
)

st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    height=440,
    column_config={
        "Risk score": st.column_config.ProgressColumn(
            "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "Learner ID": st.column_config.NumberColumn("Learner ID", format="%d"),
        "Clicks to cutoff": st.column_config.NumberColumn("Clicks to cutoff", format="%d"),
        "Days since activity": st.column_config.NumberColumn("Days since activity", format="%d"),
        "Submission rate": st.column_config.NumberColumn("Submission rate", format="%.2f"),
        "Mean score": st.column_config.NumberColumn("Mean score", format="%.1f"),
    },
)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download this queue as CSV",
    data=csv,
    file_name=f"risk_queue_{filters['course'].replace(' ', '_')}_w{filters['cutoff_week']}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# What drives the score
# ---------------------------------------------------------------------------
section("What drives the score")
if metrics.get("feature_indicators"):
    ind = pd.DataFrame(metrics["feature_indicators"]).head(12)
    ind = ind[ind["importance_mean"] > 0].sort_values("importance_mean")
    if len(ind):
        fig = go.Figure(
            go.Bar(
                x=ind["importance_mean"], y=ind["feature"], orientation="h",
                marker=dict(color=COLORS["primary"], line=dict(width=0)),
                error_x=dict(
                    type="data", array=ind["importance_std"],
                    color=COLORS["faint"], thickness=1,
                ),
                hovertemplate="%{y}<br>importance %{x:.5f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Permutation importance (drop in PR-AUC when shuffled)")
        st.plotly_chart(style_fig(fig, height=380, legend=False), use_container_width=True)
    st.caption(
        "Permutation importance measured on the validation split. It describes how much "
        "each feature contributed to <em>ranking</em> in this retrospective dataset. "
        "It is not a causal explanation and must not be presented to a learner as one."
    )
else:
    st.caption("Train the model to populate feature indicators.")

st.divider()
st.caption(
    "Scores come from a model trained on historical OULAD outcomes. "
    "Precision at the operating threshold is reported on the Model & data quality page; "
    "expect false positives, and treat the queue as a starting point for review."
)
