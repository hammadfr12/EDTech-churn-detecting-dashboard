"""Model & data quality: the honesty page.

Target definition, cutoff, class prevalence, held-out metrics, the confusion
matrix, the threshold trade-off, and the validation results. Anything that
could not be computed says so, with a reason.
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

from app._shared import (
    COLORS,
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
from src import config
from src.analytics import loader

setup_page(
    "Model & data quality",
    "Target definition, prediction cutoff, held-out performance, the operating "
    "threshold, and the data validation results behind them.",
    tags=["Held-out metrics", "Leakage controls"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)

metrics_result = loader.model_metrics()
audit_result = loader.data_audit()
cohort_result = loader.cohort_metrics()

if not metrics_result.ok:
    from app._shared import empty_state

    empty_state(
        "The model has not been trained",
        "There are no metrics to report yet.",
        "python -m src.model.train --cutoff-week 4",
    )
    st.stop()

m = metrics_result.value
test = m["test_metrics"]
at = test["at_threshold"]
rank = test["ranking"]
val = m["validation_metrics"]

# ---------------------------------------------------------------------------
# Definitions first — before any number
# ---------------------------------------------------------------------------
section("Target definition and cutoff")
c1, c2 = st.columns([1.3, 1], gap="medium")
with c1:
    st.markdown(
        f"""
**Target** — {m['target']['label']}

{m['target']['description']}

- Positive class: `{', '.join(m['target']['positive_labels'])}`
- Negative class: `{', '.join(m['target']['negative_labels'])}`
- Excluded: `{', '.join(m['target']['excluded_labels']) or 'none'}`
"""
    )
with c2:
    ds = m["dataset"]
    st.markdown(
        f"""
**Prediction cutoff** — end of course-relative week **{m['cutoff_week']}**
(last day = day {m['cutoff_day']})

- Week mapping: `week = floor(day / 7) + 1` for `day >= 0`
- Day 0 belongs to week 1
- Negative day offsets are pre-course activity, excluded from the weekly
  series and captured separately

**Cohort** — {ds['rows_in_modelling_cohort']:,} of {ds['rows_in_snapshot']:,} records
"""
    )
    if ds["rows_excluded_by_target"]:
        st.caption(
            "Excluded by the target definition: "
            + ", ".join(f"{k} = {v:,}" for k, v in ds["rows_excluded_by_target"].items())
        )

note(
    "The primary target deliberately <strong>excludes Fail</strong>. Failing and "
    "withdrawing are different behaviours, and folding them together would make the "
    "headline number uninterpretable. Switch the target in the sidebar to see how much "
    "the result depends on that choice."
)

# ---------------------------------------------------------------------------
# Held-out metrics
# ---------------------------------------------------------------------------
section(f"Held-out performance (test split, scored once at threshold {at['threshold']:.3f})")
prev = m["splits"]["test"]["prevalence"]
base = m["splits"]["test"]["majority_class_baseline_accuracy"]

kpi_row(
    [
        kpi("ROC-AUC", fmt_num(rank["roc_auc"]), "ranking quality, held-out"),
        kpi("PR-AUC", fmt_num(rank["pr_auc"]),
            f"baseline prevalence {prev:.3f}", accent=COLORS["accent"]),
        kpi("Precision", fmt_num(at["precision"]), "of flagged learners, truly withdrawn"),
        kpi("Recall", fmt_num(at["recall"]), "of withdrawn learners, correctly flagged"),
        kpi("F1", fmt_num(at["f1"]), "harmonic mean at the threshold"),
        kpi("Majority baseline", fmt_num(base), "accuracy of always guessing the common class",
            accent="#8B93A7"),
    ]
)

note(
    f"The majority-class baseline is <strong>{base:.3f}</strong> accuracy: simply predicting "
    f"'not withdrawn' for everyone. Accuracy is therefore not a useful headline here, which "
    f"is why precision, recall and PR-AUC are reported instead."
)

left, right = st.columns([1, 1], gap="medium")

with left:
    section("Confusion matrix")
    cm = [
        [at["true_negative"], at["false_positive"]],
        [at["false_negative"], at["true_positive"]],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted: not withdrawn", "Predicted: withdrawn"],
            y=["Actual: not withdrawn", "Actual: withdrawn"],
            colorscale=[[0, "#F2F4FA"], [1, COLORS["primary"]]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=17, color=COLORS["ink"]),
            hovertemplate="%{y}<br>%{x}<br>%{z:,} learners<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style_fig(fig, height=310, legend=False), use_container_width=True)
    st.caption(
        f"{at['false_negative']:,} withdrawn learners were missed and "
        f"{at['false_positive']:,} learners were flagged who did not withdraw. "
        "Both are real costs; the threshold below is how the trade-off is set."
    )

with right:
    section("Threshold trade-off (validation split)")
    curve = m.get("threshold_curve")
    if curve:
        cur = pd.DataFrame(curve)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cur["threshold"], y=cur["precision"], name="Precision",
                line=dict(color=COLORS["primary"], width=2.2),
                hovertemplate="threshold %{x:.3f}<br>precision %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cur["threshold"], y=cur["recall"], name="Recall",
                line=dict(color=COLORS["accent"], width=2.2),
                hovertemplate="threshold %{x:.3f}<br>recall %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cur["threshold"], y=cur["flagged_share"], name="Share flagged",
                line=dict(color="#E0A100", width=1.8, dash="dot"),
                hovertemplate="threshold %{x:.3f}<br>%{y:.1%} of cohort flagged<extra></extra>",
            )
        )
        fig.add_vline(
            x=at["threshold"], line=dict(color=COLORS["ink"], width=1.6),
            annotation_text="chosen", annotation_position="top",
            annotation_font=dict(size=10),
        )
        fig.update_xaxes(title="Decision threshold")
        fig.update_yaxes(title="Value", range=[0, 1.02])
        st.plotly_chart(style_fig(fig, height=310), use_container_width=True)
    else:
        st.caption("The threshold curve was not stored with this artifact.")

# ---------------------------------------------------------------------------
# How the threshold was chosen
# ---------------------------------------------------------------------------
section("How the operating threshold was chosen")
th = m["threshold"]
st.markdown(
    f"""
**Rule** — `{th['rule']}`

{th['reason']}

- Chosen on the **validation split**, never on the test set
- Capacity constraint: at most **{th.get('max_queue_share_constraint', 0.25):.0%}** of the cohort flagged
- Validation F1 **{fmt_num(th.get('validation_f1'))}**, recall **{fmt_num(th.get('validation_recall'))}**, precision **{fmt_num(th.get('validation_precision'))}**
- Validation queue size **{fmt_pct(th.get('validation_queue_share'))}** of the cohort
"""
)
st.caption(
    "The capacity constraint is the mentor-capacity story: a queue a team can actually "
    "work through. Choosing F1 as the objective reflects that both false positives "
    "(wasted mentor time) and false negatives (missed learners) matter."
)

# ---------------------------------------------------------------------------
# Validation vs test
# ---------------------------------------------------------------------------
section("Validation vs test")
cmp = pd.DataFrame(
    [
        {
            "Split": "Validation (threshold chosen here)",
            "Learners": m["splits"]["validation"]["n"],
            "Prevalence": round(m["splits"]["validation"]["prevalence"], 4),
            "ROC-AUC": val["ranking"]["roc_auc"],
            "PR-AUC": val["ranking"]["pr_auc"],
            "Precision": val["at_threshold"]["precision"],
            "Recall": val["at_threshold"]["recall"],
            "F1": val["at_threshold"]["f1"],
        },
        {
            "Split": "Test (scored once, reported)",
            "Learners": m["splits"]["test"]["n"],
            "Prevalence": round(m["splits"]["test"]["prevalence"], 4),
            "ROC-AUC": rank["roc_auc"],
            "PR-AUC": rank["pr_auc"],
            "Precision": at["precision"],
            "Recall": at["recall"],
            "F1": at["f1"],
        },
    ]
)
st.dataframe(cmp, hide_index=True, use_container_width=True)

cv = m.get("cross_validation", {})
if cv.get("available"):
    st.markdown(
        f"**Group-aware cross-validation** on the training partition "
        f"({cv['folds']}-fold, grouped by learner-course presentation): "
        f"PR-AUC **{cv['mean']}** ± {cv['std']}"
    )
else:
    st.caption(f"Cross-validation: not available — {cv.get('reason', 'unknown reason')}")

if m.get("tree_comparison"):
    tc = m["tree_comparison"]
    st.markdown(
        f"**Comparison model** — {tc['model']}: validation PR-AUC "
        f"**{fmt_num(tc['validation_pr_auc'])}**"
    )
    st.caption(
        "The logistic baseline is kept as the shipped model. The tree does not beat it, "
        "so there is no accuracy-for-explainability trade-off here — the explainable model "
        "is also the better-ranking one."
    )

# ---------------------------------------------------------------------------
# Leakage controls
# ---------------------------------------------------------------------------
section("Leakage controls")
st.markdown(
    """
- The feature matrix is assembled by an explicit allow-list. Any column not
  declared in `src/model/features.py` raises an error rather than joining the model.
- `final_result`, `target`, `date_unregistration`, `date_registration` and
  whole-course aggregates are on a hard deny-list with a stated reason each.
- Every per-week feature is filtered to weeks `<= cutoff` in SQL, so a week-5
  click physically cannot enter a week-4 snapshot.
- The train/validation/test split is grouped by learner-course presentation, so
  no learning instance appears on both sides.
- Preprocessing (imputation, scaling, encoding) is fitted inside the pipeline
  and therefore only ever sees training folds.
- `tests/test_leakage.py` injects post-cutoff activity and asserts the week-4
  feature vector is unchanged.
"""
)

# ---------------------------------------------------------------------------
# Feature indicators
# ---------------------------------------------------------------------------
section("Feature indicators")
if m.get("feature_indicators"):
    ind = pd.DataFrame(m["feature_indicators"])
    st.dataframe(
        ind.rename(
            columns={
                "feature": "Feature",
                "importance_mean": "Importance (mean)",
                "importance_std": "Std dev",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Permutation importance on the validation split, measured as the drop in PR-AUC "
        "when a feature is shuffled. This is a ranking of predictive contribution in "
        "historical data — not a causal explanation."
    )

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
section("Data quality and validation")
if audit_result.ok:
    a = audit_result.value
    c = a["cohort"]
    cal = a["assessment_calendar"]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Learner-course records", fmt_int(c["learner_course_records"]))
    q2.metric("Duplicate learner-course rows", fmt_int(c["duplicate_learner_course_rows"]))
    q3.metric("Outcome labels", len(c["outcome_counts"]))
    q4.metric("Assessments with a due date", f"{cal['assessments_with_due_date']}/{cal['total_assessments']}")

    tables = pd.DataFrame(
        [
            {
                "File": t["filename"],
                "Rows": t["rows"],
                "Declared grain unique": t.get("duplicate_rows_on_key", "n/a"),
                "Missing required columns": ", ".join(t["missing_required_columns"]) or "none",
            }
            for t in a["tables"]
        ]
    )
    st.dataframe(tables, hide_index=True, use_container_width=True)

    for warning in a.get("warnings", []):
        note(warning, kind="warn")
else:
    st.caption(audit_result.reason)

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
section("Limitations")
st.markdown(
    "\n".join(f"- {item}" for item in m.get("limitations", []))
)
note(m["warning"], kind="danger")

st.divider()
st.caption(
    "All metrics above are computed on held-out data from a retrospective research "
    "dataset. They describe how well the model ranks historical learners, not how well "
    "any intervention would work."
)
