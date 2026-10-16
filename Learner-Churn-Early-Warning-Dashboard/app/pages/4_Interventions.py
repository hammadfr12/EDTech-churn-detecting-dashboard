"""Interventions: routing a reviewed learner to a simulated owner.

A local workflow demonstrating the *process*. It assigns an owner, an action
and a due date, and tracks status. It sends nothing to anyone, and the page
is explicit that this demonstrates design rather than measured impact.
"""

from __future__ import annotations

import pandas as pd
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
    empty_state,
    fmt_int,
    fmt_pct,
    global_filters,
    kpi,
    kpi_row,
    load_snapshot_or_stop,
    note,
    rel_path,
    risk_band_series,
    section,
    setup_page,
    sidebar_status,
)
from src import config
from src.analytics import loader
from src.interventions import store

setup_page(
    "Interventions",
    "Assign a reviewed learner to a simulated owner with a supportive action "
    "and a due date, then track the status through to resolution.",
    tags=["Process design", "Nothing is sent"],
)

snapshot = load_snapshot_or_stop()
sidebar_status()
filters = global_filters(snapshot)
bands = filters["risk_bands"]
cutoff = filters["cutoff_week"]
target_key = filters["target_key"]

queue_result = loader.risk_queue()
metrics_result = loader.model_metrics()
threshold = (
    float(metrics_result.value.get("threshold", {}).get("threshold", config.DEFAULT_THRESHOLD))
    if metrics_result.ok
    else config.DEFAULT_THRESHOLD
)

if not queue_result.ok:
    empty_state(
        "No risk queue yet",
        "Train the model before routing learners to reviewers.",
        "python -m src.model.train --cutoff-week 4",
    )
    st.stop()

queue = queue_result.value.copy()
queue["risk_band"] = risk_band_series(queue["risk_probability"], bands)
candidates = queue[queue["risk_probability"] >= threshold].copy()
candidates = candidates.sort_values("risk_probability", ascending=False)

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
records = store.fetch()
n_flagged = int((queue["risk_probability"] >= threshold).sum())
n_with_owner = 0
n_open = 0
if len(records):
    n_with_owner = int(records["owner"].notna().sum())
    n_open = int((~records["status"].isin(["Resolved", "No response", "Not appropriate"])).sum())

section("Intervention coverage")
kpi_row(
    [
        kpi("Flagged at threshold", fmt_int(n_flagged), f"score ≥ {threshold:.3f}"),
        kpi("Records with an owner", fmt_int(n_with_owner),
            f"{fmt_pct(n_with_owner / n_flagged if n_flagged else None)} of flagged learners",
            accent=COLORS["primary"],
            bar_fraction=(n_with_owner / n_flagged) if n_flagged else None),
        kpi("Still open", fmt_int(n_open), "not yet resolved or closed",
            accent=RISK_COLORS["Medium"]),
        kpi("Total records", fmt_int(len(records)), "all statuses"),
    ]
)

note(
    "Intervention coverage is a <strong>process metric</strong>. It says whether learners "
    "were reviewed, not whether retention improved. No claim of impact is made anywhere "
    "in this application, and none should be inferred without a controlled evaluation."
)

# ---------------------------------------------------------------------------
# Create / update
# ---------------------------------------------------------------------------
tab_create, tab_track = st.tabs(["Assign a learner", "Track and update"])

with tab_create:
    section("Assign a learner to a reviewer")

    if not len(candidates):
        st.caption("No learners currently meet the threshold in this cohort.")
    else:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            options = candidates.head(500)
            labels = [
                f"{r.id_student} · {r.code_module} {r.code_presentation} · "
                f"score {r.risk_probability:.3f} · {r.risk_band}"
                for r in options.itertuples()
            ]
            picked = st.selectbox("Learner (highest risk first)", range(len(labels)),
                                  format_func=lambda i: labels[i])
            chosen = options.iloc[picked]

        def _cell(column: str, fmt: str = "{}") -> str:
            """Format one optional field, or an em-dash when it is absent.

            A learner with no recorded activity has a null last-active week;
            rendering that raw would print 'nan', which reads as a bug rather
            than as 'never active'.
            """
            value = chosen.get(column)
            if value is None or pd.isna(value):
                return "—"
            return fmt.format(value)

        with c2:
            st.markdown(
                f"""
**Risk score** &nbsp; `{chosen['risk_probability']:.3f}`
**Band** &nbsp; `{chosen['risk_band']}`
**Last active week** &nbsp; `{_cell('last_active_week_in_window')}`
**Days since activity** &nbsp; `{_cell('days_since_last_activity')}`
**Submission rate** &nbsp; `{_cell('submission_rate', '{:.2f}')}`
"""
            )

        with st.form("assign_form", clear_on_submit=False):
            f1, f2, f3 = st.columns(3)
            with f1:
                owner = st.text_input("Owner (mentor or reviewer)", placeholder="e.g. mentor-A")
            with f2:
                action = st.selectbox("Suggested action", list(config.INTERVENTION_ACTIONS))
            with f3:
                due = st.date_input("Due date", value=None)
            status = st.selectbox("Status", list(config.INTERVENTION_STATUSES), index=2)
            notes = st.text_area(
                "Minimal note",
                placeholder="Keep it short. No contact details, no sensitive personal data.",
                height=80,
            )
            submitted = st.form_submit_button("Save intervention record", type="primary")

        if submitted:
            if not owner.strip():
                st.error("An owner is required before a learner can be assigned.")
            else:
                record_id = store.upsert(
                    code_module=str(chosen["code_module"]),
                    code_presentation=str(chosen["code_presentation"]),
                    id_student=int(chosen["id_student"]),
                    cutoff_week=cutoff,
                    target_key=target_key,
                    risk_probability=float(chosen["risk_probability"]),
                    risk_band=str(chosen["risk_band"]),
                    owner=owner.strip(),
                    action=action,
                    due_date=due.isoformat() if due else None,
                    status=status,
                    notes=notes.strip() or None,
                )
                st.success(f"Saved intervention record #{record_id} for learner {chosen['id_student']}.")
                st.cache_data.clear()

        st.caption(
            "Suggested actions are supportive by design: a check-in, clarifying assessment "
            "requirements, sharing a resource, or inviting to office hours. Punitive actions "
            "are deliberately not offered."
        )

with tab_track:
    section("Existing intervention records")
    records = store.fetch()

    if not len(records):
        st.caption("No intervention records yet. Assign a learner on the first tab.")
    else:
        summary = (
            records["status"].value_counts().reindex(config.INTERVENTION_STATUSES).dropna()
        )
        st.dataframe(
            summary.rename("Count").reset_index().rename(columns={"index": "Status"}),
            hide_index=True,
            use_container_width=False,
        )

        show = records[
            [
                "id", "code_module", "code_presentation", "id_student",
                "risk_probability", "risk_band", "owner", "action",
                "due_date", "status", "notes", "updated_at",
            ]
        ].rename(
            columns={
                "id": "Record", "code_module": "Course",
                "code_presentation": "Presentation", "id_student": "Learner ID",
                "risk_probability": "Risk", "risk_band": "Band",
                "owner": "Owner", "action": "Action", "due_date": "Due",
                "status": "Status", "notes": "Note", "updated_at": "Updated",
            }
        )
        st.dataframe(show, hide_index=True, use_container_width=True, height=320)

        st.download_button(
            "Download all intervention records as CSV",
            data=records.to_csv(index=False).encode("utf-8"),
            file_name="intervention_records.csv",
            mime="text/csv",
        )

        section("Update a record")
        record_ids = records["id"].tolist()
        picked_id = st.selectbox("Record", record_ids, format_func=lambda i: f"#{i}")
        current = records[records["id"] == picked_id].iloc[0]

        u1, u2, u3, u4 = st.columns(4)
        with u1:
            new_status = st.selectbox(
                "Status", list(config.INTERVENTION_STATUSES),
                index=list(config.INTERVENTION_STATUSES).index(current["status"]),
                key="upd_status",
            )
        with u2:
            new_owner = st.text_input("Owner", value=current["owner"] or "", key="upd_owner")
        with u3:
            actions = list(config.INTERVENTION_ACTIONS)
            new_action = st.selectbox(
                "Action", actions,
                index=actions.index(current["action"]) if current["action"] in actions else 0,
                key="upd_action",
            )
        with u4:
            new_due = st.text_input("Due date (YYYY-MM-DD)", value=current["due_date"] or "",
                                    key="upd_due")
        new_notes = st.text_area("Note", value=current["notes"] or "", height=70, key="upd_notes")

        b1, b2 = st.columns([1, 4])
        with b1:
            if st.button("Save update", type="primary"):
                store.update(
                    int(picked_id),
                    status=new_status,
                    owner=new_owner.strip() or None,
                    action=new_action,
                    due_date=new_due.strip() or None,
                    notes=new_notes.strip() or None,
                )
                st.success(f"Record #{picked_id} updated.")
                st.cache_data.clear()
                st.rerun()
        with b2:
            if st.button("Delete record"):
                store.delete(int(picked_id))
                st.warning(f"Record #{picked_id} deleted.")
                st.cache_data.clear()
                st.rerun()

st.divider()
st.caption(
    "Records are stored locally in a SQLite file "
    f"(`{rel_path(config.INTERVENTION_DB)}`) and hold only the "
    "pseudonymous OULAD learner ID. No names, emails, phone numbers or other contact "
    "details are collected or required."
)
