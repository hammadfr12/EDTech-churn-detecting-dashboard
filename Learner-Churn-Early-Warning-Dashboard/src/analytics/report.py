"""Build report.

Assembles a single Markdown document from the generated artifacts. Every
figure in it is read from disk; nothing is written from memory. If a stage
has not been run, the report says so rather than filling in a number.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config


def _load_json(key: str) -> Optional[Dict[str, Any]]:
    path = config.artifact_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _outcome_table(counts: Dict[str, int], total: int) -> str:
    lines = ["| Outcome | Learners | Share |", "|---|---:|---:|"]
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = (n / total * 100) if total else 0
        lines.append(f"| {label} | {n:,} | {share:.1f}% |")
    lines.append(f"| **Total** | **{total:,}** | **100.0%** |")
    return "\n".join(lines)


def build_report_markdown() -> str:
    audit = _load_json("data_audit")
    cohort = _load_json("cohort_metrics")
    model = _load_json("model_metrics")

    out: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out.append("# Build report — Learner Churn Early-Warning Dashboard\n")
    out.append(
        f"Generated {now} by `python -m src.pipeline`. "
        "Every figure below is read from a generated artifact. "
        "Anything not computed is marked **not computed** rather than estimated.\n"
    )

    # --- dataset ------------------------------------------------------------
    out.append("## 1. Dataset\n")
    if audit:
        c = audit["cohort"]
        out.append(f"- **Source**: {audit['source_url']} (OULAD, public research data)")
        out.append(f"- **Raw directory**: `{audit['data_dir']}`")
        out.append(f"- **Learner-course records**: {c['learner_course_records']:,}")
        out.append(f"- **Distinct learners**: {c['distinct_learners']:,} (pseudonymous IDs)")
        out.append(
            f"- **Course presentations**: {c['course_presentations']} "
            f"across {c['distinct_modules']} modules"
        )
        out.append(
            f"- **Duplicate learner-course rows**: {c['duplicate_learner_course_rows']}"
        )
        out.append("")
        out.append("### Outcome distribution\n")
        out.append(_outcome_table(c["outcome_counts"], c["learner_course_records"]))
        out.append("")
        cal = audit["assessment_calendar"]
        out.append("### Assessment calendar\n")
        out.append(
            f"- {cal['assessments_with_due_date']} of {cal['total_assessments']} "
            f"assessments carry a real due date ({cal['due_date_coverage'] * 100:.1f}%)."
        )
        out.append(
            "- Lateness is computed only on that subset. No due date is ever imputed."
        )
        out.append("")
        out.append("### Raw tables\n")
        out.append("| File | Rows | Missing required columns |")
        out.append("|---|---:|---|")
        for table in audit["tables"]:
            missing = ", ".join(table["missing_required_columns"]) or "none"
            out.append(f"| `{table['filename']}` | {table['rows']:,} | {missing} |")
        out.append("")
        if audit.get("warnings"):
            out.append("### Data warnings\n")
            for warning in audit["warnings"]:
                out.append(f"- {warning}")
            out.append("")
    else:
        out.append("**not computed** — run `python -m src.data.audit --data-dir data/raw`.\n")

    # --- features -----------------------------------------------------------
    out.append("## 2. Feature layer\n")
    if cohort:
        out.append(
            f"- **Week mapping**: {cohort['week_mapping']}"
        )
        out.append(f"- **Cutoff weeks built**: {cohort['cutoff_weeks_built']}")
        out.append(f"- **Default cutoff week**: {cohort['default_cutoff_week']}")
        out.append("")
        out.append("### Snapshot shape\n")
        out.append("| Cutoff week | Rows | Columns |")
        out.append("|---|---:|---:|")
        for week, shape in sorted(cohort["snapshot_shapes"].items(), key=lambda kv: int(kv[0])):
            out.append(f"| {week} | {shape['rows']:,} | {shape['columns']} |")
        out.append("")
        out.append("### Join-grain checks\n")
        out.append("| Relation | Rows | Duplicate key rows | Key |")
        out.append("|---|---:|---:|---|")
        for relation, info in cohort["grain"].items():
            flag = " ⚠️" if info["duplicate_key_rows"] else ""
            out.append(
                f"| `{relation}` | {info['rows']:,} | {info['duplicate_key_rows']}{flag} "
                f"| `{info['key']}` |"
            )
        out.append("")
        out.append("### Measured activity decline\n")
        out.append(
            "Activity decline is **not** withdrawal. Denominator is the previous "
            "week's active learners across all cohorts.\n"
        )
        out.append("| Week | Active (prev) | Active (now) | Net change | Drop-off |")
        out.append("|---:|---:|---:|---:|---:|")
        for row in cohort["cohort"]["steepest_dropoff_weeks"]:
            out.append(
                f"| {int(row['week'])} | {int(row['prev_active']):,} "
                f"| {int(row['active_learners']):,} | {int(row['net_change']):,} "
                f"| {row['dropoff_rate'] * 100:.1f}% |"
            )
        out.append("")
    else:
        out.append("**not computed** — run `python -m src.features.build`.\n")

    # --- model --------------------------------------------------------------
    out.append("## 3. Model\n")
    if model:
        t = model["target"]
        out.append("### Target definition\n")
        out.append(f"- **{t['label']}**")
        out.append(f"- {t['description']}")
        out.append(f"- Positive: `{', '.join(t['positive_labels'])}`")
        out.append(f"- Negative: `{', '.join(t['negative_labels'])}`")
        out.append(f"- Excluded: `{', '.join(t['excluded_labels']) or 'none'}`")
        out.append("")
        out.append("### Cutoff and cohort\n")
        out.append(
            f"- **Prediction cutoff**: end of week {model['cutoff_week']} "
            f"(last day = day {model['cutoff_day']})"
        )
        out.append(
            f"- **Modelling cohort**: {model['dataset']['rows_in_modelling_cohort']:,} of "
            f"{model['dataset']['rows_in_snapshot']:,} records"
        )
        if model["dataset"]["rows_excluded_by_target"]:
            excluded = ", ".join(
                f"{k} = {v:,}" for k, v in model["dataset"]["rows_excluded_by_target"].items()
            )
            out.append(f"- **Excluded by target**: {excluded}")
        out.append(f"- **Features**: {model['feature_count']}")
        out.append("")

        out.append("### Split strategy\n")
        out.append(f"- {model['splits']['strategy']}")
        out.append("")
        out.append("| Split | Learners | Positive | Prevalence | Majority baseline |")
        out.append("|---|---:|---:|---:|---:|")
        for split in ("train", "validation", "test"):
            s = model["splits"][split]
            out.append(
                f"| {split.title()} | {s['n']:,} | {s['n_positive']:,} "
                f"| {_fmt(s['prevalence'], 4)} "
                f"| {_fmt(s['majority_class_baseline_accuracy'], 4)} |"
            )
        out.append("")

        th = model["threshold"]
        at = model["test_metrics"]["at_threshold"]
        rank = model["test_metrics"]["ranking"]
        out.append("### Operating threshold\n")
        out.append(f"- **Threshold**: {_fmt(at['threshold'], 4)}")
        out.append(f"- **Rule**: `{th['rule']}` — {th['reason']}")
        out.append(f"- **Chosen on**: {th['selected_on']}")
        out.append("")

        out.append("### Held-out performance (test split)\n")
        out.append("| Metric | Value |")
        out.append("|---|---:|")
        out.append(f"| ROC-AUC | {_fmt(rank['roc_auc'])} |")
        out.append(f"| PR-AUC | {_fmt(rank['pr_auc'])} |")
        out.append(f"| Brier score | {_fmt(rank['brier'])} |")
        out.append(f"| Precision | {_fmt(at['precision'])} |")
        out.append(f"| Recall | {_fmt(at['recall'])} |")
        out.append(f"| F1 | {_fmt(at['f1'])} |")
        out.append("")
        out.append("### Confusion matrix (test split)\n")
        out.append("| | Predicted negative | Predicted positive |")
        out.append("|---|---:|---:|")
        out.append(
            f"| **Actual negative** | {at['true_negative']:,} | {at['false_positive']:,} |"
        )
        out.append(
            f"| **Actual positive** | {at['false_negative']:,} | {at['true_positive']:,} |"
        )
        out.append("")

        cv = model.get("cross_validation", {})
        if cv.get("available"):
            out.append(
                f"**Group-aware cross-validation** (training partition, {cv['folds']}-fold): "
                f"PR-AUC **{cv['mean']}** ± {cv['std']}\n"
            )
        else:
            out.append(
                f"**Cross-validation**: not available — {cv.get('reason', 'unknown')}\n"
            )

        out.append("### Top feature indicators\n")
        out.append("| Feature | Permutation importance (mean) | Std dev |")
        out.append("|---|---:|---:|")
        for row in model["feature_indicators"][:15]:
            out.append(
                f"| `{row['feature']}` | {row['importance_mean']:+.5f} "
                f"| {row['importance_std']:.5f} |"
            )
        out.append("")
        out.append(
            "Permutation importance on the validation split, measured as the drop in "
            "PR-AUC when a feature is shuffled. A ranking of predictive contribution, "
            "**not** a causal explanation.\n"
        )

        out.append("### Limitations\n")
        for item in model["limitations"]:
            out.append(f"- {item}")
        out.append("")
        out.append(f"> {model['warning']}\n")
    else:
        out.append("**not computed** — run `python -m src.model.train --cutoff-week 4`.\n")

    # --- how to reproduce ---------------------------------------------------
    out.append("## 4. Reproducing this report\n")
    out.append("```bash")
    out.append("python -m venv .venv && source .venv/bin/activate")
    out.append("pip install -r requirements.txt")
    out.append("python -m src.pipeline --download")
    out.append("pytest")
    out.append("streamlit run app/Home.py")
    out.append("```\n")

    return "\n".join(out)


def write_build_report() -> Path:
    config.ensure_dirs()
    path = config.artifact_path("build_report")
    path.write_text(build_report_markdown(), encoding="utf-8")
    return path


if __name__ == "__main__":  # pragma: no cover
    print(write_build_report())
