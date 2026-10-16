"""Train and evaluate the baseline model, then write every artifact the
dashboard needs.

Flow
----
1. Load the cutoff-safe snapshot built by ``src.features.build``.
2. Build the documented target.
3. Split by learning instance into train / validation / test.
4. Fit the preprocessing + logistic pipeline on train only.
5. Choose the operating threshold on validation under a capacity constraint.
6. Score the test set once, at the frozen threshold, and report honestly.
7. Save the pipeline, metrics, feature list, and a scored risk queue.

The test set is scored exactly once, at a threshold that was chosen without
looking at it.

Usage
-----
    python -m src.model.train --cutoff-week 4
    python -m src.model.train --cutoff-week 4 --target fail_as_churn
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src import config
from src.model import evaluate, pipeline as pl
from src.model.features import build_feature_matrix, feature_columns
from src.model.target import TargetSpec, class_balance, label_frame

def rel(path: Path) -> str:
    """Project-relative path for display, falling back to the absolute path."""
    try:
        return str(Path(path).relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


WARNING_TEXT = (
    "This model is trained on the anonymised, retrospective Open University "
    "Learning Analytics Dataset. It is a research and demonstration artifact. "
    "It is NOT validated for real student-support decisions, it does not "
    "establish learner intent or ability, and no learner should be contacted "
    "or judged on the basis of its score without human review."
)


def load_snapshot(cutoff_week: int) -> pd.DataFrame:
    """Prefer the per-cutoff snapshot; fall back to the default artifact."""
    per_week = config.PROCESSED_DIR / f"learner_feature_snapshot_w{cutoff_week}.parquet"
    if per_week.exists():
        return pd.read_parquet(per_week)
    default = config.artifact_path("learner_snapshot")
    if default.exists() and cutoff_week == config.DEFAULT_CUTOFF_WEEK:
        return pd.read_parquet(default)
    raise FileNotFoundError(
        f"No feature snapshot for cutoff week {cutoff_week}. "
        "Run: python -m src.features.build --data-dir data/raw"
    )


def assign_risk_bands(scores: pd.Series, bands: Dict[str, float] | None = None) -> pd.Series:
    """Map probabilities to Low / Medium / High.

    Bands are a queue-organising device, not a claim about the learner. The
    cut points are configurable and shown in the UI.
    """
    bands = bands or config.DEFAULT_RISK_BANDS
    low, high = bands["low"], bands["high"]
    return pd.cut(
        scores,
        bins=[-np.inf, low, high, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)


def train(
    cutoff_week: int = config.DEFAULT_CUTOFF_WEEK,
    target_key: str = config.DEFAULT_TARGET,
    compare_tree: bool = True,
    max_queue_share: float = 0.25,
) -> Dict[str, Any]:
    config.ensure_dirs()
    snapshot = load_snapshot(cutoff_week)
    labelled = label_frame(snapshot, target_key)
    spec = TargetSpec.from_config(target_key)

    X_all, y_all = build_feature_matrix(labelled, cutoff_week=cutoff_week)
    groups_all = pl.group_key(labelled)

    # --- split: train | (validation + test) -------------------------------
    train_df, holdout_df = pl.group_train_test_split(labelled)
    # --- split holdout: validation | test ---------------------------------
    val_df, test_df = pl.group_train_validation_split(holdout_df, test_size=0.5)

    def _xy(frame: pd.DataFrame):
        X, y = build_feature_matrix(frame, cutoff_week=cutoff_week)
        return X, y

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            "Training partition contains a single class; cannot fit a classifier. "
            "Try the other target definition."
        )

    # --- fit on train only -------------------------------------------------
    model = pl.build_logistic_pipeline()
    model.fit(X_train, y_train)

    # --- threshold chosen on VALIDATION only ------------------------------
    val_scores = model.predict_proba(X_val)[:, 1]
    val_curve = evaluate.threshold_curve(y_val.to_numpy(), val_scores)
    threshold_choice = evaluate.choose_threshold(val_curve, max_queue_share=max_queue_share)
    threshold = float(threshold_choice["threshold"])

    # --- score TEST once, at the frozen threshold -------------------------
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate.full_evaluation(y_test.to_numpy(), test_scores, threshold)
    val_metrics = evaluate.full_evaluation(y_val.to_numpy(), val_scores, threshold)

    # --- cross-validated PR-AUC on the training partition -----------------
    cv = pl.cross_validate_pr_auc(
        pl.build_logistic_pipeline(),
        X_train,
        y_train,
        groups=pl.group_key(train_df),
        folds=config.CV_FOLDS,
    )

    # --- optional shallow-tree comparison ---------------------------------
    tree_comparison = None
    if compare_tree:
        tree = pl.build_tree_pipeline(max_depth=4)
        tree.fit(X_train, y_train)
        tree_scores = tree.predict_proba(X_val)[:, 1]
        tree_comparison = {
            "model": "DecisionTreeClassifier(max_depth=4, class_weight=balanced)",
            "validation": evaluate.full_evaluation(y_val.to_numpy(), tree_scores, threshold),
            "validation_pr_auc": evaluate.ranking_metrics(
                y_val.to_numpy(), tree_scores
            )["pr_auc"],
        }

    # --- feature indicators -----------------------------------------------
    indicators = pl.permutation_importance(model, X_val, y_val)

    # --- score the full cohort for the risk queue -------------------------
    X_full, _ = build_feature_matrix(labelled, cutoff_week=cutoff_week)
    full_scores = model.predict_proba(X_full)[:, 1]

    queue = labelled[
        ["code_module", "code_presentation", "id_student", "final_result", "target"]
    ].copy()
    queue["risk_probability"] = full_scores
    queue["risk_band"] = assign_risk_bands(queue["risk_probability"])
    queue["flagged_at_threshold"] = queue["risk_probability"] >= threshold
    # Display-only context, carried so the queue is readable without a join.
    for col in (
        "last_active_week_in_window",
        "days_since_last_activity",
        "total_clicks_to_cutoff",
        "submission_rate",
        "mean_score",
        "click_slope",
    ):
        if col in labelled.columns:
            queue[col] = labelled[col].to_numpy()
    queue = queue.sort_values("risk_probability", ascending=False).reset_index(drop=True)

    # --- persist ----------------------------------------------------------
    model_path = config.artifact_path("model")
    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": feature_columns(),
            "numeric_features": list(X_all.columns),
            "threshold": threshold,
            "cutoff_week": cutoff_week,
            "target_key": target_key,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )

    queue_path = config.artifact_path("risk_queue")
    queue.to_parquet(queue_path, index=False)

    metrics: Dict[str, Any] = {
        "generated_by": "src.model.train",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_week": cutoff_week,
        "cutoff_day": cutoff_week * config.DAYS_PER_WEEK - 1,
        "target": {
            "key": spec.key,
            "label": spec.label,
            "description": spec.description,
            "positive_labels": list(spec.positive_labels),
            "negative_labels": list(spec.negative_labels),
            "excluded_labels": list(spec.excluded_labels),
        },
        "dataset": {
            "rows_in_snapshot": int(len(snapshot)),
            "rows_in_modelling_cohort": int(len(labelled)),
            "rows_excluded_by_target": labelled.attrs.get("excluded_counts", {}),
        },
        "splits": {
            "strategy": (
                "GroupShuffleSplit on (code_module, code_presentation, id_student); "
                "the holdout is split again 50/50 into validation and test."
            ),
            "train": class_balance(train_df),
            "validation": class_balance(val_df),
            "test": class_balance(test_df),
        },
        "threshold": {
            **threshold_choice,
            "max_queue_share_constraint": max_queue_share,
            "selected_on": "validation split",
        },
        # The full validation curve is stored so the dashboard can show the
        # precision/recall/queue-size trade-off the threshold was chosen from.
        "threshold_curve": val_curve,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cross_validation": cv,
        "tree_comparison": tree_comparison,
        "feature_indicators": indicators,
        "feature_count": int(X_all.shape[1]),
        "warning": WARNING_TEXT,
        "limitations": [
            "Retrospective research dataset; not a live operational feed.",
            "Outcomes are historical. Nothing here establishes causation.",
            "The primary target excludes Fail; see the sensitivity target.",
            "Fairness-sensitive attributes (region, imd_band, age_band, gender, "
            "disability) are available to the model but must not drive "
            "intervention decisions on their own.",
            "Assessment lateness is only computed for the subset of assessments "
            "that carry a real due date; it is never imputed.",
        ],
    }
    config.artifact_path("model_metrics").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    test = metrics["test_metrics"]
    at = test["at_threshold"]
    rank = test["ranking"]
    ds = metrics["dataset"]
    print("Model training complete")
    print("=" * 62)
    print(f"  cutoff week            : {metrics['cutoff_week']}")
    print(f"  target                 : {metrics['target']['label']}")
    print(
        f"  modelling cohort       : {ds['rows_in_modelling_cohort']:,} "
        f"of {ds['rows_in_snapshot']:,} snapshot rows"
    )
    if ds["rows_excluded_by_target"]:
        print(f"  excluded by target     : {ds['rows_excluded_by_target']}")
    print(f"  features               : {metrics['feature_count']}")
    prev = metrics["splits"]["test"]["prevalence"]
    base = metrics["splits"]["test"]["majority_class_baseline_accuracy"]
    print(f"  test prevalence        : {prev:.3f}")
    print(f"  majority-class baseline: {base:.3f} accuracy")
    print()
    print(f"  operating threshold    : {at['threshold']:.4f}")
    print(f"    rule                 : {metrics['threshold']['rule']}")
    print(f"  ROC-AUC                : {rank['roc_auc']}")
    print(f"  PR-AUC                 : {rank['pr_auc']}")
    print(f"  precision              : {at['precision']}")
    print(f"  recall                 : {at['recall']}")
    print(f"  F1                     : {at['f1']}")
    print(
        f"  confusion              : TP={at['true_positive']} FP={at['false_positive']} "
        f"FN={at['false_negative']} TN={at['true_negative']}"
    )
    if metrics["cross_validation"]["available"]:
        cv = metrics["cross_validation"]
        print(f"  CV PR-AUC ({cv['folds']}-fold)      : {cv['mean']} +/- {cv['std']}")
    else:
        print(f"  CV PR-AUC              : not available ({metrics['cross_validation']['reason']})")
    print()
    print("  top indicators (permutation importance on validation):")
    for row in metrics["feature_indicators"][:8]:
        print(f"      {row['feature']:<30} {row['importance_mean']:+.5f}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the churn baseline model.")
    parser.add_argument("--cutoff-week", type=int, default=config.DEFAULT_CUTOFF_WEEK)
    parser.add_argument("--target", default=config.DEFAULT_TARGET, choices=list(config.TARGETS))
    parser.add_argument("--no-tree", action="store_true", help="Skip the tree comparison.")
    parser.add_argument(
        "--max-queue-share",
        type=float,
        default=0.25,
        help="Capacity constraint used to choose the threshold on validation.",
    )
    args = parser.parse_args(argv)

    metrics = train(
        cutoff_week=args.cutoff_week,
        target_key=args.target,
        compare_tree=not args.no_tree,
        max_queue_share=args.max_queue_share,
    )
    print_report(metrics)
    for label, key in (("model", "model"), ("metrics", "model_metrics"), ("queue", "risk_queue")):
        print(f"\n  {label:<8}-> {rel(config.artifact_path(key))}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
