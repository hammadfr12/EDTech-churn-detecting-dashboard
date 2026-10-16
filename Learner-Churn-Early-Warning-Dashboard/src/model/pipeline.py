"""Model definition and the leakage-safe split.

Split strategy
--------------
The group key is the learner-course presentation. A learner who appears in
two presentations is a different learning instance in each, so the same
instance must not straddle the split. :class:`GroupShuffleSplit` enforces
that. Preprocessing is fitted inside the sklearn pipeline, so imputers and
encoders only ever see training folds.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.model.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def group_key(df: pd.DataFrame) -> pd.Series:
    """The split group: one learning instance = module + presentation + learner."""
    cols = [c for c in config.GROUP_KEY_COLUMNS if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1)


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals.

    Numeric imputation uses the median and adds an explicit missing indicator,
    because "score was never observed" is information the model should be able
    to use rather than a value to be smoothed away.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_logistic_pipeline() -> Pipeline:
    """The baseline model: L2 logistic regression on a reproducible pipeline.

    ``class_weight='balanced'`` is deliberate. With ~31% prevalence the
    unweighted fit predicts the majority class far too readily; balancing
    makes the model produce usable probabilities across the range, and the
    operating threshold then controls queue size explicitly.
    """
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_tree_pipeline(max_depth: int = 4) -> Pipeline:
    """Optional shallow tree, used only as a comparison."""
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def group_train_test_split(
    df: pd.DataFrame, test_size: float | None = None, seed: int | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by learning instance so no instance appears on both sides."""
    test_size = test_size or config.TEST_SIZE
    seed = config.RANDOM_STATE if seed is None else seed

    groups = group_key(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    overlap = set(group_key(train)) & set(group_key(test))
    assert not overlap, f"Split leaked {len(overlap)} learning instances across partitions."
    return train, test


def group_train_validation_split(
    df: pd.DataFrame, test_size: float = 0.25, seed: int | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training partition again, for threshold selection only."""
    return group_train_test_split(df, test_size=test_size, seed=(seed or config.RANDOM_STATE) + 1)


def cross_validate_pr_auc(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int = 5
) -> Dict[str, object]:
    """Group-aware CV of PR-AUC, the metric that suits an imbalanced label.

    Returns a dict with the fold scores and their mean, or an explanation if
    the cross-validation could not run.
    """
    n_groups = groups.nunique()
    if n_groups < folds:
        return {
            "available": False,
            "reason": f"Only {n_groups} learning instances; {folds}-fold CV is not meaningful.",
            "scores": [],
            "mean": None,
        }
    try:
        scores = cross_val_score(
            pipeline,
            X,
            y,
            groups=groups,
            cv=GroupKFold(n_splits=folds),
            scoring="average_precision",
            n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": str(exc), "scores": [], "mean": None}

    return {
        "available": True,
        "reason": None,
        "scores": [round(float(s), 6) for s in scores],
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
        "folds": folds,
    }


def permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    seed: int | None = None,
    top_n: int = 15,
) -> List[Dict[str, object]]:
    """Permutation importance on the raw feature columns.

    Permutation importance is used instead of logistic coefficients because
    the one-hot expansion makes raw coefficients hard to read back to a
    business feature. The result is a ranking of *predictive contribution*,
    which the UI presents as "indicators", never as causes.
    """
    from sklearn.inspection import permutation_importance as sk_permutation_importance

    seed = config.RANDOM_STATE if seed is None else seed
    result = sk_permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="average_precision",
        n_jobs=1,
    )
    rows = [
        {
            "feature": str(name),
            "importance_mean": round(float(mean), 8),
            "importance_std": round(float(std), 8),
        }
        for name, mean, std in zip(X.columns, result.importances_mean, result.importances_std)
    ]
    rows.sort(key=lambda r: -r["importance_mean"])
    return rows[:top_n]
