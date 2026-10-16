"""Target construction.

The target is a documented *choice*, not a fact about the data. Both choices
live in :mod:`src.config` and both are evaluated, so the reader can see how
much the headline number depends on the definition.

    primary         Withdrawn = 1, Pass/Distinction = 0, Fail EXCLUDED
    fail_as_churn   Withdrawn or Fail = 1, Pass/Distinction = 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src import config


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    description: str
    positive_labels: tuple
    negative_labels: tuple
    excluded_labels: tuple

    @classmethod
    def from_config(cls, key: str) -> "TargetSpec":
        if key not in config.TARGETS:
            raise KeyError(
                f"Unknown target '{key}'. Available: {sorted(config.TARGETS)}"
            )
        raw = config.TARGETS[key]
        return cls(
            key=str(raw["key"]),
            label=str(raw["label"]),
            description=str(raw["description"]),
            positive_labels=tuple(raw["positive_labels"]),
            negative_labels=tuple(raw["negative_labels"]),
            excluded_labels=tuple(raw["excluded_labels"]),
        )


def available_targets() -> List[str]:
    return list(config.TARGETS)


def label_frame(df: pd.DataFrame, target_key: str) -> pd.DataFrame:
    """Return the modelling cohort with a binary ``target`` column.

    Rows whose outcome is neither in the positive nor the negative set are
    dropped and counted, never silently coerced.
    """
    spec = TargetSpec.from_config(target_key)
    out = df.copy()
    outcome = out["final_result"].astype(str)

    out["is_positive"] = outcome.isin(spec.positive_labels)
    out["is_negative"] = outcome.isin(spec.negative_labels)
    in_cohort = out["is_positive"] | out["is_negative"]

    dropped = out.loc[~in_cohort, "final_result"].value_counts().to_dict()
    out = out.loc[in_cohort].copy()
    out["target"] = out["is_positive"].astype(int)
    # Only JSON-serialisable metadata goes in attrs: these frames get written
    # to parquet, and pandas serialises attrs as JSON. The spec itself is
    # re-derived from its key by the caller.
    out.attrs["excluded_counts"] = {str(k): int(v) for k, v in dropped.items()}
    out.attrs["target_key"] = spec.key
    return out


def class_balance(df: pd.DataFrame) -> Dict[str, object]:
    """Prevalence plus the majority-class baseline accuracy.

    The majority-class baseline is the number any model must beat to be
    worth anything: predicting the most common class for everyone.
    """
    if df.empty:
        return {
            "n": 0,
            "n_positive": 0,
            "n_negative": 0,
            "prevalence": None,
            "majority_class_baseline_accuracy": None,
        }
    n = int(len(df))
    n_pos = int(df["target"].sum())
    n_neg = n - n_pos
    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": round(n_pos / n, 6),
        "majority_class_baseline_accuracy": round(max(n_pos, n_neg) / n, 6),
    }
