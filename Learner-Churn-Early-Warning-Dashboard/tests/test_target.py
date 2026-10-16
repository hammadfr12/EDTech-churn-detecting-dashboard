"""Target construction: the positive/negative/excluded split must be exact."""

from __future__ import annotations

import pandas as pd
import pytest

from src.model.target import TargetSpec, available_targets, class_balance, label_frame


@pytest.fixture()
def outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code_module": ["AAA"] * 4,
            "code_presentation": ["2013J"] * 4,
            "id_student": [1, 2, 3, 4],
            "final_result": ["Pass", "Withdrawn", "Distinction", "Fail"],
        }
    )


def test_primary_target_excludes_fail(outcomes):
    labelled = label_frame(outcomes, "primary")
    assert set(labelled["final_result"]) == {"Pass", "Withdrawn", "Distinction"}
    assert "Fail" not in set(labelled["final_result"])
    assert labelled.attrs["excluded_counts"] == {"Fail": 1}


def test_primary_target_encoding(outcomes):
    labelled = label_frame(outcomes, "primary").set_index("id_student")
    assert labelled.loc[2, "target"] == 1     # Withdrawn -> positive
    assert labelled.loc[1, "target"] == 0     # Pass -> negative
    assert labelled.loc[3, "target"] == 0     # Distinction -> negative


def test_sensitivity_target_counts_fail_as_churn(outcomes):
    labelled = label_frame(outcomes, "fail_as_churn").set_index("id_student")
    assert len(labelled) == 4
    assert labelled.loc[2, "target"] == 1     # Withdrawn
    assert labelled.loc[4, "target"] == 1     # Fail now counts
    assert labelled.loc[1, "target"] == 0
    assert labelled.loc[3, "target"] == 0


def test_target_never_silently_coerces_unknown_labels():
    df = pd.DataFrame(
        {
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "id_student": [9],
            "final_result": ["Deferred"],
        }
    )
    labelled = label_frame(df, "primary")
    assert labelled.empty
    assert labelled.attrs["excluded_counts"] == {"Deferred": 1}


def test_unknown_target_key_raises():
    with pytest.raises(KeyError):
        TargetSpec.from_config("does_not_exist")


def test_both_targets_are_registered():
    assert set(available_targets()) == {"primary", "fail_as_churn"}


def test_class_balance_reports_prevalence_and_baseline(outcomes):
    labelled = label_frame(outcomes, "fail_as_churn")   # 2 pos, 2 neg
    balance = class_balance(labelled)
    assert balance["n"] == 4
    assert balance["n_positive"] == 2
    assert balance["prevalence"] == 0.5
    assert balance["majority_class_baseline_accuracy"] == 0.5


def test_class_balance_handles_empty_frame(outcomes):
    empty = label_frame(outcomes.iloc[0:0], "primary")
    balance = class_balance(empty)
    assert balance["n"] == 0
    assert balance["prevalence"] is None
    assert balance["majority_class_baseline_accuracy"] is None
