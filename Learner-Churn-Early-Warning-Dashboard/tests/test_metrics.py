"""Metric computation: correctness, edge cases, and honest 'not available'."""

from __future__ import annotations

import numpy as np
import pytest

from src.model import evaluate


def test_perfect_separation_gives_auc_of_one():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    rank = evaluate.ranking_metrics(y, scores)
    assert rank["roc_auc"] == 1.0
    assert rank["pr_auc"] == 1.0


def test_inverted_scores_give_auc_below_half():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert evaluate.ranking_metrics(y, scores)["roc_auc"] < 0.5


def test_single_class_returns_none_with_a_reason():
    y = np.array([1, 1, 1])
    scores = np.array([0.2, 0.5, 0.9])
    rank = evaluate.ranking_metrics(y, scores)
    assert rank["roc_auc"] is None
    assert rank["pr_auc"] is None
    assert rank["notes"], "a metric that cannot be computed must carry a reason"


def test_metrics_at_threshold_counts_are_consistent():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.6, 0.9])
    m = evaluate.metrics_at_threshold(y, scores, 0.5)
    assert m["true_positive"] == 2
    assert m["false_positive"] == 0
    assert m["true_negative"] == 2
    assert m["false_negative"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["true_positive"] + m["false_positive"] + m["true_negative"] + m["false_negative"] == len(y)


def test_threshold_of_zero_flags_everyone():
    y = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    m = evaluate.metrics_at_threshold(y, scores, 0.0)
    assert m["flagged"] == 2
    assert m["recall"] == 1.0


def test_threshold_above_one_flags_nobody():
    y = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    m = evaluate.metrics_at_threshold(y, scores, 1.01)
    assert m["flagged"] == 0
    assert m["recall"] == 0.0
    assert m["notes"], "an undefined precision must be explained, not silently zero"


def test_threshold_curve_is_monotonic_in_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    scores = rng.random(400)
    curve = evaluate.threshold_curve(y, scores)
    thresholds = [row["threshold"] for row in curve]
    assert thresholds == sorted(thresholds)
    flagged = [row["flagged"] for row in curve]
    assert flagged == sorted(flagged, reverse=True), "queue size must shrink as threshold rises"


def test_choose_threshold_respects_the_capacity_constraint():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    scores = rng.random(800)
    curve = evaluate.threshold_curve(y, scores)
    choice = evaluate.choose_threshold(curve, max_queue_share=0.20)
    assert choice["rule"] in {"max_f1_under_capacity", "max_f1_fallback"}
    if choice["rule"] == "max_f1_under_capacity":
        assert choice["validation_queue_share"] <= 0.20


def test_choose_threshold_falls_back_and_says_so():
    """An infeasible constraint must be reported, not hidden."""
    y = np.array([1, 1, 1, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    curve = evaluate.threshold_curve(y, scores)
    choice = evaluate.choose_threshold(curve, max_queue_share=0.01)
    assert choice["rule"] in {"max_f1_fallback", "default"}


def test_brier_score_is_a_proper_probability_metric():
    y = np.array([0, 1])
    good = evaluate.ranking_metrics(y, np.array([0.01, 0.99]))
    bad = evaluate.ranking_metrics(y, np.array([0.49, 0.51]))
    assert good["brier"] < bad["brier"]


def test_full_evaluation_shape():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.7, 0.3, 0.8])
    out = evaluate.full_evaluation(y, scores, 0.5)
    assert set(out) == {"class_balance", "ranking", "at_threshold"}
    assert out["class_balance"]["n"] == 4


def test_metrics_are_json_serialisable():
    """Every metric goes into a JSON artifact, so no numpy scalars allowed."""
    import json

    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.7, 0.3, 0.8])
    out = evaluate.full_evaluation(y, scores, 0.5)
    json.dumps(out)  # must not raise
