"""Held-out evaluation metrics.

Every metric is computed on held-out data. If a metric cannot be computed
(single-class held-out fold, no positive predictions) the function returns
``None`` with a stated reason rather than a fabricated number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.target import class_balance


def _safe(value: float) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return None if (np.isnan(value) or np.isinf(value)) else round(value, 6)


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Precision, recall, F1 and the confusion matrix at one threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    notes: List[str] = []
    if tp + fp == 0:
        notes.append("No learners flagged at this threshold, so precision is undefined.")
    if tp + fn == 0:
        notes.append("No positive cases in the evaluation set, so recall is undefined.")

    return {
        "threshold": round(float(threshold), 6),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": _safe(precision_score(y_true, y_pred, zero_division=np.nan)),
        "recall": _safe(recall_score(y_true, y_pred, zero_division=np.nan)),
        "f1": _safe(f1_score(y_true, y_pred, zero_division=np.nan)),
        "flagged": int(y_pred.sum()),
        "flagged_share": _safe(y_pred.mean()),
        "notes": notes,
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """Threshold-free ranking quality."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    notes: List[str] = []

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "brier": None,
            "notes": ["Held-out set contains a single class; ranking metrics are undefined."],
        }

    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    return {
        "roc_auc": _safe(roc),
        "pr_auc": _safe(pr),
        "brier": _safe(brier),
        "notes": notes,
    }


def threshold_curve(
    y_true: np.ndarray, y_score: np.ndarray, max_points: int = 200
) -> List[Dict[str, Any]]:
    """Precision/recall/queue-size trade-off across candidate thresholds.

    This is what the operating threshold is chosen from: the mentor-capacity
    trade-off is visible as queue size against recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return []

    order = np.argsort(thresholds)
    thresholds, precision, recall = thresholds[order], precision[order], recall[order]

    if len(thresholds) > max_points:
        idx = np.linspace(0, len(thresholds) - 1, max_points).astype(int)
        thresholds, precision, recall = thresholds[idx], precision[idx], recall[idx]

    n = len(y_true)
    n_pos = int(np.sum(y_true))
    curve: List[Dict[str, Any]] = []
    for t, p, r in zip(thresholds, precision, recall):
        flagged = int((y_score >= t).sum())
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        curve.append(
            {
                "threshold": round(float(t), 6),
                "precision": _safe(p),
                "recall": _safe(r),
                "f1": _safe(f1),
                "flagged": flagged,
                "flagged_share": _safe(flagged / n) if n else None,
                "true_positives": int(round(r * n_pos)),
            }
        )
    return curve


def choose_threshold(
    curve: List[Dict[str, Any]], max_queue_share: float = 0.25
) -> Dict[str, Any]:
    """Pick an operating threshold under a mentor-capacity constraint.

    The rule is explicit and deliberately not "maximise accuracy":

        Among thresholds that flag at most ``max_queue_share`` of the cohort,
        choose the one with the highest F1.

    The constraint is the mentor-capacity story. The objective is F1 because
    with an imbalanced target, precision and recall both matter. The threshold
    is chosen on a validation split, never on the final held-out test set.
    """
    candidates = [
        row
        for row in curve
        if row["flagged_share"] is not None
        and row["flagged_share"] <= max_queue_share
        and row["f1"] is not None
    ]
    if not candidates:
        # Fall back to the highest-F1 point if the capacity constraint is
        # infeasible, and say so.
        candidates = [row for row in curve if row["f1"] is not None]
        if not candidates:
            return {"threshold": 0.5, "rule": "default", "reason": "No usable curve."}
        best = max(candidates, key=lambda r: r["f1"])
        return {
            "threshold": best["threshold"],
            "rule": "max_f1_fallback",
            "reason": (
                f"No threshold flagged <= {max_queue_share:.0%} of the cohort; "
                "fell back to the highest-F1 point on validation."
            ),
            "validation_f1": best["f1"],
            "validation_recall": best["recall"],
            "validation_precision": best["precision"],
            "validation_queue_share": best["flagged_share"],
        }

    best = max(candidates, key=lambda r: r["f1"])
    return {
        "threshold": best["threshold"],
        "rule": "max_f1_under_capacity",
        "reason": (
            f"Highest F1 among thresholds flagging at most {max_queue_share:.0%} "
            "of the cohort on the validation split."
        ),
        "validation_f1": best["f1"],
        "validation_recall": best["recall"],
        "validation_precision": best["precision"],
        "validation_queue_share": best["flagged_share"],
    }


def full_evaluation(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Everything the model card and dashboard page need, in one dict."""
    return {
        "class_balance": class_balance(
            # Rebuild a frame-shaped input for the shared helper
            _as_label_frame(y_true)
        ),
        "ranking": ranking_metrics(y_true, y_score),
        "at_threshold": metrics_at_threshold(y_true, y_score, threshold),
    }


def _as_label_frame(y_true: np.ndarray):
    import pandas as pd

    return pd.DataFrame({"target": np.asarray(y_true)})
