from __future__ import annotations

from collections.abc import Mapping

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> Mapping[str, float]:
    """Core metrics reported across classification experiments."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
