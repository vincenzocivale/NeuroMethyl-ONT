"""Classification metrics for the V1 representation ablation (docs/EVIDENCE_LINEAR_V1.md #14).

Extends :func:`neuromethyl_ont.evaluation.metrics.classification_metrics`
with the calibration and confidence-threshold metrics the depth curve needs:
top-3 accuracy, NLL, ECE, and accuracy/coverage at Sturgeon's General-model
operating thresholds (0.8, 0.95).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from neuromethyl_ont.evaluation.metrics import classification_metrics

DEFAULT_SCORE_THRESHOLDS = (0.8, 0.95)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true)
    top_k = np.argsort(-probs, axis=1)[:, :k]
    hit = (top_k == y_true[:, None]).any(axis=1)
    return float(hit.mean())


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true)
    true_probs = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.log(true_probs).mean())


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Standard top-label ECE: bin by max predicted probability, compare mean
    confidence to observed accuracy in each bin, weight by bin occupancy.
    """
    y_true = np.asarray(y_true)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        above_lo = (confidence > lo) if lo > 0 else (confidence >= lo)
        in_bin = above_lo & (confidence <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidence[in_bin].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def score_threshold_metrics(
    probs: np.ndarray, y_true: np.ndarray, thresholds: tuple[float, ...] = DEFAULT_SCORE_THRESHOLDS
) -> Mapping[str, float]:
    y_true = np.asarray(y_true)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == y_true)

    out: dict[str, float] = {}
    for threshold in thresholds:
        kept = confidence >= threshold
        out[f"fraction_score_ge_{threshold}"] = float(kept.mean())
        out[f"accuracy_among_score_ge_{threshold}"] = (
            float(correct[kept].mean()) if kept.any() else float("nan")
        )
    return out


def depth_curve_metrics(logits: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """The full section-14 metric set for one (representation, depth) cell."""
    probs = softmax_probs(logits)
    prediction = probs.argmax(axis=1)

    metrics = dict(classification_metrics(y_true, prediction))
    metrics["top3_accuracy"] = top_k_accuracy(probs, y_true, k=3)
    metrics["nll"] = negative_log_likelihood(probs, y_true)
    metrics["ece"] = expected_calibration_error(probs, y_true)
    metrics.update(score_threshold_metrics(probs, y_true))
    return metrics
