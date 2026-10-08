"""Paired stratified-bootstrap significance testing between two models' predictions.

Pure-numpy statistical core with no dataset- or model-specific coupling.

None of NeuroMethyl-ONT's existing comparison scripts
(``scripts/evaluation/evaluate_rapidcns2.py``,
``evaluate_magibu_pools.py``, ...) currently report whether a point-estimate
accuracy gap (e.g. B0 vs crossNN) is statistically significant given the
sample size -- this module adds that, operating directly on two models'
per-sample class probabilities (already computed, however they were
produced), rather than re-running inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def per_sample_metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (correct, negative_log_likelihood, brier_score) per sample."""
    n = len(labels)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    p_true = probabilities[np.arange(n), labels].clip(1e-12, 1.0)
    nll = -np.log(p_true)
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(n), labels] = 1.0
    brier = np.sum((probabilities - onehot) ** 2, axis=1)
    return correct, nll, brier


def _stratified_resample_indices(
    labels: np.ndarray, num_classes: int, rng: np.random.Generator
) -> np.ndarray:
    by_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
    parts = [rng.choice(idx, size=len(idx), replace=True) for idx in by_class if len(idx) > 0]
    return np.concatenate(parts)


def _bootstrap_delta(
    labels: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    metric_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    metric_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_classes: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified-by-class bootstrap over accuracy/NLL/Brier/macro-F1 deltas
    (B - A), keeping the two models' predictions paired per resampled sample."""
    rng = np.random.default_rng(seed)
    accuracy_deltas = np.empty(iterations)
    nll_deltas = np.empty(iterations)
    brier_deltas = np.empty(iterations)
    macro_f1_deltas = np.empty(iterations)
    correct_a, nll_a, brier_a = metric_a
    correct_b, nll_b, brier_b = metric_b

    for i in range(iterations):
        resampled = _stratified_resample_indices(labels, num_classes, rng)
        accuracy_deltas[i] = correct_b[resampled].mean() - correct_a[resampled].mean()
        nll_deltas[i] = nll_b[resampled].mean() - nll_a[resampled].mean()
        brier_deltas[i] = brier_b[resampled].mean() - brier_a[resampled].mean()
        f1_a = f1_score(labels[resampled], predicted_a[resampled], average="macro", zero_division=0)
        f1_b = f1_score(labels[resampled], predicted_b[resampled], average="macro", zero_division=0)
        macro_f1_deltas[i] = f1_b - f1_a

    def _summary(deltas: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {"mean": float(deltas.mean()), "ci_low": float(lo), "ci_high": float(hi)}

    return {
        "accuracy": _summary(accuracy_deltas),
        "negative_log_likelihood": _summary(nll_deltas),
        "brier_score": _summary(brier_deltas),
        "macro_f1": _summary(macro_f1_deltas),
    }


def _win_tie_loss(
    metric_a: np.ndarray, metric_b: np.ndarray, *, lower_is_better: bool
) -> dict[str, int]:
    diff = (metric_a - metric_b) if lower_is_better else (metric_b - metric_a)
    return {
        "b_wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "b_losses": int((diff < 0).sum()),
    }


def compare_paired(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    name_a: str = "A",
    name_b: str = "B",
    iterations: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compare two models' predictions on the same held-out samples.

    ``probabilities_a``/``probabilities_b`` are ``(n_samples, n_classes)``
    per-sample class probabilities from the two models being compared, and
    ``labels`` is the shared ground truth (same sample order for both
    models -- caller is responsible for that alignment). Returns
    point estimates, a stratified paired bootstrap over the delta (B - A)
    with 95% CIs, per-sample win/tie/loss counts, and per-class accuracy.
    """
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError(
            f"probabilities_a shape {probabilities_a.shape} != "
            f"probabilities_b shape {probabilities_b.shape}"
        )
    if len(labels) != probabilities_a.shape[0]:
        raise ValueError("labels length must match the number of samples")

    num_classes = len(class_names)
    correct_a, nll_a, brier_a = per_sample_metrics(probabilities_a, labels)
    correct_b, nll_b, brier_b = per_sample_metrics(probabilities_b, labels)
    predicted_a = probabilities_a.argmax(axis=1)
    predicted_b = probabilities_b.argmax(axis=1)

    bootstrap = _bootstrap_delta(
        labels,
        predicted_a,
        predicted_b,
        (correct_a, nll_a, brier_a),
        (correct_b, nll_b, brier_b),
        num_classes,
        iterations,
        bootstrap_seed,
    )

    win_tie_loss = {
        "accuracy": _win_tie_loss(correct_a, correct_b, lower_is_better=False),
        "negative_log_likelihood": _win_tie_loss(nll_a, nll_b, lower_is_better=True),
        "brier_score": _win_tie_loss(brier_a, brier_b, lower_is_better=True),
    }

    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not np.any(mask):
            continue
        per_class[class_name] = {
            "n": int(mask.sum()),
            f"accuracy_{name_a}": float(correct_a[mask].mean()),
            f"accuracy_{name_b}": float(correct_b[mask].mean()),
        }

    return {
        "model_a": name_a,
        "model_b": name_b,
        "num_samples": int(len(labels)),
        "point_estimate": {
            f"accuracy_{name_a}": float(correct_a.mean()),
            f"accuracy_{name_b}": float(correct_b.mean()),
            f"macro_f1_{name_a}": float(
                f1_score(labels, predicted_a, average="macro", zero_division=0)
            ),
            f"macro_f1_{name_b}": float(
                f1_score(labels, predicted_b, average="macro", zero_division=0)
            ),
            f"nll_{name_a}": float(nll_a.mean()),
            f"nll_{name_b}": float(nll_b.mean()),
        },
        "bootstrap_delta_b_minus_a": bootstrap,
        "win_tie_loss_b_vs_a": win_tie_loss,
        "per_class": per_class,
    }
