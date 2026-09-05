import numpy as np

from neuromethyl_ont.evaluation.classification import (
    depth_curve_metrics,
    expected_calibration_error,
    score_threshold_metrics,
    softmax_probs,
    top_k_accuracy,
)


def test_softmax_probs_rows_sum_to_one():
    logits = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    probs = softmax_probs(logits)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])


def test_top_k_accuracy_perfect_top1():
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    y_true = np.array([0, 1])
    assert top_k_accuracy(probs, y_true, k=1) == 1.0


def test_top_k_accuracy_recovers_via_top3():
    probs = np.array([[0.1, 0.2, 0.7]])  # true class 0 is only 3rd-ranked
    y_true = np.array([0])
    assert top_k_accuracy(probs, y_true, k=1) == 0.0
    assert top_k_accuracy(probs, y_true, k=3) == 1.0


def test_score_threshold_metrics_fraction_and_accuracy():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    y_true = np.array([0, 0, 1])
    out = score_threshold_metrics(probs, y_true, thresholds=(0.8,))
    assert out["fraction_score_ge_0.8"] == 1 / 3
    assert out["accuracy_among_score_ge_0.8"] == 1.0


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    probs = np.array([[1.0, 0.0]] * 10)
    y_true = np.array([0] * 10)
    assert expected_calibration_error(probs, y_true) == 0.0


def test_depth_curve_metrics_reports_the_full_section14_set():
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    y_true = np.array([0, 1, 2])

    metrics = depth_curve_metrics(logits, y_true)

    for key in (
        "accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "nll", "ece",
        "fraction_score_ge_0.8", "accuracy_among_score_ge_0.8",
        "fraction_score_ge_0.95", "accuracy_among_score_ge_0.95",
    ):
        assert key in metrics
    assert metrics["accuracy"] == 1.0
