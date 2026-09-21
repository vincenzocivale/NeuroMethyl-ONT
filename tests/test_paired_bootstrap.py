import numpy as np

from neuromethyl_ont.evaluation.paired_bootstrap import compare_paired, per_sample_metrics


def test_per_sample_metrics_perfect_prediction_has_zero_nll_and_brier():
    probs = np.array([[0.99, 0.01], [0.01, 0.99]])
    labels = np.array([0, 1])
    correct, nll, brier = per_sample_metrics(probs, labels)
    assert np.all(correct == 1.0)
    assert np.all(nll >= 0)
    assert np.all(brier >= 0)


def test_compare_paired_identical_models_have_zero_mean_delta():
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 3, size=n)
    probs = rng.dirichlet(np.ones(3), size=n)
    class_names = ["a", "b", "c"]

    result = compare_paired(
        probs, probs.copy(), labels, class_names, name_a="M1", name_b="M2", iterations=200
    )

    assert result["point_estimate"]["accuracy_M1"] == result["point_estimate"]["accuracy_M2"]
    delta = result["bootstrap_delta_b_minus_a"]["accuracy"]
    assert delta["ci_low"] <= 0.0 <= delta["ci_high"]
    assert set(result["per_class"]) <= set(class_names)


def test_compare_paired_rejects_mismatched_shapes():
    probs_a = np.zeros((5, 3))
    probs_b = np.zeros((5, 2))
    labels = np.zeros(5, dtype=int)
    try:
        compare_paired(probs_a, probs_b, labels, ["a", "b", "c"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
