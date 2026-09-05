from neuromethyl_ont.evaluation.metrics import classification_metrics


def test_classification_metrics_perfect_prediction():
    metrics = classification_metrics([0, 1, 2], [0, 1, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
