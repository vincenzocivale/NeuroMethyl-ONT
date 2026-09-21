import numpy as np
import torch

from neuromethyl_ont.evaluation.explainability import explain_prediction, verify_decomposition
from neuromethyl_ont.models.evidence_linear import build_linear_classifier


def _toy_model(n_features: int = 5, n_classes: int = 3) -> torch.nn.Module:
    torch.manual_seed(0)
    model = build_linear_classifier(n_features=n_features, n_classes=n_classes)
    return model


def test_decomposition_matches_forward_logit():
    model = _toy_model()
    x = np.random.default_rng(0).normal(size=5)
    for class_idx in range(3):
        assert verify_decomposition(model, x, class_idx) < 1e-5


def test_explain_prediction_reports_top_classes_and_contributions():
    model = _toy_model()
    x = np.random.default_rng(1).normal(size=5)
    probe_ids = [f"cg{i}" for i in range(5)]
    class_names = ["classA", "classB", "classC"]

    report = explain_prediction(
        model, x, probe_ids, class_names, sample_id="s1", top_n=3, n_runner_up=1
    )

    assert report.predicted_class in class_names
    assert len(report.top_classes) == 2
    assert report.predicted_class == report.top_classes[0]
    assert set(report.contributions_by_class) == set(report.top_classes)
    for contributions in report.contributions_by_class.values():
        assert len(contributions) == 3
        assert all(c.probe_id in probe_ids for c in contributions)
