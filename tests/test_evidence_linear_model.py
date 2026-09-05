import torch

from neuromethyl_ont.models.evidence_linear import LinearClassifier, build_linear_classifier


def test_linear_classifier_has_no_hidden_layer():
    model = build_linear_classifier(n_features=100, n_classes=91)

    assert isinstance(model, LinearClassifier)
    param_names = [name for name, _ in model.named_parameters()]
    # Exactly one Linear layer's weight and bias -- no hidden layers, no extra modules.
    assert sorted(param_names) == ["linear.bias", "linear.weight"]
    assert model.linear.weight.shape == (91, 100)
    assert model.linear.bias.shape == (91,)


def test_linear_classifier_forward_shape():
    model = build_linear_classifier(n_features=10, n_classes=91)
    x = torch.randn(4, 10)

    logits = model(x)

    assert logits.shape == (4, 91)
