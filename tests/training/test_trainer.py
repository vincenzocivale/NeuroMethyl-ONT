import numpy as np
import pytest
import torch

from neuromethyl_ont.models.evidence_linear import build_linear_classifier
from neuromethyl_ont.training.trainer import TrainingConfig, predict_logits, train_linear_classifier


def _separable_dataset(n_per_class: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    n_classes = 3
    beta_rows, y_rows = [], []
    for c in range(n_classes):
        center = np.full(n_features, 0.15 + 0.3 * c)
        noise = rng.normal(0, 0.02, size=(n_per_class, n_features))
        beta_rows.append(np.clip(center + noise, 0, 1))
        y_rows.append(np.full(n_per_class, c))
    beta = np.concatenate(beta_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    return beta, y


def test_train_linear_classifier_runs_and_tracks_best_epoch():
    beta_train, y_train = _separable_dataset(n_per_class=20, n_features=8, seed=0)
    beta_val, y_val = _separable_dataset(n_per_class=5, n_features=8, seed=1)
    mu = beta_train.mean(axis=0)

    model = build_linear_classifier(n_features=8, n_classes=3)
    config = TrainingConfig(
        representation="continuous",
        max_epochs=10,
        early_stopping_patience=10,
        batch_size=8,
        mixed_precision="none",
        seed=17,
    )

    result = train_linear_classifier(model, beta_train, y_train, beta_val, y_val, mu, config)

    assert len(result.history) > 0
    assert result.best_state_dict is not None
    assert 0.0 <= result.best_val_accuracy <= 1.0
    assert 0 <= result.best_epoch < len(result.history)
    best_recorded_accuracy = result.history[result.best_epoch].val_accuracy
    assert best_recorded_accuracy == result.best_val_accuracy


def test_train_linear_classifier_separates_easy_classes_well():
    beta_train, y_train = _separable_dataset(n_per_class=30, n_features=10, seed=0)
    beta_val, y_val = _separable_dataset(n_per_class=10, n_features=10, seed=1)
    mu = beta_train.mean(axis=0)

    model = build_linear_classifier(n_features=10, n_classes=3)
    # lr raised well above the spec's 3e-4 (docs/EVIDENCE_LINEAR_V1.md #11) so this
    # smoke test converges in a handful of epochs -- it checks the training loop
    # is correct, not that the production hyperparameters are fast to converge.
    config = TrainingConfig(
        representation="continuous",
        max_epochs=60,
        early_stopping_patience=60,
        batch_size=16,
        mixed_precision="none",
        lr=0.05,
        seed=17,
    )

    result = train_linear_classifier(model, beta_train, y_train, beta_val, y_val, mu, config)

    assert result.best_val_accuracy > 0.8


def test_train_linear_classifier_works_for_all_three_representations():
    beta_train, y_train = _separable_dataset(n_per_class=15, n_features=6, seed=0)
    beta_val, y_val = _separable_dataset(n_per_class=5, n_features=6, seed=1)
    mu = beta_train.mean(axis=0)

    for representation in ("binary", "continuous", "evidence"):
        model = build_linear_classifier(n_features=6, n_classes=3)
        config = TrainingConfig(
            representation=representation,
            max_epochs=5,
            early_stopping_patience=5,
            batch_size=8,
            mixed_precision="none",
            kappa=2.0,
            seed=17,
        )
        result = train_linear_classifier(model, beta_train, y_train, beta_val, y_val, mu, config)
        assert len(result.history) > 0


def test_train_linear_classifier_runs_fixed_epochs_without_validation():
    beta_train, y_train = _separable_dataset(n_per_class=15, n_features=6, seed=0)
    mu = beta_train.mean(axis=0)

    model = build_linear_classifier(n_features=6, n_classes=3)
    config = TrainingConfig(
        representation="continuous",
        max_epochs=7,
        early_stopping_patience=2,  # must be ignored: no validation signal to stop on
        batch_size=8,
        mixed_precision="none",
        seed=17,
    )

    result = train_linear_classifier(model, beta_train, y_train, None, None, mu, config)

    assert len(result.history) == 7  # ran every epoch, early stopping never triggers
    assert not result.stopped_early
    assert result.best_epoch == 6
    assert result.best_val_accuracy is None
    assert result.best_state_dict is not None
    assert all(r.val_loss is None and r.val_accuracy is None for r in result.history)


def test_train_linear_classifier_rejects_partial_validation_args():
    beta_train, y_train = _separable_dataset(n_per_class=5, n_features=4, seed=0)
    beta_val, y_val = _separable_dataset(n_per_class=5, n_features=4, seed=1)
    mu = beta_train.mean(axis=0)
    model = build_linear_classifier(n_features=4, n_classes=3)
    config = TrainingConfig(representation="continuous", max_epochs=1, mixed_precision="none")

    with pytest.raises(ValueError):
        train_linear_classifier(model, beta_train, y_train, beta_val, None, mu, config)


def test_predict_logits_matches_model_device_cpu():
    model = build_linear_classifier(n_features=6, n_classes=3)
    x = np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)

    logits = predict_logits(model, x)

    assert isinstance(logits, np.ndarray)
    assert logits.shape == (4, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_predict_logits_moves_cuda_model_output_back_to_cpu():
    """Regression test: a model trained with --device cuda must not crash when
    its full-array validation logits are converted to numpy (docs bugfix --
    scripts/training/train_classifier.py used to build the input tensor on
    the CPU and call ``.numpy()`` directly on a CUDA-resident output, which
    raises ``TypeError: can't convert cuda:0 device type tensor to numpy``).
    """
    model = build_linear_classifier(n_features=6, n_classes=3).to("cuda")
    x = np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)

    logits = predict_logits(model, x)

    assert isinstance(logits, np.ndarray)
    assert logits.shape == (4, 3)
