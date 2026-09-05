import numpy as np

from neuromethyl_ont.models.evidence_linear import build_linear_classifier
from neuromethyl_ont.training.trainer import TrainingConfig, train_linear_classifier


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
