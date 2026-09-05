import numpy as np

from neuromethyl_ont.evaluation.depth_curve import (
    DEPTH_GRID,
    build_fixed_depth_realizations,
    run_depth_curve,
)
from neuromethyl_ont.models.evidence_linear import build_linear_classifier
from neuromethyl_ont.training.trainer import TrainingConfig, train_linear_classifier


def _toy_dataset(n_per_class: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    beta_rows, y_rows = [], []
    for c in range(3):
        center = np.full(n_features, 0.15 + 0.3 * c)
        noise = rng.normal(0, 0.02, size=(n_per_class, n_features))
        beta_rows.append(np.clip(center + noise, 0, 1))
        y_rows.append(np.full(n_per_class, c))
    return np.concatenate(beta_rows, axis=0), np.concatenate(y_rows, axis=0)


def test_build_fixed_depth_realizations_is_deterministic_and_covers_full_grid():
    beta, _ = _toy_dataset(n_per_class=4, n_features=5, seed=0)

    realizations_a = build_fixed_depth_realizations(beta, seed=42)
    realizations_b = build_fixed_depth_realizations(beta, seed=42)

    assert set(realizations_a.keys()) == set(DEPTH_GRID)
    for depth in DEPTH_GRID:
        np.testing.assert_array_equal(realizations_a[depth].m, realizations_b[depth].m)
        np.testing.assert_array_equal(realizations_a[depth].n, realizations_b[depth].n)


def test_run_depth_curve_produces_one_row_per_model_per_depth():
    beta_train, y_train = _toy_dataset(n_per_class=20, n_features=6, seed=0)
    beta_val, y_val = _toy_dataset(n_per_class=6, n_features=6, seed=1)
    mu = beta_train.mean(axis=0)

    models, representations = {}, {}
    for name, representation in (("B0", "binary"), ("C0", "continuous"), ("E1", "evidence")):
        model = build_linear_classifier(n_features=6, n_classes=3)
        config = TrainingConfig(
            representation=representation, max_epochs=5, early_stopping_patience=5,
            batch_size=8, mixed_precision="none", kappa=2.0, seed=17,
        )
        train_linear_classifier(model, beta_train, y_train, beta_val, y_val, mu, config)
        models[name] = model
        representations[name] = representation

    depth_curve = run_depth_curve(
        models, representations, beta_val, y_val, mu=mu, kappa=2.0, seed=0
    )

    assert len(depth_curve) == len(models) * len(DEPTH_GRID)
    assert set(depth_curve["model"]) == {"B0", "C0", "E1"}
    assert "accuracy" in depth_curve.columns
