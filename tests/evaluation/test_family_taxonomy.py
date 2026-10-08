import numpy as np
import pytest

from neuromethyl_ont.evaluation.family_taxonomy import (
    UNRESOLVED,
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)


def test_aggregate_to_family_sums_probability_mass_and_buckets_unresolved():
    probs = np.array([[0.5, 0.3, 0.2]])
    labels = ["classA", "classB", "classUnknown"]
    family_by_class = {"classA": "FAM1", "classB": "FAM1"}

    fam_probs, families = aggregate_to_family(probs, labels, family_by_class)

    assert families == ["FAM1", UNRESOLVED]
    np.testing.assert_allclose(fam_probs[0], [0.8, 0.2])


def test_aggregate_to_family_requires_probability_mass_preserved():
    probs = np.array([[0.5, 0.5]])
    labels = ["classA", "classB"]
    family_by_class = {"classA": "FAM1", "classB": "FAM2"}

    fam_probs, families = aggregate_to_family(probs, labels, family_by_class)

    np.testing.assert_allclose(fam_probs.sum(axis=1), [1.0])


def test_family_metrics_perfect_prediction():
    families = ["FAM1", "FAM2"]
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    true = np.array(["FAM1", "FAM2"])

    metrics = family_metrics(true, probs, families)

    assert metrics["family_accuracy"] == 1.0
    assert metrics["n_samples"] == 2


@pytest.mark.parametrize(
    "column,expected",
    [
        ("Control - INFLAM - REACTIVE", "CONTR, REACTIVE"),
        ("Glioblastoma - GBM - MES", "GBM, MES"),
        ("Glioblastoma - DMG - K27", "DMG, K27"),
    ],
)
def test_sturgeon_model_class_translates_known_examples(column, expected):
    assert sturgeon_model_class(column) == expected


def test_sturgeon_model_class_rejects_unknown_group():
    with pytest.raises(ValueError):
        sturgeon_model_class("NotAGroup - X - Y")
