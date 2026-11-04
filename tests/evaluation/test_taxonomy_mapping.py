from pathlib import Path

import numpy as np
import pytest

from neuromethyl_ont.evaluation.taxonomy_mapping import coarse_truth, load_taxonomy, map_to_coarse

YAML = Path(__file__).resolve().parents[2] / "configs" / "taxonomies" / "methylyzr_who_level.yaml"
TAX = load_taxonomy(YAML)
CLASSES = [c["code"] for c in TAX["classes"]]


def test_every_one_of_the_91_classes_assigned_exactly_once():
    assert len(CLASSES) == 91 and len(set(CLASSES)) == 91
    assert {c["coarse"] for c in TAX["classes"]} <= set(TAX["coarse_labels"])


def test_mass_conserved_and_known_merges():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(91), size=20)
    out, labels = map_to_coarse(p, CLASSES, TAX)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    gbm = labels.index("GBM")
    cols = [CLASSES.index(c) for c in ("GBM, RTK I", "GBM, RTK II", "GBM, RTK III", "GBM, MES", "GBM, MYCN", "GBM, MID")]
    np.testing.assert_allclose(out[:, gbm], p[:, cols].sum(axis=1))


def test_column_order_of_input_does_not_matter():
    p = np.random.default_rng(1).dirichlet(np.ones(91), size=3)
    perm = np.random.default_rng(2).permutation(91)
    a, _ = map_to_coarse(p, CLASSES, TAX)
    b, _ = map_to_coarse(p[:, perm], [CLASSES[i] for i in perm], TAX)
    np.testing.assert_allclose(a, b)


def test_wrong_class_set_rejected():
    p = np.full((1, 90), 1 / 90)
    with pytest.raises(ValueError):
        map_to_coarse(p, CLASSES[:90], TAX)
    with pytest.raises(ValueError):
        map_to_coarse(np.full((1, 91), 1 / 91), CLASSES[:90] + [CLASSES[0]], TAX)


def test_metastasis_labels_are_out_of_scope_abstention_tests():
    for name in ("Lung cancer metastasis", "Breast cancer metastasis", "Melanoma metastasis",
                 "Colorectal cancer metastasis", "Endometrial cancer metastasis"):
        assert coarse_truth(name, TAX) == ("OUT_OF_SCOPE", True)
    assert coarse_truth("GBM", TAX) == ("GBM", False)
    assert TAX["out_of_scope_label"] not in TAX["coarse_labels"]


def test_duplicate_class_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "coarse_labels: [X]\nout_of_scope_label: O\ncohort_labels: {}\n"
        "classes: [{code: a, coarse: X}, {code: a, coarse: X}]\n"
    )
    with pytest.raises(ValueError):
        load_taxonomy(bad)
