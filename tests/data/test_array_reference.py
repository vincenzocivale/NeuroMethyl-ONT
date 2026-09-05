import numpy as np

from neuromethyl_ont.data.array_reference import (
    align_to_feature_index,
    build_array_reference,
    load_array_reference,
    save_array_reference,
    verify_feature_index_checksum,
)


def _toy_beta():
    # 4 samples x 5 probes:
    #   cgA: varies, fully observed -> kept
    #   cgB: constant 0.5 -> dropped (zero variance)
    #   cgC: mostly missing -> dropped (insufficient valid_fraction)
    #   cgD: duplicate probe_id of cgA's slot -> second occurrence dropped
    #   cgA (dup): identical column, same probe_id as column 0
    beta = np.array(
        [
            [0.1, 0.5, np.nan, 0.9, 0.1],
            [0.2, 0.5, np.nan, 0.9, 0.2],
            [0.3, 0.5, np.nan, 0.9, 0.3],
            [0.4, 0.5, 0.4, 0.9, 0.4],
        ]
    )
    probe_ids = ["cgA", "cgB", "cgC", "cgA", "cgA"]
    sample_ids = ["s1", "s2", "s3", "s4"]
    labels = ["classX", "classX", "classY", "classY"]
    return beta, sample_ids, probe_ids, labels


def test_build_array_reference_drops_constant_missing_and_duplicate_probes():
    beta, sample_ids, probe_ids, labels = _toy_beta()

    ref = build_array_reference(beta, sample_ids, probe_ids, labels, min_valid_fraction=0.95)

    assert ref.probe_ids == ["cgA"]
    assert ref.beta.shape == (4, 1)
    np.testing.assert_allclose(ref.beta[:, 0], [0.1, 0.2, 0.3, 0.4])
    assert ref.sample_ids == sample_ids
    assert ref.labels == labels


def test_save_and_load_array_reference_roundtrip(tmp_path):
    beta, sample_ids, probe_ids, labels = _toy_beta()
    ref = build_array_reference(beta, sample_ids, probe_ids, labels, min_valid_fraction=0.95)

    feature_index = save_array_reference(ref, tmp_path, source_dataset="GSE90496")
    assert verify_feature_index_checksum(feature_index)
    assert feature_index.probe_ids == ["cgA"]

    loaded = load_array_reference(tmp_path)
    np.testing.assert_allclose(loaded.beta, ref.beta)
    assert loaded.probe_ids == ref.probe_ids
    assert loaded.sample_ids == ref.sample_ids
    assert loaded.labels == ref.labels


def test_align_to_feature_index_fills_missing_probes_with_nan():
    from neuromethyl_ont.data.array_reference import FeatureIndex

    feature_index = FeatureIndex(
        probe_ids=["cgA", "cgZ"], checksum="ignored", source_dataset="GSE90496", created_at="now"
    )
    beta = np.array([[0.7, 0.3]])  # columns: cgA, cgOther
    aligned = align_to_feature_index(
        beta, probe_ids=["cgA", "cgOther"], feature_index=feature_index
    )

    assert aligned.shape == (1, 2)
    assert aligned[0, 0] == 0.7
    assert np.isnan(aligned[0, 1])  # cgZ absent from source -> missing
