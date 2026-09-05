import numpy as np
import pytest

from neuromethyl_ont.data.idat_reconstruction import compare_beta_matrices, load_mnp_preprocessed


def _write_f8(path, matrix_probes_by_samples: np.ndarray):
    # Mirrors idat_to_beta_mnp.R: column-major flatten of an (n_probes, n_samples) matrix.
    path.write_bytes(matrix_probes_by_samples.astype("<f8").tobytes(order="F"))


def test_load_mnp_preprocessed_roundtrips_rs_column_major_layout(tmp_path):
    probe_ids = ["cg1", "cg2", "cg3"]
    sample_ids = ["GSM1", "GSM2"]
    # R matrix, probes x samples: rows=probes, cols=samples.
    meth = np.array([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]])
    unmeth = meth + 1.0
    beta = meth / (meth + unmeth + 100)
    detection_p = np.zeros_like(meth)

    (tmp_path / "probe_ids.txt").write_text("\n".join(probe_ids) + "\n")
    (tmp_path / "sample_ids.txt").write_text("\n".join(sample_ids) + "\n")
    _write_f8(tmp_path / "meth.f8", meth)
    _write_f8(tmp_path / "unmeth.f8", unmeth)
    _write_f8(tmp_path / "beta.f8", beta)
    _write_f8(tmp_path / "detection_p.f8", detection_p)

    loaded = load_mnp_preprocessed(tmp_path)

    assert loaded.probe_ids == probe_ids
    assert loaded.sample_ids == sample_ids
    assert loaded.meth.shape == (2, 3)  # (n_samples, n_probes)
    # loaded.meth[sample, probe] must equal meth[probe, sample]
    np.testing.assert_allclose(loaded.meth, meth.T)
    np.testing.assert_allclose(loaded.unmeth, unmeth.T)
    np.testing.assert_allclose(loaded.beta, beta.T)


def test_load_mnp_preprocessed_rejects_wrong_element_count(tmp_path):
    (tmp_path / "probe_ids.txt").write_text("cg1\ncg2\n")
    (tmp_path / "sample_ids.txt").write_text("GSM1\n")
    (tmp_path / "meth.f8").write_bytes(np.array([1.0], dtype="<f8").tobytes())  # only 1, need 2
    (tmp_path / "unmeth.f8").write_bytes(np.zeros(2, dtype="<f8").tobytes())
    (tmp_path / "beta.f8").write_bytes(np.zeros(2, dtype="<f8").tobytes())
    (tmp_path / "detection_p.f8").write_bytes(np.zeros(2, dtype="<f8").tobytes())

    with pytest.raises(ValueError, match="expected 2 values"):
        load_mnp_preprocessed(tmp_path)


def test_compare_beta_matrices_perfect_agreement():
    beta = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]])
    probe_ids = ["cg1", "cg2", "cg3"]

    result = compare_beta_matrices(beta, probe_ids, beta, probe_ids)

    assert result["n_common_probes"] == 3
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["median_abs_diff"] == 0.0
    assert result["max_abs_diff"] == 0.0
    assert result["fraction_within_0_05"] == 1.0


def test_compare_beta_matrices_uses_only_common_probes_and_ignores_nan():
    reconstructed = np.array([[0.1, 0.5, 0.9]])
    reconstructed_probes = ["cg1", "cg2", "cgOnlyInReconstructed"]
    reference = np.array([[0.12, np.nan, 0.30]])
    reference_probes = ["cg1", "cg2", "cgOnlyInReference"]

    result = compare_beta_matrices(reconstructed, reconstructed_probes, reference, reference_probes)

    # Common probes: cg1, cg2. cg2's reference value is NaN -> excluded from comparison.
    assert result["n_common_probes"] == 2
    assert result["n_compared_values"] == 1
    assert result["max_abs_diff"] == pytest.approx(0.02, abs=1e-9)


def test_compare_beta_matrices_rejects_sample_count_mismatch():
    with pytest.raises(ValueError, match="sample count mismatch"):
        compare_beta_matrices(
            np.zeros((2, 3)), ["cg1", "cg2", "cg3"], np.zeros((3, 3)), ["cg1", "cg2", "cg3"]
        )
