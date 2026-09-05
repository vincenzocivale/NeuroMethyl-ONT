import gzip

import numpy as np
import pytest

from neuromethyl_ont.data.geo_series import (
    GeoSeriesParseError,
    parse_soft_family,
    read_geo_beta_matrix,
    read_geo_beta_matrix_tolerant,
)

# Minimal SOFT family fixture: one sample resolved via the explicit
# "!Sample_description = SAMPLE n" line (GSE109379-style), one via the
# "sample n" token inside the title only (GSE90496-style). Sample 1 also
# carries the Grn/Red IDAT supplementary-file lines.
SOFT_FIXTURE = """\
^SERIES = GSE00000
!Series_title = fixture series
^SAMPLE = GSM0000001
!Sample_title = GBM, MES, sample 1 [reference set]
!Sample_characteristics_ch1 = methylation class: GBM, MES
!Sample_description = SAMPLE 1
!Sample_supplementary_file = ftp://example/GSM0000001_1234_R01C01_Grn.idat.gz
!Sample_supplementary_file = ftp://example/GSM0000001_1234_R01C01_Red.idat.gz
^SAMPLE = GSM0000002
!Sample_title = LGG, PA, sample 2 [reference set]
!Sample_characteristics_ch1 = methylation class: LGG, PA
"""

MATRIX_FIXTURE = "\t".join(
    ["ID_REF", "SAMPLE 1", "Detection Pval", "SAMPLE 2", "Detection Pval"]
) + "\n" + "\n".join(
    [
        "cg00000001\t0.10\t0.0\t0.90\t0.0",
        "cg00000002\t0.55\t0.0\t0.45\t0.0",
    ]
) + "\n"

# Same as MATRIX_FIXTURE but with a third, truncated row -- mimics the real
# GSE109379_processed_data.txt.gz's mid-download truncation.
MALFORMED_MATRIX_FIXTURE = MATRIX_FIXTURE + "cg00000003\t0.20\t0.0\n"


def _write_gz(path, content: str):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(content)


def test_parse_soft_family_resolves_sample_number_from_description_and_title(tmp_path):
    soft_path = tmp_path / "fixture_family.soft.gz"
    _write_gz(soft_path, SOFT_FIXTURE)

    labels = parse_soft_family(soft_path)

    assert list(labels["gsm"]) == ["GSM0000001", "GSM0000002"]
    assert list(labels["sample_column"]) == ["SAMPLE 1", "SAMPLE 2"]
    assert list(labels["methylation_class"]) == ["GBM, MES", "LGG, PA"]
    assert list(labels["idat_basename"]) == ["GSM0000001_1234_R01C01", None]


def test_parse_soft_family_rejects_duplicate_sample_column(tmp_path):
    soft_path = tmp_path / "dup_family.soft.gz"
    _write_gz(
        soft_path,
        SOFT_FIXTURE.replace(
            "!Sample_title = LGG, PA, sample 2 [reference set]",
            "!Sample_title = LGG, PA, sample 1 [reference set]",
        ),
    )

    with pytest.raises(GeoSeriesParseError):
        parse_soft_family(soft_path)


def test_parse_soft_family_rejects_missing_methylation_class(tmp_path):
    soft_path = tmp_path / "missing_class_family.soft.gz"
    _write_gz(
        soft_path,
        "^SAMPLE = GSM0000001\n"
        "!Sample_title = GBM, MES, sample 1 [reference set]\n",
    )

    with pytest.raises(GeoSeriesParseError):
        parse_soft_family(soft_path)


def test_read_geo_beta_matrix_selects_only_requested_sample_columns_in_order(tmp_path):
    matrix_path = tmp_path / "fixture_beta.txt.gz"
    _write_gz(matrix_path, MATRIX_FIXTURE)

    beta, probe_ids = read_geo_beta_matrix(matrix_path, sample_columns=["SAMPLE 2", "SAMPLE 1"])

    assert probe_ids == ["cg00000001", "cg00000002"]
    assert beta.shape == (2, 2)  # (samples, probes), requested order: SAMPLE 2 then SAMPLE 1
    np.testing.assert_allclose(beta[0], [0.90, 0.45])  # SAMPLE 2
    np.testing.assert_allclose(beta[1], [0.10, 0.55])  # SAMPLE 1


def test_read_geo_beta_matrix_rejects_unknown_column(tmp_path):
    matrix_path = tmp_path / "fixture_beta2.txt.gz"
    _write_gz(matrix_path, MATRIX_FIXTURE)

    with pytest.raises(GeoSeriesParseError):
        read_geo_beta_matrix(matrix_path, sample_columns=["SAMPLE 999"])


def test_parse_soft_family_rejects_mismatched_grn_red_basenames(tmp_path):
    soft_path = tmp_path / "mismatched_idat_family.soft.gz"
    _write_gz(
        soft_path,
        SOFT_FIXTURE.replace(
            "!Sample_supplementary_file = ftp://example/GSM0000001_1234_R01C01_Red.idat.gz",
            "!Sample_supplementary_file = ftp://example/GSM0000001_9999_R01C01_Red.idat.gz",
        ),
    )

    with pytest.raises(GeoSeriesParseError):
        parse_soft_family(soft_path)


def test_read_geo_beta_matrix_strict_raises_on_malformed_row(tmp_path):
    matrix_path = tmp_path / "malformed_beta.txt.gz"
    _write_gz(matrix_path, MALFORMED_MATRIX_FIXTURE)

    with pytest.raises(Exception):  # noqa: B017 - pyarrow raises its own ArrowInvalid
        read_geo_beta_matrix(matrix_path, sample_columns=["SAMPLE 1", "SAMPLE 2"])


def test_read_geo_beta_matrix_tolerant_skips_malformed_row(tmp_path):
    matrix_path = tmp_path / "malformed_beta2.txt.gz"
    _write_gz(matrix_path, MALFORMED_MATRIX_FIXTURE)

    beta, probe_ids, n_skipped = read_geo_beta_matrix_tolerant(
        matrix_path, sample_columns=["SAMPLE 1", "SAMPLE 2"]
    )

    assert n_skipped == 1
    assert probe_ids == ["cg00000001", "cg00000002"]
    assert beta.shape == (2, 2)
    np.testing.assert_allclose(beta[0], [0.10, 0.55])  # SAMPLE 1
    np.testing.assert_allclose(beta[1], [0.90, 0.45])  # SAMPLE 2
