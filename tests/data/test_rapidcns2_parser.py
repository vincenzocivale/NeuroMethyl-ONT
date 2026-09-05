import math

import pytest

from neuromethyl_ont.data.rapidcns2 import (
    SchemaDetectionError,
    read_rapidcns2_sample,
)


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_parser_with_header(tmp_path):
    content = (
        "chr start end coverage methylation_percentage IlmnID\n"
        "chr1 10524 10525 20 80.00 cg14817997\n"
    )
    path = _write(tmp_path, "with_header.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.detected_schema == "coverage_percentage"
    assert stats.n_header_lines == 1
    assert stats.n_data_rows == 1
    assert stats.n_malformed_rows == 0
    row = df.iloc[0]
    assert row["coverage"] == 20
    assert math.isclose(row["methylation_fraction"], 0.8)
    assert row["methylated_count"] == 16
    assert row["unmethylated_count"] == 4


def test_parser_headerless(tmp_path):
    content = "chr1 10524 10525 3 66.67 cg14817997\n"
    path = _write(tmp_path, "headerless.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.detected_schema == "coverage_percentage_headerless"
    row = df.iloc[0]
    assert row["coverage"] == 3
    assert row["methylated_count"] == 2
    assert row["unmethylated_count"] == 1


def test_parser_with_mod_column(tmp_path):
    content = (
        "chr\tstart\tend\tmod\tcoverage\tmethylation_percentage\tIlmnID\n"
        "chr1\t10524\t10525\tm\t16\t100\tcg14817997\n"
    )
    path = _write(tmp_path, "with_mod.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.has_mod_column is True
    assert stats.detected_schema == "coverage_percentage_mod"
    row = df.iloc[0]
    assert row["coverage"] == 16
    assert math.isclose(row["methylation_fraction"], 1.0)
    assert row["methylated_count"] == 16
    assert row["unmethylated_count"] == 0


def test_parser_quoted_header_and_values(tmp_path):
    content = (
        '"chr" "start" "end" "coverage" "methylation_percentage" "IlmnID"\n'
        '"chr1" 10524 10526 34 97.06 "cg14817997"\n'
    )
    path = _write(tmp_path, "quoted.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.n_data_rows == 1
    row = df.iloc[0]
    assert row["chrom"] == "chr1"
    assert row["probe_id"] == "cg14817997"
    assert row["coverage"] == 34


def test_parser_missing_methylation_is_not_zero(tmp_path):
    content = "chr1 10524 10525 5 NA cg14817997\n"
    path = _write(tmp_path, "missing.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.n_missing_methylation == 1
    row = df.iloc[0]
    assert row["methylation_fraction"] is None or (
        isinstance(row["methylation_fraction"], float) and math.isnan(row["methylation_fraction"])
    )
    assert row["coverage"] == 5
    assert row["methylated_count"] is None or pd_is_na(row["methylated_count"])


def pd_is_na(value):
    import pandas as pd

    return bool(pd.isna(value))


def test_parser_non_cg_probe_ids_do_not_fail(tmp_path):
    content = (
        "chr start end coverage methylation_percentage IlmnID\n"
        "chr10 129466555 129466556 14 92.86 MGMT\n"
        "chr1 1 2 1 100 rs11249206\n"
    )
    path = _write(tmp_path, "special_probes.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.n_data_rows == 2
    assert stats.special_probe_counts.get("MGMT") == 1
    assert stats.special_probe_counts.get("rs11249206") == 1


def test_parser_malformed_row_excluded_and_counted(tmp_path):
    content = (
        "chr start end coverage methylation_percentage IlmnID\n"
        "chr1 10524 10525 20 80.00 cg14817997\n"
        "chr1 10524 10525 20 150.00 cg00000001\n"  # out of range percentage
    )
    path = _write(tmp_path, "malformed.txt", content)

    df, stats = read_rapidcns2_sample(path)

    assert stats.n_data_rows == 1
    assert stats.n_malformed_rows == 1
    assert len(stats.anomalies) == 1


def test_unrecognized_schema_raises(tmp_path):
    content = "just some text that is not a methylation row\nmore garbage here too\n"
    path = _write(tmp_path, "garbage.txt", content)

    with pytest.raises(SchemaDetectionError):
        read_rapidcns2_sample(path)
