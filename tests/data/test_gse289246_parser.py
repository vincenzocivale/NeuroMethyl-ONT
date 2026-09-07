import math

import pytest

from neuromethyl_ont.data.gse289246 import GSE289246ParseError, read_gse289246_sample

HEADER = (
    "chrom\tstart\tend\titem\tscore\tstrand\tthickStart\tthickEnd\titemRgb\t"
    "coverage\tpercentMeth\tProbe_ID_450K\n"
)


def _write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(HEADER + "".join(rows))
    return path


def test_drops_rows_without_probe_annotation(tmp_path):
    rows = [
        # no probe annotation -> dropped
        "chr1\t10468\t10469\tCG\t0.86\t+\t10468\t10469\t0,0,0\t1\t86.2745\t\n",
        "chr1\t15865\t15866\tCG\t1.0\t+\t15865\t15866\t0,0,0\t10\t100.0\tcg13869341\n",
    ]
    path = _write(tmp_path, "sample.bed", rows)

    df = read_gse289246_sample(path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["probe_id"] == "cg13869341"
    assert row["coverage"] == 10
    assert row["methylated_count"] == 10
    assert row["unmethylated_count"] == 0
    assert math.isclose(row["methylation_fraction"], 1.0)


def test_derives_counts_from_coverage_and_percent(tmp_path):
    # 8/13 methylated at percentMeth = 61.5385 (rounds cleanly to 8)
    rows = ["chr2\t100\t101\tCG\t0.6154\t+\t100\t101\t0,0,0\t13\t61.5385\tcg00000957\n"]
    path = _write(tmp_path, "sample.bed", rows)

    df = read_gse289246_sample(path)

    row = df.iloc[0]
    assert row["coverage"] == 13
    assert row["methylated_count"] == 8
    assert row["unmethylated_count"] == 5


def test_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.bed"
    path.write_text("chrom\tstart\tend\n1\t2\t3\n")

    with pytest.raises(GSE289246ParseError):
        read_gse289246_sample(path)


def test_rejects_non_cg_probe_id(tmp_path):
    rows = ["chr1\t1\t2\tCG\t1.0\t+\t1\t2\t0,0,0\t5\t100.0\tMGMT\n"]
    path = _write(tmp_path, "sample.bed", rows)

    with pytest.raises(GSE289246ParseError):
        read_gse289246_sample(path)


def test_rejects_out_of_range_percent(tmp_path):
    rows = ["chr1\t1\t2\tCG\t1.0\t+\t1\t2\t0,0,0\t5\t150.0\tcg00000957\n"]
    path = _write(tmp_path, "sample.bed", rows)

    with pytest.raises(GSE289246ParseError):
        read_gse289246_sample(path)


def test_empty_probe_annotated_set_returns_empty_frame(tmp_path):
    rows = ["chr1\t1\t2\tCG\t0.5\t+\t1\t2\t0,0,0\t1\t50.0\t\n"]
    path = _write(tmp_path, "sample.bed", rows)

    df = read_gse289246_sample(path)

    assert df.empty
    assert list(df.columns) == [
        "chrom", "start", "end", "probe_id", "coverage",
        "methylation_fraction", "methylated_count", "unmethylated_count",
    ]


def test_output_feeds_canonical_aggregation(tmp_path):
    """Sums duplicate-coordinate evidence for the same probe, exactly as
    Rapid-CNS2 does (docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md)."""
    from neuromethyl_ont.data.canonical import aggregate_to_probe_level, to_coordinate_level

    rows = [
        "chr1\t15865\t15866\tCG\t1.0\t+\t15865\t15866\t0,0,0\t10\t100.0\tcg13869341\n",
        "chr1\t15866\t15867\tCG\t0.0\t-\t15866\t15867\t0,0,0\t4\t0.0\tcg13869341\n",
    ]
    path = _write(tmp_path, "sample.bed", rows)

    raw_df = read_gse289246_sample(path)
    coord_df = to_coordinate_level(raw_df, sample_id="GSM_TEST")
    probe_df = aggregate_to_probe_level(coord_df)

    assert len(probe_df) == 1
    row = probe_df.iloc[0]
    assert row["probe_id"] == "cg13869341"
    assert row["methylated_count"] == 10
    assert row["unmethylated_count"] == 4
    assert row["coverage"] == 14
