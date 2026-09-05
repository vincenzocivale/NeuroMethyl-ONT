"""Parser for GEO SOFT family files and the accompanying beta-value matrix,
as published for GSE90496 (Capper et al. 2018 reference cohort) and GSE109379
(its prospective validation cohort) -- see docs/EVIDENCE_LINEAR_V1.md #2.

Two files per dataset:

- ``*_family.soft.gz``: one ``^SAMPLE = GSMxxxxxxx`` block per sample, with
  ``!Sample_characteristics_ch1 = methylation class: <label>`` and either an
  explicit ``!Sample_description = SAMPLE <n>`` line (GSE109379) or a
  ``sample <n>`` token inside ``!Sample_title`` (GSE90496). Both encode which
  column of the matrix file that GSM corresponds to.
- ``*_beta.txt.gz`` / ``*_processed_data.txt.gz``: one row per probe
  (``ID_REF`` = probe id), columns alternating ``SAMPLE <n>`` (beta value)
  and ``Detection Pval`` (discarded here -- not part of the V1 feature space).

Samples are matched to matrix columns by the literal ``SAMPLE <n>`` string,
never by position, so a gap or reordering in either file fails loudly instead
of silently mislabeling a sample.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

SAMPLE_TITLE_NUMBER_RE = re.compile(r"\bsample\s+(\d+)\b", re.IGNORECASE)
SAMPLE_DESCRIPTION_RE = re.compile(r"^SAMPLE\s+(\d+)\s*$", re.IGNORECASE)
IDAT_SUPPLEMENTARY_FILE_RE = re.compile(r"/([^/]+)_(Grn|Red)\.idat\.gz$", re.IGNORECASE)


class GeoSeriesParseError(ValueError):
    """Raised when a SOFT family file's structure doesn't match what this parser expects."""


@dataclass(frozen=True)
class GeoSampleLabel:
    gsm: str
    sample_column: str  # e.g. "SAMPLE 42", the exact matrix column header
    methylation_class: str
    sample_title: str
    idat_basename: str | None  # e.g. "GSM2940725_10003886252_R05C02", or None if absent


def _idat_basename_from_block(block: dict[str, list[str]], gsm: str) -> str | None:
    """Derive the shared ``<basename>`` from the Grn/Red ``!Sample_supplementary_file``
    lines (``.../<basename>_Grn.idat.gz`` and ``..._Red.idat.gz``). Returns ``None``
    if the block has no IDAT supplementary files (e.g. array data was not deposited);
    raises if the two channels disagree on basename, since that would silently pair
    the wrong Red/Green files for this sample.
    """
    basenames: dict[str, str] = {}
    for supplementary_file in block.get("supplementary_file", []):
        m = IDAT_SUPPLEMENTARY_FILE_RE.search(supplementary_file)
        if m:
            basenames[m.group(2).lower()] = m.group(1)
    if not basenames:
        return None
    if set(basenames) != {"grn", "red"} or basenames["grn"] != basenames["red"]:
        raise GeoSeriesParseError(
            f"{gsm}: inconsistent Grn/Red IDAT supplementary files: "
            f"{block.get('supplementary_file', [])}"
        )
    return basenames["grn"]


def _sample_number_from_block(block: dict[str, list[str]], gsm: str) -> int:
    """Prefer the explicit ``!Sample_description = SAMPLE <n>`` line; fall back
    to parsing ``sample <n>`` out of ``!Sample_title``.
    """
    for description in block.get("description", []):
        m = SAMPLE_DESCRIPTION_RE.match(description.strip())
        if m:
            return int(m.group(1))
    for title in block.get("title", []):
        m = SAMPLE_TITLE_NUMBER_RE.search(title)
        if m:
            return int(m.group(1))
    raise GeoSeriesParseError(
        f"{gsm}: could not determine sample number from title/description"
    )


def parse_soft_family(path: str | Path) -> pd.DataFrame:
    """Parse a ``*_family.soft.gz`` file into one row per sample.

    Returns a DataFrame with columns ``gsm``, ``sample_column``,
    ``methylation_class``, ``sample_title``. Raises :class:`GeoSeriesParseError`
    if a sample is missing its methylation class, or if two samples resolve
    to the same matrix column (both would silently corrupt the label join).
    """
    path = Path(path)
    labels: list[GeoSampleLabel] = []
    seen_columns: dict[str, str] = {}

    gsm: str | None = None
    block: dict[str, list[str]] = {}

    def flush() -> None:
        if gsm is None:
            return
        classes = block.get("methylation_class", [])
        if len(classes) != 1:
            raise GeoSeriesParseError(
                f"{gsm}: expected exactly one 'methylation class' characteristic, "
                f"found {len(classes)}"
            )
        sample_number = _sample_number_from_block(block, gsm)
        sample_column = f"SAMPLE {sample_number}"
        if sample_column in seen_columns:
            raise GeoSeriesParseError(
                f"{sample_column} claimed by both {seen_columns[sample_column]} and {gsm}"
            )
        seen_columns[sample_column] = gsm
        labels.append(
            GeoSampleLabel(
                gsm=gsm,
                sample_column=sample_column,
                methylation_class=classes[0],
                sample_title=block.get("title", [""])[0],
                idat_basename=_idat_basename_from_block(block, gsm),
            )
        )

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                flush()
                gsm = line.split("=", 1)[1].strip()
                block = {}
                continue
            if gsm is None:
                continue
            if line.startswith("!Sample_title = "):
                block.setdefault("title", []).append(line.split("=", 1)[1].strip())
            elif line.startswith("!Sample_characteristics_ch1 = methylation class: "):
                block.setdefault("methylation_class", []).append(
                    line.split(": ", 1)[1].strip()
                )
            elif line.startswith("!Sample_description = "):
                block.setdefault("description", []).append(line.split("=", 1)[1].strip())
            elif line.startswith("!Sample_supplementary_file = "):
                block.setdefault("supplementary_file", []).append(
                    line.split("=", 1)[1].strip()
                )
        flush()

    if not labels:
        raise GeoSeriesParseError(f"{path}: no ^SAMPLE blocks found")

    return pd.DataFrame(
        {
            "gsm": [label.gsm for label in labels],
            "sample_column": [label.sample_column for label in labels],
            "methylation_class": [label.methylation_class for label in labels],
            "sample_title": [label.sample_title for label in labels],
            "idat_basename": [label.idat_basename for label in labels],
        }
    )


def read_geo_beta_matrix(
    path: str | Path, sample_columns: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Read only the requested ``SAMPLE <n>`` beta columns (never the
    ``Detection Pval`` columns) from a GEO series matrix file, in the given
    order.

    Returns ``(beta, probe_ids)`` with ``beta.shape == (len(sample_columns), n_probes)``.
    Column lookup is done on the real header (by exact name), then reading is
    done against synthetic unique column names -- ``Detection Pval`` repeats
    once per sample and would otherwise collide.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")

    if header[0] != "ID_REF":
        raise GeoSeriesParseError(f"{path}: expected first column 'ID_REF', got {header[0]!r}")

    column_positions: dict[str, list[int]] = {}
    for i, name in enumerate(header):
        column_positions.setdefault(name, []).append(i)

    sample_indices = []
    for sample_column in sample_columns:
        positions = column_positions.get(sample_column)
        if not positions:
            raise GeoSeriesParseError(f"{path}: column {sample_column!r} not found in header")
        if len(positions) > 1:
            raise GeoSeriesParseError(f"{path}: column {sample_column!r} appears more than once")
        sample_indices.append(positions[0])

    synthetic_names = [f"c{i}" for i in range(len(header))]
    id_ref_name = synthetic_names[0]
    wanted_names = [id_ref_name] + [synthetic_names[i] for i in sample_indices]

    table = pa_csv.read_csv(
        path,
        read_options=pa_csv.ReadOptions(column_names=synthetic_names, skip_rows=1),
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        convert_options=pa_csv.ConvertOptions(
            include_columns=wanted_names,
            column_types={
                id_ref_name: pa.string(),
                **{synthetic_names[i]: pa.float32() for i in sample_indices},
            },
        ),
    )

    probe_ids = table.column(id_ref_name).to_pylist()
    beta_probes_by_samples = np.column_stack(
        [table.column(synthetic_names[i]).to_numpy(zero_copy_only=False) for i in sample_indices]
    )
    return beta_probes_by_samples.T, probe_ids


def read_geo_beta_matrix_tolerant(
    path: str | Path, sample_columns: list[str]
) -> tuple[np.ndarray, list[str], int]:
    """Like :func:`read_geo_beta_matrix`, but skips rows with the wrong field
    count instead of raising.

    For validating a reconstruction against a GEO matrix file that is known
    to be malformed (e.g. truncated mid-download): the file's own intact rows
    are still real, independently-computed evidence, and the ragged tail
    should not block using them. Never use this for anything other than a
    known-malformed comparison source -- a normal read still uses
    :func:`read_geo_beta_matrix` so a truncation is caught, not silently
    absorbed.

    Returns ``(beta, probe_ids, n_rows_skipped)``.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")

    if header[0] != "ID_REF":
        raise GeoSeriesParseError(f"{path}: expected first column 'ID_REF', got {header[0]!r}")

    column_positions: dict[str, list[int]] = {}
    for i, name in enumerate(header):
        column_positions.setdefault(name, []).append(i)

    sample_indices = []
    for sample_column in sample_columns:
        positions = column_positions.get(sample_column)
        if not positions:
            raise GeoSeriesParseError(f"{path}: column {sample_column!r} not found in header")
        if len(positions) > 1:
            raise GeoSeriesParseError(f"{path}: column {sample_column!r} appears more than once")
        sample_indices.append(positions[0])

    synthetic_names = [f"c{i}" for i in range(len(header))]
    id_ref_name = synthetic_names[0]
    wanted_names = [id_ref_name] + [synthetic_names[i] for i in sample_indices]

    n_skipped = 0

    def _skip_invalid_row(_row) -> str:
        nonlocal n_skipped
        n_skipped += 1
        return "skip"

    table = pa_csv.read_csv(
        path,
        read_options=pa_csv.ReadOptions(column_names=synthetic_names, skip_rows=1),
        parse_options=pa_csv.ParseOptions(delimiter="\t", invalid_row_handler=_skip_invalid_row),
        convert_options=pa_csv.ConvertOptions(
            include_columns=wanted_names,
            column_types={
                id_ref_name: pa.string(),
                **{synthetic_names[i]: pa.float32() for i in sample_indices},
            },
        ),
    )

    probe_ids = table.column(id_ref_name).to_pylist()
    beta_probes_by_samples = np.column_stack(
        [table.column(synthetic_names[i]).to_numpy(zero_copy_only=False) for i in sample_indices]
    )
    return beta_probes_by_samples.T, probe_ids, n_skipped
