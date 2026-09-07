"""Ground-truth methylation-class labels for the Rapid-CNS2 benchmark cohort.

The Zenodo record (13351527) that ships Rapid-CNS2/ONT-WGS raw methylation
files has no per-sample diagnosis -- that lives in "Supplementary Table 1" of
the accompanying paper (Patel et al., *Prospective, multicentre validation of
a platform for rapid molecular profiling of central nervous system tumours*,
Nature Medicine 31, 849-861 (2025), DOI 10.1038/s41591-025-03562-5,
Supplementary Data file ``41591_2025_3562_MOESM3_ESM.xlsx``), expected at
``$NEUROMETHYL_DATA_ROOT/datasets/RapidCNS2/metadata/Supplementary_Tables_1_2.xlsx``.

``Study ID`` in that table (1..301) is the numeric suffix of this project's
``ID_<n>`` Rapid-CNS2 sample ids -- confirmed by both ranging 1..301 and by
the file counts matching. "Supplementary Table 2" (a separate Twist-panel
validation cohort, sample ids like ``Twist_1``) does not cover ONT-WGS: this
project has no ground-truth diagnosis source for ONT-WGS yet, so
:func:`load_rapidcns2_labels` only ever returns Rapid-CNS2 (``ID_*``) rows.

The table's "true class" column uses the Heidelberg v11b4 classifier's
underscore-joined short code (e.g. ``GBM_RTK_II``), not the comma/space GEO
convention GSE90496 uses (``GBM, RTK II``). :func:`normalize_v11_class_code`
converts the latter to the former (``", " -> "_"``, ``" " -> "_"``,
``"/" -> "_"``) so the two can be joined by exact string match --
:data:`MAPPING_CHECKED_AGAINST_ALL_91_GSE90496_CLASSES` in the test suite
guards this stays a lossless mapping for every current GSE90496 class name.

Rows are dropped (not guessed) when the true class is:

- family-only / subtype-unresolved (``GBM_nnn``, ``PITAD_nnn``, ``PLEX_nn``,
  ``PLEX_PED_nn``, ``CPH_nnn`` -- the ``nnn``/``nn`` suffix is the paper's own
  "not resolved to subclass" marker; scoring these against a specific 91-class
  subtype would be guessing what the paper explicitly left undetermined),
- explicitly non-classifiable (``not_in_classifier``, ``not_classifiable ...``),
- or outside the 91 GSE90496 v11 classes entirely (e.g. ``AAP (HGAP)`` /
  "high-grade astrocytoma with piloid features", a v12-only class; ``CPNT``,
  which does not match any v11 short code).

See ``load_rapidcns2_labels``'s return value for the accounting of how many
rows were kept vs. dropped and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STUDY_ID_COLUMN = "Study ID"
TRUE_CLASS_COLUMN = "Rapid-CNS2 (i.e. v11 reference set) true class (including inferred) "
SHEET_NAME = "Supplementary Table 1"
HEADER_ROW_0INDEXED = 3  # 3 title/blank rows precede the real header row


class RapidCNS2LabelsError(ValueError):
    """Raised when the supplementary table doesn't match the expected layout."""


def normalize_v11_class_code(gse90496_label: str) -> str:
    """``"GBM, RTK II"`` -> ``"GBM_RTK_II"`` (the Rapid-CNS2 table's convention)."""
    return gse90496_label.replace(", ", "_").replace(" ", "_").replace("/", "_")


@dataclass(frozen=True)
class RapidCNS2LabelsResult:
    labels: pd.DataFrame  # columns: sample_id, label (mapped, GSE90496 convention)
    n_total_rows: int
    n_kept: int
    dropped_reasons: dict[str, int] = field(default_factory=dict)  # raw true-class -> count


def load_rapidcns2_labels(
    xlsx_path: str | Path, gse90496_classes: set[str] | list[str]
) -> RapidCNS2LabelsResult:
    """Read ``Supplementary Table 1`` and return Rapid-CNS2 ``ID_<n>`` labels
    mapped onto ``gse90496_classes`` (in GSE90496's comma/space convention).

    Only rows whose true class maps unambiguously onto one of
    ``gse90496_classes`` are kept; everything else is dropped and counted in
    ``RapidCNS2LabelsResult.dropped_reasons`` (see module docstring for why).
    """
    import openpyxl  # local import: only needed by this one function

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise RapidCNS2LabelsError(f"not found: {xlsx_path}")

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise RapidCNS2LabelsError(
            f"{xlsx_path}: expected sheet {SHEET_NAME!r}, found {wb.sheetnames}"
        )
    rows = list(wb[SHEET_NAME].iter_rows(values_only=True))
    if len(rows) <= HEADER_ROW_0INDEXED:
        raise RapidCNS2LabelsError(f"{xlsx_path}: sheet has too few rows ({len(rows)})")

    header = rows[HEADER_ROW_0INDEXED]
    if STUDY_ID_COLUMN not in header or TRUE_CLASS_COLUMN not in header:
        raise RapidCNS2LabelsError(
            f"{xlsx_path}: expected columns {STUDY_ID_COLUMN!r} and {TRUE_CLASS_COLUMN!r} "
            f"in header row, got {header}"
        )
    idx_id = header.index(STUDY_ID_COLUMN)
    idx_class = header.index(TRUE_CLASS_COLUMN)

    norm_to_gse90496 = {normalize_v11_class_code(c): c for c in gse90496_classes}

    data_rows = [r for r in rows[HEADER_ROW_0INDEXED + 1 :] if r[idx_id] is not None]

    sample_ids: list[str] = []
    labels: list[str] = []
    dropped_reasons: dict[str, int] = {}

    for r in data_rows:
        study_id = r[idx_id]
        raw_class = r[idx_class]
        if raw_class is None:
            dropped_reasons["<blank>"] = dropped_reasons.get("<blank>", 0) + 1
            continue
        raw_class = str(raw_class).strip()
        mapped = norm_to_gse90496.get(raw_class)
        if mapped is None:
            dropped_reasons[raw_class] = dropped_reasons.get(raw_class, 0) + 1
            continue
        sample_ids.append(f"ID_{int(study_id)}")
        labels.append(mapped)

    labels_df = pd.DataFrame({"sample_id": sample_ids, "label": labels})
    return RapidCNS2LabelsResult(
        labels=labels_df,
        n_total_rows=len(data_rows),
        n_kept=len(labels_df),
        dropped_reasons=dropped_reasons,
    )
