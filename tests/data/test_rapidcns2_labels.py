import openpyxl
import pytest

from neuromethyl_ont.data.rapidcns2_labels import (
    RapidCNS2LabelsError,
    load_rapidcns2_labels,
    normalize_v11_class_code,
)

GSE90496_CLASSES = {"GBM, RTK II", "O IDH", "MNG", "LGG, PA/GG ST"}


@pytest.mark.parametrize(
    "gse90496_label,expected",
    [
        ("GBM, RTK II", "GBM_RTK_II"),
        ("O IDH", "O_IDH"),
        ("MNG", "MNG"),
        ("LGG, PA/GG ST", "LGG_PA_GG_ST"),
    ],
)
def test_normalize_v11_class_code(gse90496_label, expected):
    assert normalize_v11_class_code(gse90496_label) == expected


def _write_supplementary_table_1(tmp_path, data_rows):
    """Build a minimal xlsx matching the real file's layout: sheet named
    'Supplementary Table 1', 3 title/blank rows, then a header row at index 3
    with 'Study ID' and the true-class column among others.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Supplementary Table 1"
    ws.append(["Supplementary Table 1: Overview of Rapid-CNS2 cohort"])
    ws.append([])
    ws.append([])
    ws.append(
        [
            "Study ID",
            "Sample type",
            "Rapid-CNS2 (i.e. v11 reference set) true class (including inferred) ",
        ]
    )
    for row in data_rows:
        ws.append(row)
    path = tmp_path / "Supplementary_Tables_1_2.xlsx"
    wb.save(path)
    return path


def test_load_rapidcns2_labels_maps_and_drops_correctly(tmp_path):
    path = _write_supplementary_table_1(
        tmp_path,
        [
            [1, "primary", "GBM_RTK_II"],
            [2, "low_grade", "O_IDH"],
            [3, "primary", "MNG"],
            [4, "primary", "LGG_PA_GG_ST"],
            [5, "primary", "GBM_nnn"],  # subtype-unresolved -> dropped
            [6, "primary", "not_in_classifier"],  # explicit non-classification -> dropped
            [7, "primary", "AAP (HGAP)"],  # not a v11/GSE90496 class -> dropped
            [8, "primary", None],  # blank -> dropped
        ],
    )

    result = load_rapidcns2_labels(path, GSE90496_CLASSES)

    assert result.n_total_rows == 8
    assert result.n_kept == 4
    assert list(result.labels["sample_id"]) == ["ID_1", "ID_2", "ID_3", "ID_4"]
    assert list(result.labels["label"]) == ["GBM, RTK II", "O IDH", "MNG", "LGG, PA/GG ST"]
    assert result.dropped_reasons == {
        "GBM_nnn": 1,
        "not_in_classifier": 1,
        "AAP (HGAP)": 1,
        "<blank>": 1,
    }


def test_load_rapidcns2_labels_missing_file(tmp_path):
    with pytest.raises(RapidCNS2LabelsError):
        load_rapidcns2_labels(tmp_path / "does_not_exist.xlsx", GSE90496_CLASSES)


def test_normalize_v11_class_code_is_injective_over_known_gse90496_classes():
    """The 91 GSE90496 class strings this project trains on (docs/EVIDENCE_LINEAR_V1.md
    #2) must normalize to 91 distinct codes -- if two different classes ever
    collided, load_rapidcns2_labels would silently mislabel one of them.
    """
    # A frozen snapshot of GSE90496's 91 methylation_class values (see
    # test_load_rapidcns2_labels_maps_and_drops_correctly's docstring companion:
    # neuromethyl_ont.data.geo_series.parse_soft_family on the real SOFT file).
    classes = {
        "A IDH", "A IDH, HG", "ANA PA", "ATRT, MYC", "ATRT, SHH", "ATRT, TYR", "CHGL",
        "CHORDM", "CN", "CNS NB, FOXR2", "CONTR, ADENOPIT", "CONTR, CEBM", "CONTR, HEMI",
        "CONTR, HYPTHAL", "CONTR, INFLAM", "CONTR, PINEAL", "CONTR, PONS", "CONTR, REACT",
        "CONTR, WM", "CPH, ADM", "CPH, PAP", "DLGNT", "DMG, K27", "EFT, CIC", "ENB, A",
        "ENB, B", "EPN, MPE", "EPN, PF A", "EPN, PF B", "EPN, RELA", "EPN, SPINE",
        "EPN, YAP", "ETMR", "EWS", "GBM, G34", "GBM, MES", "GBM, MID", "GBM, MYCN",
        "GBM, RTK I", "GBM, RTK II", "GBM, RTK III", "HGNET, BCOR", "HGNET, MN1", "HMB",
        "IHG", "LGG, DIG/DIA", "LGG, DNT", "LGG, GG", "LGG, MYB", "LGG, PA MID",
        "LGG, PA PF", "LGG, PA/GG ST", "LGG, RGNT", "LGG, SEGA", "LIPN", "LYMPHO",
        "MB, G3", "MB, G4", "MB, SHH CHL AD", "MB, SHH INF", "MB, WNT", "MELAN", "MELCYT",
        "MNG", "O IDH", "PGG, nC", "PIN T,  PB A", "PIN T,  PB B", "PIN T, PPT",
        "PITAD, ACTH", "PITAD, FSH LH", "PITAD, PRL", "PITAD, STH DNS A",
        "PITAD, STH DNS B", "PITAD, STH SPA", "PITAD, TSH", "PITUI", "PLASMA",
        "PLEX, AD", "PLEX, PED A", "PLEX, PED B", "PTPR, A", "PTPR, B", "PXA", "RETB",
        "SCHW", "SCHW, MEL", "SFT HMPC", "SUBEPN, PF", "SUBEPN, SPINE", "SUBEPN, ST",
    }
    assert len(classes) == 91
    codes = [normalize_v11_class_code(c) for c in classes]
    assert len(set(codes)) == 91


def test_load_rapidcns2_labels_wrong_sheet(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.title = "Sheet1"
    path = tmp_path / "wrong.xlsx"
    wb.save(path)

    with pytest.raises(RapidCNS2LabelsError):
        load_rapidcns2_labels(path, GSE90496_CLASSES)
