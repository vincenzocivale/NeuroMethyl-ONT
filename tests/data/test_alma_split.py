import numpy as np
import pandas as pd

from neuromethyl_ont.data.alma_split import make_patient_split


def _frame(n_classes=4, patients_per_class=30, dup_every=5):
    rows = []
    for c in range(n_classes):
        for p in range(patients_per_class):
            pid = f"c{c}_p{p}"
            rows.append((f"{pid}_a", pid, f"class{c}"))
            if p % dup_every == 0:
                rows.append((f"{pid}_b", pid, f"class{c}"))
    return pd.DataFrame(rows, columns=["sample_id", "patient", "label"]).set_index("sample_id")


def test_split_is_patient_disjoint_and_covers_every_class():
    frame = _frame()
    split = make_patient_split(frame, "label", "patient", seed=1)
    assert set(split) == {"train", "val", "test"}
    per_patient = split.groupby(frame["patient"]).nunique()
    assert (per_patient == 1).all()
    for name in ("train", "val", "test"):
        assert set(frame.loc[split == name, "label"]) == set(frame["label"])


def test_split_is_deterministic():
    frame = _frame()
    a = make_patient_split(frame, "label", "patient", seed=3)
    b = make_patient_split(frame, "label", "patient", seed=3)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
