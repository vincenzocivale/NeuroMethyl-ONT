"""Many-to-one class mapping from a source taxonomy (e.g. the 91 GSE90496 classes) to coarse labels.

The mapping lives in a versioned YAML under ``configs/taxonomies/`` (see ``methylyzr_who_level.yaml``).
Probabilities are summed over merged classes, so no mass is lost or renormalized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_taxonomy(path: str | Path) -> dict:
    """Load a taxonomy YAML and check it is internally consistent."""
    tax = yaml.safe_load(Path(path).read_text())
    codes = [c["code"] for c in tax["classes"]]
    if len(set(codes)) != len(codes):
        raise ValueError(f"source classes assigned more than once: {sorted({c for c in codes if codes.count(c) > 1})}")
    known = set(tax["coarse_labels"])
    for c in tax["classes"]:
        for key in ("coarse", "alt_coarse"):
            if c.get(key) is not None and c[key] not in known:
                raise ValueError(f"class {c['code']!r}: {key} {c[key]!r} not in coarse_labels")
    for name, entry in tax["cohort_labels"].items():
        if entry["coarse"] not in known | {tax["out_of_scope_label"]}:
            raise ValueError(f"cohort label {name!r} maps to unknown {entry['coarse']!r}")
    return tax


def map_to_coarse(
    probs: np.ndarray, class_names: list[str], taxonomy: dict
) -> tuple[np.ndarray, list[str]]:
    """Sum (n_samples x n_classes) probabilities into coarse labels; columns follow ``coarse_labels``.

    ``class_names`` gives the column order of ``probs`` and must be exactly the taxonomy's source
    classes (each once), otherwise this raises instead of silently dropping or double counting mass.
    """
    coarse_by_class = {c["code"]: c["coarse"] for c in taxonomy["classes"]}
    if len(set(class_names)) != len(class_names) or set(class_names) != set(coarse_by_class):
        missing = sorted(set(coarse_by_class) - set(class_names))
        extra = sorted(set(class_names) - set(coarse_by_class))
        raise ValueError(f"class_names do not match the taxonomy (missing={missing}, unexpected={extra})")
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(class_names):
        raise ValueError(f"probs shape {probs.shape} does not match {len(class_names)} classes")
    labels = list(taxonomy["coarse_labels"])
    out = np.zeros((len(probs), len(labels)))
    for j, name in enumerate(class_names):
        out[:, labels.index(coarse_by_class[name])] += probs[:, j]
    if not np.allclose(out.sum(axis=1), probs.sum(axis=1)):
        raise RuntimeError("coarse mapping did not preserve probability mass")
    return out, labels


def coarse_truth(label_short: str, taxonomy: dict) -> tuple[str, bool]:
    """Cohort label -> (coarse label, abstention_test flag). Unknown labels raise KeyError."""
    entry = taxonomy["cohort_labels"][label_short]
    return entry["coarse"], bool(entry.get("abstention_test", False))
