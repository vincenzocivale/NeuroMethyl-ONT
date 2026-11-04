"""Family-level taxonomy aggregation shared by every SOTA comparison script.

Every model compared this way (Binary, Beta, Evidence, the Consensus ensemble, Sturgeon,
crossNN) is aggregated identically: no
probability mass is discarded or renormalized, and any class not covered by
the taxonomy's ``family_code`` column lands in one explicit ``UNRESOLVED``
bucket rather than being dropped.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

UNRESOLVED = "UNRESOLVED"


def aggregate_to_family(
    probs: np.ndarray, labels: list[str], family_by_class: dict[str, str]
) -> tuple[np.ndarray, list[str]]:
    """Aggregate every class exactly once; unresolved classes form one bucket."""
    families = sorted({family_by_class.get(label, UNRESOLVED) for label in labels})
    out = np.zeros((len(probs), len(families)), dtype=np.float64)
    for j, label in enumerate(labels):
        out[:, families.index(family_by_class.get(label, UNRESOLVED))] += probs[:, j]
    if not np.allclose(out.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise RuntimeError("family aggregation did not preserve probability mass")
    return out, families


def sturgeon_model_class(column: str) -> str:
    """Translate Sturgeon's public class label into the matching v11 short code."""
    group, code, subtype = column.split(" - ", 2)
    if group == "Control":
        return f"CONTR, {subtype}"
    if group == "Embryonal":
        return {"CNS NB": "CNS NB, FOXR2", "ETMR": "ETMR", "HGNET": "HGNET, BCOR"}.get(
            code, f"{code}, {subtype}"
        ) if code not in {"MB G3G4", "MB SHH", "MB WNT"} else {
            "MB G3G4": f"MB, {subtype}",
            "MB SHH": "MB, SHH CHL AD" if subtype == "CHL AD INF" else "MB, SHH INF",
            "MB WNT": "MB, WNT",
        }[code]
    if group == "Ependymal":
        return "SUBEPN, PF" if code == "SUBEPN" else f"{code}, {subtype}"
    if group == "Glio-neuronal":
        return {"CN": "CN", "DLGNT": "DLGNT", "ENB": f"ENB, {subtype}", "LIPN": "LIPN",
                "PGG": "PGG, nC", "RETB": "RETB"}.get(code, f"LGG, {subtype}")
    if group == "Glioblastoma":
        return "DMG, K27" if code == "DMG" else f"GBM, {subtype}"
    if group == "Glioma IDH":
        return f"{code}, {subtype}" if code == "A IDH" else "O IDH"
    if group == "Haematopoeitic":
        return code
    if group == "Melanocytic":
        return code
    if group == "Mesenchymal":
        return {"CHORDM": "CHORDM", "EFT": "EFT, CIC", "EWS": "EWS", "HMB": "HMB",
                "MNG": "MNG", "SFT HMPC": "SFT HMPC"}[code]
    if group == "Nerve":
        return "SCHW, MEL" if subtype == "SCHW MEL" else "SCHW"
    if group == "Other glioma":
        return {"ANA PA": "ANA PA", "CHGL": "CHGL", "HGNET": "HGNET, MN1", "IHG": "IHG",
                "LGG MYB": "LGG, MYB", "LGG PA": f"LGG, {subtype}", "LGG SEGA": "LGG, SEGA",
                "PXA": "PXA"}[code]
    if group == "Pineal":
        return f"PIN T,  {subtype}" if code == "PIN T" else f"PTPR, {subtype}"
    if group == "Plexus":
        return f"PLEX, {subtype}"
    if group == "Sella":
        return "PITUI" if code == "PITUI SCO GCT" else f"{code}, {subtype}"
    raise ValueError(f"unrecognised Sturgeon class: {column!r}")


def family_metrics(
    true: np.ndarray, probs: np.ndarray, families: list[str]
) -> dict[str, float | int]:
    idx = {f: i for i, f in enumerate(families)}
    truth = np.array([idx[f] for f in true], dtype=np.int64)
    pred = probs.argmax(axis=1)
    order = np.argsort(-probs, axis=1)
    top2 = np.take_along_axis(probs, order[:, :2], axis=1)
    return {
        "n_samples": len(true),
        "family_accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "top3_family_accuracy": float(np.mean(np.any(order[:, :3] == truth[:, None], axis=1))),
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-15, 1.0)))),
        "mean_confidence_top1": float(probs.max(axis=1).mean()),
        "mean_true_family_probability": float(probs[np.arange(len(truth)), truth].mean()),
        "mean_top1_top2_margin": float((top2[:, 0] - top2[:, 1]).mean()),
    }
