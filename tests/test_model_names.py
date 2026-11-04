import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuromethyl_ont.data.array_reference import FeatureIndex
from neuromethyl_ont.inference.two_stage import Head, load_linear_checkpoint
from neuromethyl_ont.model_names import (
    LEGACY_MODEL_NAMES,
    canonical_model_name,
    canonicalize_frame,
    read_results_tsv,
)
from neuromethyl_ont.models.evidence_linear import build_linear_classifier

CLASSIFY = Path(__file__).resolve().parents[1] / "scripts" / "inference" / "classify_sample.py"


def test_legacy_mapping_and_passthrough():
    assert LEGACY_MODEL_NAMES == {
        "B0": "Binary", "C0": "Beta", "E1": "Evidence", "B0+E1": "Consensus",
    }
    assert canonical_model_name("B0+E1") == "Consensus"
    assert canonical_model_name("Binary") == "Binary"
    assert canonical_model_name("EB1") == "EB1"  # not renamed
    assert canonical_model_name("Sturgeon") == "Sturgeon"


def test_old_results_tsv_is_read_with_new_names(tmp_path):
    old = pd.DataFrame({
        "model": ["B0", "B0+E1", "crossNN"], "accuracy": [0.9, 0.95, 0.8],
        "B0_predicted_family": ["a"] * 3, "B0+E1_confidence": [0.5] * 3, "E1": [1, 2, 3],
        "EB1_confidence": [0.1] * 3,
    })
    path = tmp_path / "old.tsv"
    old.to_csv(path, sep="\t", index=False)
    new = read_results_tsv(path)
    assert list(new["model"]) == ["Binary", "Consensus", "crossNN"]
    assert list(new.columns) == [
        "model", "accuracy", "Binary_predicted_family", "Consensus_confidence", "Evidence",
        "EB1_confidence",
    ]
    assert canonicalize_frame(new).equals(new)  # idempotent on new files


def test_old_checkpoint_still_loads(tmp_path):
    # checkpoints store the representation, never a model name: format unchanged by the rename
    model = build_linear_classifier(n_features=4, n_classes=2)
    path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(), "class_to_idx": {"a": 0, "b": 1},
            "representation": "binary", "n_features": 4,
        },
        path,
    )
    loaded, classes = load_linear_checkpoint(path, "binary")
    assert classes == ["a", "b"]


def test_classify_report_uses_new_names(tmp_path):
    spec = importlib.util.spec_from_file_location("classify_sample", CLASSIFY)
    classify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify)
    probes = [f"cg{i}" for i in range(4)]
    head = Head(
        "cns", FeatureIndex(probes, "x", "toy", "now"), np.full(4, 0.5),
        build_linear_classifier(n_features=4, n_classes=2),
        build_linear_classifier(n_features=4, n_classes=2), ["a", "b"],
    )
    counts = tmp_path / "counts.tsv"
    pd.DataFrame({"probe_id": probes, "methylated_count": [3, 0, 2, 1],
                  "unmethylated_count": [0, 2, 1, 1]}).to_csv(counts, sep="\t", index=False)
    report = classify.classify_with_head(head, counts, "s1", top_n=2, n_runner_up=1)
    assert set(report["per_model"]) == {"Binary", "Evidence"}
    assert report["top3_model"] == "Consensus"
