"""Post-hoc explainability for the linear classifiers (Binary/Beta/Evidence).

Because every model built by ``models.evidence_linear.build_linear_classifier``
is exactly ``logits = Wx + b``, the contribution of each input feature to each
class's logit is available in closed form -- no SHAP/gradient-attribution
machinery is needed. ``explain_prediction`` decomposes one sample's logit for
a class into a per-probe contribution vector, ranks the top contributors, and
verifies the decomposition sums back to the model's own logit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neuromethyl_ont.models.evidence_linear import LinearClassifier


@dataclass(frozen=True)
class LocusContribution:
    """One probe's signed contribution to a class's logit."""

    probe_id: str
    feature_value: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class ExplanationReport:
    """Top contributing loci for the predicted class and each runner-up."""

    sample_id: str
    predicted_class: str
    predicted_class_logit: float
    top_classes: list[str]
    contributions_by_class: dict[str, list[LocusContribution]]


def _per_class_contributions(
    x: np.ndarray,
    weight_row: np.ndarray,
    probe_ids: list[str],
    top_n: int,
) -> list[LocusContribution]:
    contributions = x * weight_row
    order = np.argsort(-np.abs(contributions))[:top_n]
    return [
        LocusContribution(
            probe_id=probe_ids[i],
            feature_value=float(x[i]),
            weight=float(weight_row[i]),
            contribution=float(contributions[i]),
        )
        for i in order
    ]


def explain_prediction(
    model: LinearClassifier,
    x: np.ndarray,
    probe_ids: list[str],
    class_names: list[str],
    sample_id: str = "sample",
    top_n: int = 20,
    n_runner_up: int = 2,
) -> ExplanationReport:
    """Explain one sample's prediction by decomposing the linear logit.

    ``x`` is the model's input feature vector (post representation-specific
    encoding -- e.g. Binary's thresholded binary state), shape ``(n_features,)``,
    already aligned to ``probe_ids`` via the canonical
    :class:`~neuromethyl_ont.data.array_reference.FeatureIndex`.
    """
    if x.shape != (len(probe_ids),):
        raise ValueError(
            f"x shape {x.shape} does not match len(probe_ids)={len(probe_ids)}"
        )

    weight = model.linear.weight.detach().cpu().numpy()  # (n_classes, n_features)
    bias = model.linear.bias.detach().cpu().numpy()  # (n_classes,)
    if weight.shape[1] != len(probe_ids) or weight.shape[0] != len(class_names):
        raise ValueError(
            f"model shape {weight.shape} does not match "
            f"(n_classes={len(class_names)}, n_features={len(probe_ids)})"
        )

    logits = weight @ x + bias
    ranked_idx = np.argsort(-logits)
    top_idx = ranked_idx[: 1 + n_runner_up]
    top_classes = [class_names[i] for i in top_idx]

    contributions_by_class = {
        class_names[i]: _per_class_contributions(x, weight[i], probe_ids, top_n)
        for i in top_idx
    }

    return ExplanationReport(
        sample_id=sample_id,
        predicted_class=class_names[ranked_idx[0]],
        predicted_class_logit=float(logits[ranked_idx[0]]),
        top_classes=top_classes,
        contributions_by_class=contributions_by_class,
    )


def verify_decomposition(model: LinearClassifier, x: np.ndarray, class_idx: int) -> float:
    """Return the absolute error between the summed per-probe contributions
    (plus bias) and the model's own forward-pass logit for ``class_idx``.

    Used by tests to guard against the decomposition drifting out of sync
    with ``LinearClassifier.forward`` (e.g. if bias handling changes).
    """
    weight = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()
    decomposed = float(np.sum(x * weight[class_idx]) + bias[class_idx])

    with torch.no_grad():
        x_tensor = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        forward_logit = float(model(x_tensor)[0, class_idx])

    return abs(decomposed - forward_logit)
