"""Domain router: CNS tumor vs acute leukemia.

A linear classifier over Binary-style binary features (+1/-1, 0 = unobserved)
restricted to probes shared by the CNS (GSE90496) and ALMA references, so the
router cannot separate the domains by which probes exist. Random probe
masking during training mimics sparse nanopore coverage. It only decides which
specialist head (91-class CNS or 28-class ALMA) to run, and abstains when the
domain probability is not confident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

DOMAINS = ("cns", "leukemia")


@dataclass
class RouterTrainingResult:
    model: nn.Linear
    best_epoch: int
    history: list[dict[str, float]]


def _scale(n_features: int) -> float:
    return float(np.sqrt(n_features))


def predict_domain_proba(model: nn.Linear, x: np.ndarray, device: str = "cpu", chunk: int = 256):
    """Return P(domain) as an (n, 2) array, columns ordered as ``DOMAINS``."""
    model = model.to(device).eval()
    scale = _scale(x.shape[1])
    out = []
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(x[start:start + chunk])).to(device)
            out.append(torch.softmax(model(xb / scale), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_router(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cuda",
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 128,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    keep_range: tuple[float, float] = (0.2, 1.0),
    seed: int = 17,
) -> RouterTrainingResult:
    """Fit the router; select the epoch by validation log-loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_features = x_train.shape[1]
    scale = _scale(n_features)
    model = nn.Linear(n_features, len(DOMAINS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    # Features stay in host memory; only each batch is moved to the device.
    x_host = torch.from_numpy(x_train)
    y_host = torch.from_numpy(y_train.astype(np.int64))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    best_loss, best_epoch, best_state, history = np.inf, -1, None, []
    for epoch in range(max_epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(x_train)))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = x_host[idx].to(device)
            keep = torch.empty(len(idx), 1, device=device).uniform_(*keep_range)
            xb = xb * (torch.rand_like(xb) < keep)
            loss = loss_fn(model(xb / scale), y_host[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        proba = predict_domain_proba(model, x_val, device)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(proba) + 1e-12), y_val_t))
        val_acc = float((proba.argmax(1) == y_val).mean())
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return RouterTrainingResult(model.cpu().eval(), best_epoch, history)
