"""Generic training loop shared by B0/C0/E1 (docs/EVIDENCE_LINEAR_V1.md #11).

Only the representation (``config.representation``) changes what feature
tensor a batch turns into; the optimizer, schedule and architecture
(``LinearClassifier``) are identical across configs, per spec section 5-6.
Per-epoch training features are re-sampled (depth augmentation, spec #6-9);
per-epoch validation uses the full-array closed form (spec #13's "verify the
taxonomy was learned" role) -- the fixed synthetic depth curve is a separate,
offline evaluation (``evaluation/depth_curve.py``), not part of early
stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from neuromethyl_ont.data.sequencing_simulator import encode_batch, encode_full_array
from neuromethyl_ont.training.objectives import build_loss


@dataclass
class TrainingConfig:
    representation: str  # "binary" | "continuous" | "evidence"
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 16
    max_epochs: int = 80
    early_stopping_patience: int = 12
    mixed_precision: str = "bf16"  # "bf16" | "none"
    kappa: float = 2.0
    binary_threshold: float = 0.6
    seed: int = 17
    loss_name: str = "cross_entropy"
    device: str = "cpu"


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


@dataclass
class TrainingResult:
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_accuracy: float = float("-inf")
    best_state_dict: dict | None = None
    stopped_early: bool = False


def _autocast(config: TrainingConfig):
    if config.mixed_precision == "bf16":
        return torch.autocast(device_type=config.device, dtype=torch.bfloat16)
    return torch.autocast(device_type=config.device, enabled=False)


def _encode(
    config: TrainingConfig, beta: np.ndarray, mu: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    return encode_batch(
        config.representation,
        beta,
        rng,
        mu=mu,
        kappa=config.kappa,
        threshold=config.binary_threshold,
    )


def train_linear_classifier(
    model: nn.Module,
    beta_train: np.ndarray,
    y_train: np.ndarray,
    beta_val: np.ndarray,
    y_val: np.ndarray,
    mu: np.ndarray,
    config: TrainingConfig,
) -> TrainingResult:
    """Train ``model`` with per-epoch depth-augmented features.

    ``mu`` is the training-set probe mean beta (unused by the binary
    representation, required by continuous/evidence).
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = build_loss(config.loss_name)

    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_val = torch.as_tensor(
        encode_full_array(
            config.representation, beta_val, mu=mu, threshold=config.binary_threshold
        ),
        dtype=torch.float32,
        device=device,
    )
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    n_train = beta_train.shape[0]
    result = TrainingResult()
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(n_train)
        epoch_loss_sum = 0.0
        for start in range(0, n_train, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            x_batch = torch.as_tensor(
                _encode(config, beta_train[batch_idx], mu, rng), dtype=torch.float32, device=device
            )
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            with _autocast(config):
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach()) * len(batch_idx)
        train_loss = epoch_loss_sum / n_train

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(loss_fn(val_logits, y_val_t))
            val_accuracy = float((val_logits.argmax(dim=1) == y_val_t).float().mean())

        result.history.append(
            EpochRecord(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy
            )
        )

        if val_accuracy > result.best_val_accuracy:
            result.best_val_accuracy = val_accuracy
            result.best_epoch = epoch
            result.best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                result.stopped_early = True
                break

    return result
