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

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from neuromethyl_ont.data.read_simulator import ReadModel
from neuromethyl_ont.data.sequencing_simulator import (
    apply_representation,
    encode_batch,
    encode_binary_batch,
    encode_full_array,
    sample_lambda_batch,
    simulate_counts,
)
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
    batch_encoding_workers: int = 1
    read_model: ReadModel | None = None  # None: i.i.d. Poisson depth; else read-structured


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_accuracy: float | None


@dataclass
class TrainingResult:
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_accuracy: float | None = float("-inf")
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
        read_model=config.read_model,
    )


def _encode_chunk(
    representation: str,
    beta: np.ndarray,
    lam: np.ndarray,
    seed: int,
    mu: np.ndarray | None,
    kappa: float,
    threshold: float,
    read_model: ReadModel | None = None,
) -> np.ndarray:
    """Encode one independent batch slice with its own deterministic RNG."""
    chunk_rng = np.random.default_rng(seed)
    if representation == "binary" and read_model is None:
        return encode_binary_batch(beta, lam=lam, rng=chunk_rng, threshold=threshold)
    counts = simulate_counts(beta, lam, chunk_rng, read_model=read_model)
    return apply_representation(
        representation, beta, counts, mu=mu, kappa=kappa, threshold=threshold
    )


def _encode_parallel(
    config: TrainingConfig,
    beta: np.ndarray,
    mu: np.ndarray | None,
    rng: np.random.Generator,
    pool: ThreadPoolExecutor | None = None,
) -> np.ndarray:
    """Bounded-memory, deterministic parallel batch encoding.

    The simulator's Poisson/Binomial draws dominate wall time.  Independent
    row slices permit a small number of NumPy worker threads while keeping the
    full reference shared and allocating only one batch-sized output.
    """
    n_workers = min(config.batch_encoding_workers, beta.shape[0])
    if n_workers <= 1:
        return _encode(config, beta, mu, rng)

    lam = sample_lambda_batch(rng, size=beta.shape[0])
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=n_workers, dtype=np.uint64)
    boundaries = np.linspace(0, beta.shape[0], num=n_workers + 1, dtype=int)
    encoded = np.empty(beta.shape, dtype=np.float64)

    owns_pool = pool is None
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="methyl-encode")
    try:
        futures = [
            (
                start,
                end,
                pool.submit(
                    _encode_chunk,
                    config.representation,
                    beta[start:end],
                    lam[start:end],
                    int(seeds[i]),
                    mu,
                    config.kappa,
                    config.binary_threshold,
                    config.read_model,
                ),
            )
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True))
        ]
        for start, end, future in futures:
            encoded[start:end] = future.result()
    finally:
        if owns_pool:
            pool.shutdown()
    return encoded


def predict_logits(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Run ``model`` on a full feature matrix and return logits as a numpy array.

    Always runs inference on the device ``model``'s parameters already live on
    (e.g. after training with ``config.device == "cuda"``) and always brings
    the result back to CPU before the ``.numpy()`` conversion -- ``.numpy()``
    on a CUDA tensor raises, so this must not be skipped.
    """
    device = next(model.parameters()).device
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x_t).cpu().numpy()


def train_linear_classifier(
    model: nn.Module,
    beta_train: np.ndarray,
    y_train: np.ndarray,
    beta_val: np.ndarray | None,
    y_val: np.ndarray | None,
    mu: np.ndarray,
    config: TrainingConfig,
) -> TrainingResult:
    """Train ``model`` with per-epoch depth-augmented features.

    ``mu`` is the training-set probe mean beta (unused by the binary
    representation, required by continuous/evidence).

    ``beta_val``/``y_val`` are optional. When both are ``None`` there is no
    per-epoch validation signal: training runs for exactly
    ``config.max_epochs`` (early stopping is disabled -- there is nothing to
    compare against) and the returned checkpoint is the model's state after
    the final epoch, not a "best" epoch by any validation metric. Pass both
    or neither; passing only one is a caller error.
    """
    if (beta_val is None) != (y_val is None):
        raise ValueError("beta_val and y_val must both be given or both be None")
    has_validation = beta_val is not None

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = build_loss(config.loss_name)

    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    if has_validation:
        x_val = torch.as_tensor(
            encode_full_array(
                config.representation, beta_val, mu=mu, threshold=config.binary_threshold
            ),
            dtype=torch.float32,
            device=device,
        )
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    n_train = beta_train.shape[0]
    result = TrainingResult(best_val_accuracy=float("-inf") if has_validation else None)
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(n_train)
        epoch_loss_sum = 0.0
        for start in range(0, n_train, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            x_batch = torch.as_tensor(
                _encode_parallel(config, beta_train[batch_idx], mu, rng),
                dtype=torch.float32,
                device=device,
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
        epoch_seconds = time.perf_counter() - epoch_started
        print(
            f"Epoch {epoch + 1:03d}/{config.max_epochs}: "
            f"train_loss={train_loss:.6f} elapsed={epoch_seconds:.1f}s",
            flush=True,
        )

        if not has_validation:
            result.history.append(
                EpochRecord(epoch=epoch, train_loss=train_loss, val_loss=None, val_accuracy=None)
            )
            continue

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

    if not has_validation:
        result.best_epoch = len(result.history) - 1
        result.best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    return result
