"""Training and evaluation utilities for hopf-layers experiments.

Provides a unified training loop, metric computation, and result logging
that is shared across all experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    patience: int = 10
    device: str = "auto"
    verbose: bool = True


@dataclass
class TrainResult:
    """Results from a single training run."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_metrics: list[float] = field(default_factory=list)
    best_val_metric: float = 0.0
    best_epoch: int = 0
    test_metric: float = 0.0
    test_predictions: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None


def train_classification(
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: TrainConfig,
    seed: int = 42,
) -> TrainResult:
    """Train a classification model and evaluate.

    Args:
        model: Model to train.
        train_dataset: Training data (will be split for validation).
        test_dataset: Held-out test data.
        config: Training hyperparameters.
        seed: Random seed.

    Returns:
        TrainResult with losses, metrics, and test predictions.
    """
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if config.device == "auto" else torch.device(config.device)
    model = model.to(device)

    # Split train into train/val
    n_val = int(len(train_dataset) * config.val_fraction)
    n_train = len(train_dataset) - n_val
    train_sub, val_sub = random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_sub, batch_size=config.batch_size,
                            pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             pin_memory=use_cuda)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        result.train_losses.append(epoch_loss / n_train)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / max(total, 1)
        result.val_losses.append(val_loss / max(n_val, 1))
        result.val_metrics.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={result.train_losses[-1]:.4f}, "
                  f"val_acc={val_acc:.4f}")

        if patience_counter >= config.patience:
            if config.verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    result.best_val_metric = best_val_acc
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    result.test_metric = (all_preds == all_labels).mean()
    result.test_predictions = all_preds
    result.test_labels = all_labels

    return result


def train_regression(
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    config: TrainConfig,
    seed: int = 42,
) -> TrainResult:
    """Train a regression model and evaluate.

    Uses MSE loss for training, R² for validation metric.

    Args:
        model: Model to train.
        train_dataset: Training data (will be split for validation).
        test_dataset: Held-out test data.
        config: Training hyperparameters.
        seed: Random seed.

    Returns:
        TrainResult with losses, metrics (R²), and test predictions.
    """
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if config.device == "auto" else torch.device(config.device)
    model = model.to(device)

    n_val = int(len(train_dataset) * config.val_fraction)
    n_train = len(train_dataset) - n_val
    train_sub, val_sub = random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_sub, batch_size=config.batch_size,
                            pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             pin_memory=use_cuda)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    result = TrainResult()
    best_val_r2 = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        result.train_losses.append(epoch_loss / n_train)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
                val_preds.append(pred.cpu())
                val_targets.append(y.cpu())

        result.val_losses.append(val_loss / max(n_val, 1))

        if len(val_preds) > 0:
            val_preds_cat = torch.cat(val_preds)
            val_targets_cat = torch.cat(val_targets)
            ss_res = ((val_preds_cat - val_targets_cat) ** 2).sum().item()
            ss_tot = ((val_targets_cat - val_targets_cat.mean()) ** 2).sum().item()
            val_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        else:
            val_r2 = 0.0
        result.val_metrics.append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if config.verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={result.train_losses[-1]:.4f}, "
                  f"val_R²={val_r2:.4f}")

        if patience_counter >= config.patience:
            if config.verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    result.best_val_metric = best_val_r2
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ss_res = ((all_preds - all_labels) ** 2).sum()
    ss_tot = ((all_labels - all_labels.mean()) ** 2).sum()
    result.test_metric = float(1.0 - ss_res / max(ss_tot, 1e-8))
    result.test_predictions = all_preds
    result.test_labels = all_labels

    return result


def compute_r2(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = ((predictions - labels) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def compute_mae(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.abs(predictions - labels).mean())
