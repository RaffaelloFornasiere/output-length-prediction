"""
Shared utilities for training probes across experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from typing import Dict, List, Callable


class LogMSELoss(nn.Module):
    """MSE loss computed in log space: (log(pred + 1) - log(true + 1))^2"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions in original space
            targets: True values in original space

        Returns:
            MSE loss in log space
        """
        log_pred = torch.log(predictions + 1)
        log_target = torch.log(targets + 1)
        return torch.mean((log_pred - log_target) ** 2)


class HiddenStatesDataset(Dataset):
    """Dataset for hidden states and remaining tokens."""

    def __init__(self, hidden_states: np.ndarray, remaining_tokens: np.ndarray):
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.remaining_tokens = torch.FloatTensor(remaining_tokens).unsqueeze(1)

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.remaining_tokens[idx]


def load_data(data_dir: Path, batch_size: int):
    """Load train and val splits.

    Args:
        data_dir: Directory containing the numpy data files
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, val_labels (original space)
    """
    print("Loading data...")
    train_hidden = np.load(data_dir / "train_hidden_states.npy")
    train_labels = np.load(data_dir / "train_remaining_tokens.npy")
    val_hidden = np.load(data_dir / "val_hidden_states.npy")
    val_labels = np.load(data_dir / "val_remaining_tokens.npy")

    print(f"Train: {train_hidden.shape}, Val: {val_hidden.shape}")

    train_dataset = HiddenStatesDataset(train_hidden, train_labels)
    val_dataset = HiddenStatesDataset(val_hidden, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Store original labels for metric computation
    return train_loader, val_loader, val_labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.

    Args:
        model: The probe model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0

    for hidden_states, labels in train_loader:
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(hidden_states)
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================================
# Metric Functions
# ============================================================================

class Metric:
    """Base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        raise NotImplementedError


class MAE(Metric):
    """Mean Absolute Error in original space."""

    def __init__(self):
        super().__init__("mae")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']
        return torch.mean(torch.abs(predictions - labels)).item()


class RMSE(Metric):
    """Root Mean Squared Error in original space."""

    def __init__(self):
        super().__init__("rmse")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']
        return torch.sqrt(torch.mean((predictions - labels) ** 2)).item()


class MAELog(Metric):
    """Mean Absolute Error in log space."""

    def __init__(self):
        super().__init__("mae_log")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        log_predictions = metric_inputs['log_predictions']
        log_labels = metric_inputs['log_labels']
        return torch.mean(torch.abs(log_predictions - log_labels)).item()


class RMSELog(Metric):
    """Root Mean Squared Error in log space."""

    def __init__(self):
        super().__init__("rmse_log")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        log_predictions = metric_inputs['log_predictions']
        log_labels = metric_inputs['log_labels']
        return torch.sqrt(torch.mean((log_predictions - log_labels) ** 2)).item()


class MAPE(Metric):
    """Mean Absolute Percentage Error (only for samples where label > threshold)."""

    def __init__(self, threshold: float = 5.0):
        super().__init__("mape")
        self.threshold = threshold

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']

        mask = labels.squeeze() > self.threshold
        if mask.sum() == 0:
            return 0.0

        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        relative_errors = torch.abs(filtered_predictions - filtered_labels) / torch.abs(filtered_labels)
        return (torch.mean(relative_errors).item() * 100)


class RelativeMAE(Metric):
    """Relative Mean Absolute Error (only for samples where label > threshold)."""

    def __init__(self, threshold: float = 5.0):
        super().__init__("rel_mae")
        self.threshold = threshold

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']

        mask = labels.squeeze() > self.threshold
        if mask.sum() == 0:
            return 0.0

        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        relative_errors = torch.abs(filtered_predictions - filtered_labels) / torch.abs(filtered_labels)
        return torch.mean(relative_errors).item()


class R2Score(Metric):
    """R² (coefficient of determination) in original space."""

    def __init__(self):
        super().__init__("r2")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']

        ss_res = torch.sum((labels - predictions) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()


class SpearmanCorrelation(Metric):
    """Spearman rank correlation coefficient."""

    def __init__(self):
        super().__init__("spearman")

    def __call__(self, metric_inputs: Dict[str, torch.Tensor]) -> float:
        predictions = metric_inputs['predictions']
        labels = metric_inputs['labels']

        corr, _ = spearmanr(
            predictions.squeeze().numpy(),
            labels.squeeze().numpy()
        )
        return corr


# Default metrics to use if none specified
DEFAULT_METRICS = [
    MAE(),
    RMSE(),
    MAELog(),
    RMSELog(),
    MAPE(),
    RelativeMAE(),
    R2Score(),
    SpearmanCorrelation(),
]


def evaluate(model, val_loader, criterion, device, metrics: List[Metric] = None):
    """Evaluate on validation set.

    Args:
        model: The probe model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        metrics: List of metric functions to compute. If None, uses DEFAULT_METRICS.

    Returns:
        Tuple of (val_loss, metrics_dict) where metrics_dict maps metric names to values
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for hidden_states, labels in val_loader:
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)

            predictions = model(hidden_states)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels (both in original space)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Precompute common inputs for metrics
    metric_inputs = {
        'predictions': all_predictions,
        'labels': all_labels,
        'log_predictions': torch.log(all_predictions + 1),
        'log_labels': torch.log(all_labels + 1),
    }

    # Compute all metrics
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.name] = metric(metric_inputs)

    return total_loss / len(val_loader), metrics_dict
