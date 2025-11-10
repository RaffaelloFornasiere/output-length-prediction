"""
Shared utilities for training probes across experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr


class HiddenStatesDataset(Dataset):
    """Dataset for hidden states and remaining tokens."""

    def __init__(self, hidden_states: np.ndarray, remaining_tokens: np.ndarray, use_log: bool = True):
        self.hidden_states = torch.FloatTensor(hidden_states)
        # Transform to log space: log(remaining + 1)
        if use_log:
            self.remaining_tokens = torch.log(torch.FloatTensor(remaining_tokens) + 1).unsqueeze(1)
        else:
            self.remaining_tokens = torch.FloatTensor(remaining_tokens).unsqueeze(1)
        self.use_log = use_log

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.remaining_tokens[idx]


def load_data(data_dir: Path, batch_size: int, use_log: bool = True):
    """Load train and val splits.

    Args:
        data_dir: Directory containing the numpy data files
        batch_size: Batch size for dataloaders
        use_log: Whether to use log-space transformation

    Returns:
        train_loader, val_loader, val_labels (original space)
    """
    print("Loading data...")
    train_hidden = np.load(data_dir / "train_hidden_states.npy")
    train_labels = np.load(data_dir / "train_remaining_tokens.npy")
    val_hidden = np.load(data_dir / "val_hidden_states.npy")
    val_labels = np.load(data_dir / "val_remaining_tokens.npy")

    print(f"Train: {train_hidden.shape}, Val: {val_hidden.shape}")

    train_dataset = HiddenStatesDataset(train_hidden, train_labels, use_log=use_log)
    val_dataset = HiddenStatesDataset(val_hidden, val_labels, use_log=use_log)

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


def evaluate(model, val_loader, criterion, original_labels, device, use_log: bool = True):
    """Evaluate on validation set.

    Args:
        model: The probe model
        val_loader: Validation data loader
        criterion: Loss function
        original_labels: Original labels (not log-transformed) for computing metrics
        device: Device to evaluate on
        use_log: Whether the model predicts in log-space

    Returns:
        Tuple of (val_loss, mae, rmse, r2, mape, rel_mae, mae_log, rmse_log, spearman_corr)
    """
    model.eval()
    total_loss = 0
    all_predictions_log = []
    all_labels_log = []

    with torch.no_grad():
        for hidden_states, labels in val_loader:
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)

            predictions = model(hidden_states)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_predictions_log.append(predictions.cpu())
            all_labels_log.append(labels.cpu())

    # Concatenate all predictions and labels
    all_predictions_log = torch.cat(all_predictions_log)
    all_labels_log = torch.cat(all_labels_log)

    # Log-space metrics
    mae_log = torch.mean(torch.abs(all_predictions_log - all_labels_log)).item()
    rmse_log = torch.sqrt(torch.mean((all_predictions_log - all_labels_log) ** 2)).item()

    # Convert back from log space to original space
    if use_log:
        all_predictions = torch.exp(all_predictions_log) - 1
        all_labels = torch.FloatTensor(original_labels).unsqueeze(1)
    else:
        all_predictions = all_predictions_log
        all_labels = all_labels_log

    # Absolute metrics (in original space)
    mae = torch.mean(torch.abs(all_predictions - all_labels)).item()
    rmse = torch.sqrt(torch.mean((all_predictions - all_labels) ** 2)).item()

    # Relative metrics - only compute for samples where actual > 5 to avoid division issues
    mask = all_labels.squeeze() > 5
    if mask.sum() > 0:
        filtered_predictions = all_predictions[mask]
        filtered_labels = all_labels[mask]
        relative_errors = torch.abs(filtered_predictions - filtered_labels) / torch.abs(filtered_labels)
        mape = torch.mean(relative_errors).item() * 100  # Mean Absolute Percentage Error (%)
        rel_mae = torch.mean(relative_errors).item()
    else:
        mape = 0.0
        rel_mae = 0.0

    # R² score (in original space)
    ss_res = torch.sum((all_labels - all_predictions) ** 2)
    ss_tot = torch.sum((all_labels - torch.mean(all_labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Spearman correlation (rank-based, measures ordering)
    spearman_corr, _ = spearmanr(
        all_predictions.squeeze().numpy(),
        all_labels.squeeze().numpy()
    )

    return total_loss / len(val_loader), mae, rmse, r2.item(), mape, rel_mae, mae_log, rmse_log, spearman_corr
