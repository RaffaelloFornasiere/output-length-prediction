"""
MLP probe for predicting remaining tokens from hidden states.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from scipy.stats import spearmanr
from typing import Union, List


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data_generation"  / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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


class MLPProbe(nn.Module):
    """MLP probe with configurable hidden layers and dropout."""

    def __init__(self, input_dim: int, hidden_dims: Union[int, List[int]] = 512, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input hidden states
            hidden_dims: Either a single int for one hidden layer, or list of ints for multiple layers
            dropout: Dropout probability
        """
        super().__init__()

        # Convert single int to list for uniform handling
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_data(batch_size, use_log=True):
    """Load train and val splits."""
    print("Loading data...")
    train_hidden = np.load(DATA_DIR / "train_hidden_states.npy")
    train_labels = np.load(DATA_DIR / "train_remaining_tokens.npy")
    val_hidden = np.load(DATA_DIR / "val_hidden_states.npy")
    val_labels = np.load(DATA_DIR / "val_remaining_tokens.npy")

    print(f"Train: {train_hidden.shape}, Val: {val_hidden.shape}")

    train_dataset = HiddenStatesDataset(train_hidden, train_labels, use_log=use_log)
    val_dataset = HiddenStatesDataset(val_hidden, val_labels, use_log=use_log)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Store original labels for metric computation
    return train_loader, val_loader, val_labels


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for hidden_states, labels in train_loader:
        hidden_states = hidden_states.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        predictions = model(hidden_states)
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, original_labels, use_log=True):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    all_predictions_log = []
    all_labels_log = []

    with torch.no_grad():
        for hidden_states, labels in val_loader:
            hidden_states = hidden_states.to(DEVICE)
            labels = labels.to(DEVICE)

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


def parse_hidden_dims(hidden_dims_str: str) -> List[int]:
    """Parse hidden_dims argument from string.

    Examples:
        "512" -> [512]
        "512,256" -> [512, 256]
        "512,256,128" -> [512, 256, 128]
    """
    if ',' in hidden_dims_str:
        # Parse as comma-separated list
        return [int(x.strip()) for x in hidden_dims_str.split(',')]
    else:
        # Single integer
        return [int(hidden_dims_str)]


def main():
    """Train MLP probe."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train MLP probe for length prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=str, default="512", help="Hidden layer dimensions. Single int (e.g., '512') or comma-separated (e.g., '512,256,128')")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse hidden dimensions
    hidden_dims = parse_hidden_dims(args.hidden_dim)

    # Load data with log transformation
    use_log = True
    train_loader, val_loader, val_labels = load_data(args.batch_size, use_log=use_log)

    # Infer input_dim from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]

    # Initialize model
    print(f"\nInitializing MLP probe on {DEVICE}...")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Dropout: {args.dropout}")
    print(f"Using log-space predictions: {use_log}")

    model = MLPProbe(input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nTraining...")

    best_val_loss = float('inf')
    best_state = None

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, mae, rmse, r2, mape, rel_mae, mae_log, rmse_log, spearman = evaluate(model, val_loader, criterion, val_labels, use_log=use_log)

        # Track best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                'state_dict': model.state_dict(),
                'metrics': {
                    'val_loss': val_loss,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'rel_mae': rel_mae,
                    'mae_log': mae_log,
                    'rmse_log': rmse_log,
                    'spearman': spearman
                }
            }

        # Print metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} MAE(log): {mae_log:.4f} | "
                f"MAE: {mae:.2f} R²: {r2:.4f} MAPE: {mape:.1f}% Spearman: {spearman:.4f}"
            )

    # Use best model metrics for filename
    final_metrics = best_state['metrics']
    final_spearman = final_metrics['spearman']
    final_r2 = final_metrics['r2']

    # Create hidden dims string for filename
    hidden_str = '_'.join(map(str, hidden_dims))

    filename = f"mlp_log_h{hidden_str}_d{args.dropout}_e{args.epochs}_bs{args.batch_size}_lr{args.lr}_spear{final_spearman:.3f}_r2{final_r2:.2f}.pt"
    torch.save(best_state['state_dict'], OUTPUT_DIR / filename)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model metrics:")
    print(f"  Log-space: MAE={final_metrics['mae_log']:.4f}, RMSE={final_metrics['rmse_log']:.4f}")
    print(f"  Original-space: MAE={final_metrics['mae']:.2f}, MAPE={final_metrics['mape']:.1f}%, R²={final_metrics['r2']:.4f}")
    print(f"  Spearman correlation: {final_metrics['spearman']:.4f}")
    print(f"Model saved to {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
