"""
Linear probe for predicting remaining tokens from hidden states.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from scipy.stats import spearmanr


# Configuration
DATA_DIR = Path(__file__).parent.parent  / "data"
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


class LinearProbe(nn.Module):
    """Simple linear probe: single linear layer without activation."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


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


def main():
    """Train linear probe."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train linear probe for length prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data with log transformation
    use_log = True
    train_loader, val_loader, val_labels = load_data(args.batch_size, use_log=use_log)

    # Infer hidden_dim from data
    sample_batch = next(iter(train_loader))
    hidden_dim = sample_batch[0].shape[1]

    # Initialize model
    print(f"\nInitializing linear probe on {DEVICE}...")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Using log-space predictions: {use_log}")
    model = LinearProbe(hidden_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nTraining...")

    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, mae, rmse, r2, mape, rel_mae, mae_log, rmse_log, spearman = evaluate(model, val_loader, criterion, val_labels, use_log=use_log)

        # Print metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} MAE(log): {mae_log:.4f} | "
                f"MAE: {mae:.2f} R²: {r2:.4f} MAPE: {mape:.1f}% Spearman: {spearman:.4f}"
            )

    # Save final model with metrics
    final_val_loss, final_mae, final_rmse, final_r2, final_mape, final_rel_mae, final_mae_log, final_rmse_log, final_spearman = evaluate(model, val_loader, criterion, val_labels, use_log=use_log)
    filename = f"linear_log_e{args.epochs}_bs{args.batch_size}_lr{args.lr}_spear{final_spearman:.3f}_r2{final_r2:.2f}.pt"
    torch.save(model.state_dict(), OUTPUT_DIR / filename)

    print(f"\nTraining complete!")
    print(f"Final metrics:")
    print(f"  Log-space: MAE={final_mae_log:.4f}, RMSE={final_rmse_log:.4f}")
    print(f"  Original-space: MAE={final_mae:.2f}, MAPE={final_mape:.1f}%, R²={final_r2:.4f}")
    print(f"  Spearman correlation: {final_spearman:.4f}")
    print(f"Model saved to {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()