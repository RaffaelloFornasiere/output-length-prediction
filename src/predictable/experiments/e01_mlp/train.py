"""
Train MLP probe for predicting remaining tokens from hidden states.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Union, List

from predictable.experiments.utils import load_data, train_epoch, evaluate


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data_generation" / "01_different_prompts" / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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
    parser.add_argument("--data_dir", type=str, default=None, help="Override default data directory")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse hidden dimensions
    hidden_dims = parse_hidden_dims(args.hidden_dim)

    # Determine data directory
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Load data with log transformation
    use_log = True
    train_loader, val_loader, val_labels = load_data(data_dir, args.batch_size, use_log=use_log)

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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, mae, rmse, r2, mape, rel_mae, mae_log, rmse_log, spearman = evaluate(
            model, val_loader, criterion, val_labels, DEVICE, use_log=use_log
        )

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
