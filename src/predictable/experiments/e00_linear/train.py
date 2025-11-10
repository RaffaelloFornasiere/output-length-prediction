"""
Train linear probe for predicting remaining tokens from hidden states.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse

from predictable.experiments.utils import load_data, train_epoch, evaluate


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data_generation" / "01_different_prompts" / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class LinearProbe(nn.Module):
    """Simple linear probe: single linear layer without activation."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    """Train linear probe."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train linear probe for length prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=None, help="Override default data directory")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine data directory
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Load data with log transformation
    use_log = True
    train_loader, val_loader, val_labels = load_data(data_dir, args.batch_size, use_log=use_log)

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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, mae, rmse, r2, mape, rel_mae, mae_log, rmse_log, spearman = evaluate(
            model, val_loader, criterion, val_labels, DEVICE, use_log=use_log
        )

        # Print metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} MAE(log): {mae_log:.4f} | "
                f"MAE: {mae:.2f} R²: {r2:.4f} MAPE: {mape:.1f}% Spearman: {spearman:.4f}"
            )

    # Save final model with metrics
    final_val_loss, final_mae, final_rmse, final_r2, final_mape, final_rel_mae, final_mae_log, final_rmse_log, final_spearman = evaluate(
        model, val_loader, criterion, val_labels, DEVICE, use_log=use_log
    )
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
