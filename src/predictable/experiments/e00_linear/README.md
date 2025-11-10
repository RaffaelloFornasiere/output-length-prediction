# Linear Probe Experiment

A simple linear probe to predict remaining tokens from hidden states.

## Architecture

```python
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
```

Single linear layer mapping from hidden state dimension (3072 for Llama-3.2-3B) to a scalar prediction of remaining tokens.

## Hypothesis

Even a simple linear transformation should be able to extract length information if it's linearly encoded in the hidden states.

## Training

```bash
# Basic training
python src/predictable/experiments/linear/train.py

# Custom hyperparameters
python src/predictable/experiments/linear/train.py --epochs 100 --batch_size 128 --lr 0.0001

# Use different dataset
python src/predictable/experiments/linear/train.py --data_dir src/predictable/data_generation/00_same_prompt/data
```

### Hyperparameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--data_dir`: Override data directory

## Testing

```bash
# Interactive testing (auto-selects best model)
python src/predictable/experiments/linear/test_interactive.py

# With specific prompt
python src/predictable/experiments/linear/test_interactive.py --prompt "Print ten times the word hello"

# Use specific checkpoint
python src/predictable/experiments/linear/test_interactive.py --probe outputs/linear_log_e100_bs64_lr0.001_spear0.988_r20.97.pt
```

## Results

### Best Model

- **R² Score**: 0.97
- **Spearman Correlation**: 0.988
- **MAE**: ~2-3 tokens
- **Training**: 100 epochs, batch_size=64, lr=0.001

### Key Findings

1. **Surprisingly effective**: A single linear layer achieves very strong performance, suggesting length information is highly linearly separable in the hidden states.

2. **Log-space helps**: Training with log-transformed targets (`log(remaining + 1)`) significantly improves performance over raw token counts.

3. **Strong correlation**: High Spearman correlation (0.988) indicates the model correctly orders predictions by length, even when absolute values are slightly off.

4. **Quick convergence**: Model typically converges within 50-100 epochs.

### Example Predictions

On "Print ten times the word hello":
- Step 1: Actual=45, Predicted=43.2, Error=1.8 (4.0%)
- Step 10: Actual=36, Predicted=35.1, Error=0.9 (2.5%)
- Step 20: Actual=26, Predicted=26.8, Error=0.8 (3.1%)

## Files

- `train.py`: Training script
- `test_interactive.py`: Interactive testing with step-by-step predictions
- `outputs/`: Trained model checkpoints
