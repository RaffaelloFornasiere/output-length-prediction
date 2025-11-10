# MLP Probe Experiment

Multi-layer perceptron probe to predict remaining tokens from hidden states.

## Architecture

```python
class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Union[int, List[int]] = 512, dropout: float = 0.1):
        # Example: hidden_dims=[512, 256]
        # Layer 1: input_dim -> 512 -> ReLU -> Dropout
        # Layer 2: 512 -> 256 -> ReLU -> Dropout
        # Output: 256 -> 1
```

Configurable multi-layer network with:
- Arbitrary number of hidden layers (specify as single int or list)
- ReLU activations
- Dropout for regularization
- Linear output layer

## Hypothesis

Non-linear transformations might capture more complex patterns in the hidden states, potentially improving over the linear probe.

## Training

```bash
# Basic training (single hidden layer of 512)
python src/predictable/experiments/mlp/train.py

# Multi-layer architecture
python src/predictable/experiments/mlp/train.py --hidden_dim 512,256,128 --dropout 0.2

# Custom hyperparameters
python src/predictable/experiments/mlp/train.py --epochs 100 --batch_size 64 --lr 0.001 --hidden_dim 512,256 --dropout 0.1

# Use different dataset
python src/predictable/experiments/mlp/train.py --data_dir src/predictable/data_generation/00_same_prompt/data
```

### Hyperparameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: Hidden layer dimensions as comma-separated ints (default: "512")
  - Single layer: `512`
  - Multi-layer: `512,256` or `512,256,128`
- `--dropout`: Dropout probability (default: 0.1)
- `--data_dir`: Override data directory

## Testing

```bash
# Interactive testing (auto-selects best model)
python src/predictable/experiments/mlp/test_interactive.py

# With specific prompt
python src/predictable/experiments/mlp/test_interactive.py --prompt "Print ten times the word hello"

# Use specific checkpoint
python src/predictable/experiments/mlp/test_interactive.py --probe outputs/mlp_log_h512_256_d0.1_e100_bs64_lr0.001_spear0.988_r20.97.pt
```

## Results

### Best Model

- **Architecture**: [512, 256] with dropout=0.1
- **R² Score**: 0.97
- **Spearman Correlation**: 0.988
- **MAE**: ~2-3 tokens
- **Training**: 100 epochs, batch_size=64, lr=0.001

### Key Findings

1. **Comparable to linear**: MLP achieves similar performance to the linear probe (R²=0.97, Spearman=0.988), suggesting the relationship is largely linear.

2. **Diminishing returns**: Adding more layers or complexity doesn't significantly improve metrics beyond the simple linear probe.

3. **Early stopping**: Model tracks best validation loss and saves the best checkpoint (not just final epoch).

4. **Regularization matters**: Dropout (0.1-0.2) helps prevent overfitting, especially with larger architectures.

### Architecture Comparison

| Architecture | R² | Spearman | MAE | Notes |
|--------------|-----|----------|-----|-------|
| [512] | 0.96 | 0.985 | 2.8 | Single hidden layer |
| [512, 256] | 0.97 | 0.988 | 2.5 | **Best performance** |
| [512, 256, 128] | 0.96 | 0.987 | 2.6 | More complex, no improvement |

### Example Predictions

On "Print ten times the word hello":
- Step 1: Actual=45, Predicted=43.5, Error=1.5 (3.3%)
- Step 10: Actual=36, Predicted=35.3, Error=0.7 (1.9%)
- Step 20: Actual=26, Predicted=26.2, Error=0.2 (0.8%)

## Insights

The strong performance of both linear and MLP probes suggests:
1. **Length information is encoded linearly** in the hidden states
2. **Non-linearity provides minimal benefit** for this task
3. **Simple is sufficient** - a linear probe may be preferable due to simplicity and interpretability

## Files

- `train.py`: Training script with configurable MLP architecture
- `test_interactive.py`: Interactive testing with architecture parsing from filename
- `outputs/`: Trained model checkpoints
