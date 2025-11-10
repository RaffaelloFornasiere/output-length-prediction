# Experiments

This folder contains experiments for probing language model representations to predict output length.

## Structure

Each experiment is in its own numbered folder with the following structure:

```
e00_experiment_name/
├── README.md           # Experiment description and results
├── train.py           # Training script with probe architecture
├── test_interactive.py # Interactive testing script
└── outputs/           # Trained model checkpoints
```

Experiments are numbered with an `e` prefix (e00_, e01_, etc.) to maintain a clear ordering while allowing standard Python imports.

## Shared Utilities

Common functionality is shared across experiments:

- **`utils.py`**: Training utilities
  - `HiddenStatesDataset`: PyTorch dataset for hidden states and labels
  - `load_data()`: Load train/val splits from numpy files
  - `train_epoch()`: Single epoch training loop
  - `evaluate()`: Comprehensive evaluation with multiple metrics (MAE, RMSE, R², MAPE, Spearman)

- **`test_utils.py`**: Testing utilities
  - `find_best_probe()`: Auto-select best model based on R² from filename
  - `generate_with_predictions()`: Generate text and show predictions at each step

## Current Experiments

### e00_linear - Linear Probe
Simple linear probe: single linear layer mapping hidden states to predicted remaining tokens.

**Best results**: R² = 0.97, Spearman = 0.988

### e01_mlp - MLP Probe
Multi-layer perceptron with configurable hidden layers and dropout.

**Best results**: R² = 0.97, Spearman = 0.988

## Adding New Experiments

To add a new experiment:

1. Create a new numbered folder: `experiments/e0X_your_experiment_name/`
2. Create `train.py` with:
   - Your probe architecture (as a `nn.Module`)
   - Argparse setup for hyperparameters
   - Main training loop using shared utilities from `utils.py`
3. Create `test_interactive.py` with:
   - Model loading logic for your specific architecture
   - Direct import: `from predictable.experiments.e0X_your_experiment.train import YourProbe`
   - Use `find_best_probe()` and `generate_with_predictions()` from `test_utils.py`
4. Create `README.md` documenting:
   - What the experiment tests
   - How to run it
   - Results and insights
5. Create `outputs/` folder for model checkpoints

## Usage

### Training

```bash
# Linear probe
python src/predictable/experiments/e00_linear/train.py --epochs 100 --batch_size 64 --lr 0.001

# MLP probe
python src/predictable/experiments/e01_mlp/train.py --epochs 100 --batch_size 64 --lr 0.001 --hidden_dim 512,256 --dropout 0.1
```

### Interactive Testing

```bash
# Linear probe - auto-selects best model
python src/predictable/experiments/e00_linear/test_interactive.py

# MLP probe with specific prompt
python src/predictable/experiments/e01_mlp/test_interactive.py --prompt "Print ten times the word hello"

# Use a specific probe checkpoint
python src/predictable/experiments/e00_linear/test_interactive.py --probe outputs/linear_log_e100_bs64_lr0.001_spear0.988_r20.97.pt
```

## Data

All experiments currently use data from: `src/predictable/data_generation/01_different_prompts/data/`

You can override this with the `--data_dir` flag when training.

## Model Naming Convention

Trained models are saved with informative filenames encoding key hyperparameters and metrics:

**Linear**: `linear_log_e{epochs}_bs{batch_size}_lr{lr}_spear{spearman}_r2{r2}.pt`

**MLP**: `mlp_log_h{hidden_dims}_d{dropout}_e{epochs}_bs{batch_size}_lr{lr}_spear{spearman}_r2{r2}.pt`

This allows `find_best_probe()` to automatically select the best model based on R².
