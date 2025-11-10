# Running Experiments

This guide explains how to set up and run experiments in this project.

## Setup

First, ensure all dependencies are installed:

```bash
uv sync
```

This will install all required packages and create a virtual environment.

## Running Experiments

Each experiment is self-contained in its own directory under `src/predictable/experiments/`.

### Training a Model

Navigate to an experiment directory and run the training script:

```bash
# Example: Linear experiment
python src/predictable/experiments/e00_linear/train.py

# Example: MLP experiment
python src/predictable/experiments/e01_mlp/train.py
```

The training script will:
- Load or generate the necessary dataset
- Train the model
- Save the trained model to disk

### Testing Interactively

After training, you can test the model interactively:

```bash
# Example: Linear experiment
python src/predictable/experiments/e00_linear/test_interactive.py

# Example: MLP experiment
python src/predictable/experiments/e01_mlp/test_interactive.py
```

The interactive test will:
- Load the trained model
- Run predictions on test data
- Display results and metrics

## Experiment Structure

Each experiment typically contains:
- `train.py` - Model training script
- `test_interactive.py` - Interactive testing and evaluation
- Other supporting files as needed

## Notes

- Make sure to run `train.py` before `test_interactive.py` for each experiment
- Models are saved automatically during training
- Check individual experiment directories for experiment-specific details