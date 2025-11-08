# Dataset 02: Same Prompt Multi-Layers

## Description
This dataset uses the same single prompt template as dataset 00, but extracts hidden states from **ALL layers** instead of just the last layer. This allows analysis of how output length information is encoded at different depths of the model.

## Prompt Template
```
Print exactly {count} repetitions of the token "{word}". Do not include anything else.
```
(Same as dataset 00)

## Parameters
- **Counts**: Range from 5 to 49 (45 different values)
- **Words**: 10 different words: hello, world, cat, dog, python, test, apple, blue, sun, code
- **Total samples**: 450 (45 counts × 10 words)
- **Layers**: ALL model layers extracted and concatenated

## Configuration
- **Model**: `meta-llama/Llama-3.2-3B-Instruct` (configurable in `generate.py`)
- **Layers extracted**: All layers (for Llama-3.2-3B: 28 layers)
- **Hidden dimension per layer**: 3072
- **Total hidden dimension**: 3072 × 28 = 86,016 (concatenated)

## Files

### `generate.py`
Generates the dataset using the shared utilities from `../utils.py`.
- Uses the same single prompt template as dataset 00
- Generates all combinations of counts and words
- Extracts hidden states from **ALL layers** using `extract_all_layers=True`
- Concatenates all layer states into a single vector per token
- Saves train/val split (90/10)

### `visualize.ipynb`
Jupyter notebook for exploring and visualizing the generated data:
- Distribution of remaining tokens
- Hidden state statistics (concatenated and per-layer)
- Per-layer norm analysis
- Token frequency analysis
- Sample data inspection

### `data/` (generated)
Contains the generated dataset files after running `generate.py`:
- `train_hidden_states.npy`: Training hidden states (shape: [N, 86016] for Llama-3.2-3B)
- `train_remaining_tokens.npy`: Training target values (shape: [N,])
- `train_token_metadata.npy`: Token IDs and text (structured array)
- `val_hidden_states.npy`: Validation hidden states
- `val_remaining_tokens.npy`: Validation target values
- `val_token_metadata.npy`: Validation token metadata
- `metadata.json`: Dataset metadata including layer information

## Usage

First ensure the package is installed:
```bash
# From project root
pip install -e .
```

Then generate the dataset:
```bash
# Option 1: From project root (recommended)
python -m predictable.data_generation.02_same_prompt_multi_layers.generate

# Option 2: From this directory
python generate.py
```

Visualize the data:
```bash
jupyter notebook visualize.ipynb
```

## Purpose
This dataset enables analysis of how output length information is distributed across model layers:
- **Early layers** (0-10): May encode more syntactic/structural information
- **Middle layers** (11-18): May capture semantic relationships
- **Late layers** (19-27): May contain more task-specific representations

By comparing probes trained on:
1. Just the last layer (dataset 00)
2. All layers concatenated (dataset 02)
3. Individual layers (future analysis)

We can understand where length awareness emerges in the model architecture.

## Notes
- **Large file size**: All layers means ~28x larger files than dataset 00
- **Training considerations**: Probes will need more capacity or regularization due to high dimensionality
- **Layer analysis**: The visualization notebook includes per-layer analysis to identify which layers contribute most to length prediction