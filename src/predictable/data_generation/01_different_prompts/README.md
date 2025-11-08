# Dataset 01: Different Prompts

## Description
This dataset uses **multiple different prompt templates** (52 variations) for the same task, randomly selecting a template for each sample. This creates a more diverse dataset that tests whether the model can learn output length prediction despite variations in how the instruction is phrased.

## Prompt Templates
The dataset uses 52 different prompt templates from `../prompts.py`, including variations like:
- "Print exactly {count} repetitions of the token \"{word}\". Do not include anything else."
- "Output \"{word}\" exactly {count} times and nothing more."
- "Your response must consist of {count} copies of \"{word}\", with no surrounding text."
- ... and 49 more variations

Each template expresses the same core instruction differently, testing robustness to phrasing.

## Parameters
- **Counts**: Range from 5 to 49 (45 different values)
- **Words**: 10 different words: hello, world, cat, dog, python, test, apple, blue, sun, code
- **Prompt templates**: 52 different templates, randomly selected
- **Total samples**: 450 (45 counts × 10 words)

## Files

### `generate.py`
Generates the dataset using the shared utilities from `../utils.py`.
- Randomly selects from 52 different prompt templates
- Generates all combinations of counts and words
- Extracts hidden states from the last layer only
- Saves train/val split (90/10)
- Tracks prompt usage distribution in metadata

### `visualize.ipynb`
Jupyter notebook for exploring and visualizing the generated data:
- Distribution of remaining tokens
- Hidden state statistics
- Token frequency analysis
- Sample data inspection
- Sequence analysis

### `data/` (contains existing data)
Contains the dataset files:
- `train_hidden_states.npy`: Training hidden states (shape: [44544, 3072])
- `train_remaining_tokens.npy`: Training target values (shape: [44544,])
- `train_token_metadata.npy`: Token IDs and text (structured array)
- `val_hidden_states.npy`: Validation hidden states (shape: [4949, 3072])
- `val_remaining_tokens.npy`: Validation target values (shape: [4949,])
- `val_token_metadata.npy`: Validation token metadata
- `metadata.json`: Dataset metadata (if regenerated)

## Existing Data
This folder contains the original generated data that was previously in `../data/`. The data was generated with:
- Model: meta-llama/Llama-3.2-3B-Instruct
- Hidden dimension: 3072
- Total tokens: ~49,500 across train and validation

## Usage

First ensure the package is installed:
```bash
# From project root
pip install -e .
```

Regenerate the dataset (optional - data already exists):
```bash
# Option 1: From project root (recommended)
python -m predictable.data_generation.01_different_prompts.generate

# Option 2: From this directory
python generate.py
```

Visualize the data:
```bash
jupyter notebook visualize.ipynb
```

## Purpose
This dataset tests whether the model can learn output length prediction when the same instruction is expressed in many different ways. It evaluates robustness to prompt variation and tests if the model can extract the core semantic meaning (count and word) regardless of phrasing.