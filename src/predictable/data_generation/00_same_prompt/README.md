# Dataset 00: Same Prompt

## Description
This dataset uses a **single prompt template** for all samples, varying only the count and word parameters. This creates a highly controlled dataset where the only variation comes from the parameters themselves, not from different ways of expressing the same instruction.

## Prompt Template
```
Print exactly {count} repetitions of the token "{word}". Do not include anything else.
```

## Parameters
- **Counts**: Range from 5 to 49 (45 different values)
- **Words**: 10 different words: hello, world, cat, dog, python, test, apple, blue, sun, code
- **Total samples**: 450 (45 counts × 10 words)

## Files

### `generate.py`
Generates the dataset using the shared utilities from `../utils.py`.
- Uses a single fixed prompt template
- Generates all combinations of counts and words
- Extracts hidden states from the last layer only
- Saves train/val split (90/10)

### `visualize.ipynb`
Jupyter notebook for exploring and visualizing the generated data:
- Distribution of remaining tokens
- Hidden state statistics
- Token frequency analysis
- Sample data inspection

### `data/` (generated)
Contains the generated dataset files after running `generate.py`:
- `train_hidden_states.npy`: Training hidden states (shape: [N, hidden_dim])
- `train_remaining_tokens.npy`: Training target values (shape: [N,])
- `train_token_metadata.npy`: Token IDs and text (structured array)
- `val_hidden_states.npy`: Validation hidden states
- `val_remaining_tokens.npy`: Validation target values
- `val_token_metadata.npy`: Validation token metadata
- `metadata.json`: Dataset metadata including parameters and statistics

## Usage

First ensure the package is installed:
```bash
# From project root
pip install -e .
```

Then generate the dataset:
```bash
# Option 1: From project root (recommended)
python -m predictable.data_generation.00_same_prompt.generate

# Option 2: From this directory
python generate.py
```

Visualize the data:
```bash
jupyter notebook visualize.ipynb
```

## Purpose
This dataset serves as a baseline where prompt variation is eliminated. It tests whether the model can learn output length prediction when the instruction format is completely consistent, isolating the effect of the count/word parameters from prompt phrasing variations.