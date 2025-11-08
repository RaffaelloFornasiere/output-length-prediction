# Data Generation

This folder contains the infrastructure for generating datasets to probe language models' awareness of output length during generation.

## Structure

### Shared Utilities
- **`utils.py`**: Shared functions for all datasets
  - `generate_prompt()`: Create prompts with optional chat template formatting
  - `load_model()`: Load model and tokenizer with configurable settings
  - `extract_hidden_states()`: Generate text and extract hidden states at each step
  - `save_dataset()`: Save data with train/val split and metadata

- **`prompts.py`**: Collection of 52 different prompt templates for the repetition task

### Datasets

Each dataset follows a consistent structure:
```
XX_dataset_name/
├── README.md           # Dataset description
├── generate.py         # Generation script
├── visualize.ipynb    # Visualization notebook
└── data/              # Generated data (after running generate.py)
    ├── train_hidden_states.npy
    ├── train_remaining_tokens.npy
    ├── train_token_metadata.npy
    ├── val_hidden_states.npy
    ├── val_remaining_tokens.npy
    ├── val_token_metadata.npy
    └── metadata.json
```

### Current Datasets

#### 00_same_prompt
- Uses a **single prompt template** for all samples
- Tests baseline performance with no prompt variation
- Isolates the effect of count/word parameters

#### 01_different_prompts ✓ (data exists)
- Uses **52 different prompt templates** randomly
- Tests robustness to prompt variation
- Contains the original generated data (~49k tokens)

#### 02_same_prompt_multi_layers 🚧 (template created)
- Will extract hidden states from **multiple layers**
- Tests if different layers encode length information differently
- Enables analysis of how length awareness develops through the model
- Structure created, implementation pending

## Setup

First, install the package in development mode from the project root:
```bash
# From /Users/forna/Documents/me/output-length-prediction/
pip install -e .
```

This makes the `predictable` package available for imports.

## Usage

### Generate a Dataset

**Option 1: Run as a module from project root (recommended)**
```bash
python -m predictable.data_generation.00_same_prompt.generate
```

**Option 2: Run directly from dataset folder**
```bash
cd src/predictable/data_generation/00_same_prompt
python generate.py
```

Both options work after installing the package with `pip install -e .`

### Visualize Data
```bash
cd src/predictable/data_generation/XX_dataset_name
jupyter notebook visualize.ipynb
```

### Use Shared Utilities
```python
from predictable.data_generation.utils import (
    generate_prompt,
    load_model,
    extract_hidden_states,
    save_dataset
)

# Example: Generate with custom settings
model, tokenizer = load_model(model_name="meta-llama/Llama-3.2-3B-Instruct")

# Generate prompt (date lines removed by default)
prompt = generate_prompt(
    tokenizer,
    "Repeat {word} {count} times",
    word="hello",
    count=10,
    remove_date_lines=True  # Default: True
)

# Extract hidden states
result = extract_hidden_states(
    model, tokenizer, prompt,
    layers_to_extract=[-1, -2, -3]  # Last 3 layers
)
```

Alternatively, use the base `apply_chat_template` utility:
```python
from predictable.utils import apply_chat_template

messages = [{"role": "user", "content": "Hello!"}]
prompt = apply_chat_template(
    tokenizer,
    messages,
    remove_date_lines=True  # Removes date-related lines from chat template
)
```

## Data Format

All datasets share the same output format:

- **Hidden States**: NumPy array of shape `[N, hidden_dim]` or `[N, hidden_dim * num_layers]`
- **Remaining Tokens**: NumPy array of shape `[N,]` with target values
- **Token Metadata**: Structured array with `token_id` and `token_text` fields
- **Metadata JSON**: Dataset parameters, statistics, and generation settings

## Adding New Datasets

To add a new dataset:

1. Create a new folder: `XX_descriptive_name/`
2. Copy the structure from an existing dataset
3. Modify `generate.py` to implement your specific generation logic
4. Update the README with dataset details
5. Ensure backward compatibility when modifying `utils.py`

## Model Configuration

Each dataset can use a different model by setting `MODEL_NAME` in its `generate.py`.

Current configuration:
- **00_same_prompt**: `meta-llama/Llama-3.2-3B-Instruct`
- **01_different_prompts**: `meta-llama/Llama-3.2-3B-Instruct`
- **02_same_prompt_multi_layers**: (planned) - can use any model

Model properties (Llama-3.2-3B):
- Hidden dimension: 3072
- Device: MPS (Apple Silicon) or CPU
- Requires HF_TOKEN in `.env` file

To use a different model in a dataset, simply change the `MODEL_NAME` variable in that dataset's `generate.py`.

## Dependencies

- transformers
- torch
- numpy
- tqdm
- python-dotenv
- matplotlib (for visualization)
- jupyter (for notebooks)