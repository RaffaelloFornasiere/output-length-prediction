# Dataset 02: Same Prompt Multi-Layers

## Description
This dataset uses the same single prompt template as dataset 00, but extracts hidden states from **multiple layers** instead of just the last layer. This allows analysis of how output length information is encoded at different depths of the model.

## Configuration
- **Model**: Can be configured in `generate.py` (default: `meta-llama/Llama-3.2-3B-Instruct`)
- **Prompt**: Same single template as dataset 00
- **Layers**: Extracts from multiple layers (e.g., last 3 layers: [-1, -2, -3])
- **Hidden dimension**: Will be `hidden_dim * num_layers` (e.g., 3072 * 3 = 9216 for 3 layers)

## Purpose
Tests whether different layers encode output length information differently:
- Early layers: May encode more syntactic/structural information
- Middle layers: May capture semantic relationships
- Late layers: May contain more task-specific representations

## Status
**Planned** - Template structure created. To implement:
1. Copy `generate.py` from `00_same_prompt`
2. Modify to use `layers_to_extract=[-1, -2, -3]` or `extract_all_layers=True`
3. Update metadata to reflect multi-layer extraction

## Expected Usage
```python
# In generate.py:
result = extract_hidden_states(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    layers_to_extract=[-1, -2, -3]  # Last 3 layers
    # or: extract_all_layers=True    # All layers
)
```