"""
Shared utilities for data generation across different datasets.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dotenv import load_dotenv
import os
from typing import Optional, List, Tuple, Dict, Any

from predictable.utils import apply_chat_template

# Load environment variables
load_dotenv()

# Configuration defaults
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_prompt(
    tokenizer,
    prompt_template: str,
    use_chat_template: bool = True,
    remove_date_lines: bool = True,
    **kwargs
) -> str:
    """
    Generate a prompt using the provided template and parameters.

    Args:
        tokenizer: The tokenizer to use for chat template
        prompt_template: The prompt template string with format placeholders
        use_chat_template: Whether to apply chat template formatting
        remove_date_lines: Whether to remove date lines from chat template
        **kwargs: Additional arguments to format the prompt template

    Returns:
        Formatted prompt string
    """
    # Format the prompt with provided kwargs
    prompt_content = prompt_template.format(**kwargs)

    if use_chat_template:
        messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        prompt = apply_chat_template(
            tokenizer,
            messages,
            remove_date_lines=remove_date_lines,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt
    else:
        return prompt_content


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer.

    Args:
        model_name: Name of the model to load (defaults to DEFAULT_MODEL)
        device: Device to load model on (defaults to DEVICE)
        dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or DEFAULT_MODEL
    device = device or DEVICE

    if dtype is None:
        dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Loading model {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        token=HF_TOKEN,
    )
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    layers_to_extract: Optional[List[int]] = None,
    extract_all_layers: bool = False
) -> Dict[str, Any]:
    """
    Generate text and extract hidden states at each step.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        layers_to_extract: Specific layer indices to extract (None = last layer only)
        extract_all_layers: If True, extract all layers (overrides layers_to_extract)

    Returns:
        Dictionary containing:
            - hidden_states: List of hidden states per token (shape depends on extraction settings)
            - remaining_tokens: List of remaining tokens at each step
            - token_metadata: List of dicts with token_id and token_text
            - total_generated: Total number of generated tokens
            - layers_info: Information about extracted layers
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate with hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False if temperature == 0.0 else True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get total generated tokens
    generated_ids = outputs.sequences[0][input_length:]
    total_generated_tokens = len(generated_ids)

    # Determine which layers to extract
    num_layers = len(outputs.hidden_states[0])  # Number of layers in the model

    if extract_all_layers:
        layers_to_extract = list(range(num_layers))
    elif layers_to_extract is None:
        layers_to_extract = [-1]  # Default to last layer only

    # Normalize layer indices (handle negative indexing)
    layers_to_extract = [l if l >= 0 else num_layers + l for l in layers_to_extract]

    # Extract hidden states
    hidden_states = []
    remaining_tokens_list = []
    token_metadata = []

    for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
        # Extract specified layers
        if len(layers_to_extract) == 1:
            # Single layer - maintain backward compatibility
            layer = step_hidden_states[layers_to_extract[0]]
            last_token_state = layer[0, -1, :].cpu().numpy()
        else:
            # Multiple layers - concatenate or keep separate
            layer_states = []
            for layer_idx in layers_to_extract:
                layer = step_hidden_states[layer_idx]
                layer_state = layer[0, -1, :].cpu().numpy()
                layer_states.append(layer_state)
            # Concatenate all layer states
            last_token_state = np.concatenate(layer_states)

        # Calculate remaining tokens
        tokens_generated_so_far = step_idx + 1
        remaining_tokens = total_generated_tokens - tokens_generated_so_far

        # Get token ID and decoded text
        token_id = generated_ids[step_idx].item()
        token_text = tokenizer.decode([token_id])

        hidden_states.append(last_token_state)
        remaining_tokens_list.append(remaining_tokens)
        token_metadata.append({
            'token_id': token_id,
            'token_text': token_text,
        })

    return {
        'hidden_states': hidden_states,
        'remaining_tokens': remaining_tokens_list,
        'token_metadata': token_metadata,
        'total_generated': total_generated_tokens,
        'layers_info': {
            'num_layers': num_layers,
            'extracted_layers': layers_to_extract,
            'hidden_dim_per_layer': hidden_states[0].shape[0] // len(layers_to_extract) if hidden_states else 0
        }
    }


def save_dataset(
    output_dir,
    hidden_states_list: List[np.ndarray],
    remaining_tokens_list: List[int],
    token_metadata_list: List[Dict],
    train_val_split: float = 0.9,
    additional_metadata: Optional[Dict] = None
) -> None:
    """
    Save dataset with train/validation split.

    Args:
        output_dir: Directory to save the data
        hidden_states_list: List of hidden state arrays
        remaining_tokens_list: List of remaining token counts
        token_metadata_list: List of token metadata dicts
        train_val_split: Fraction of data to use for training
        additional_metadata: Optional additional metadata to save
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays
    hidden_states_array = np.array(hidden_states_list)
    remaining_tokens_array = np.array(remaining_tokens_list)

    # Create structured array for token metadata
    token_metadata_array = np.array(
        [(m['token_id'], m['token_text']) for m in token_metadata_list],
        dtype=[('token_id', 'i4'), ('token_text', 'U100')]
    )

    # Create train/val split
    indices = np.arange(len(hidden_states_array))
    split_idx = int(train_val_split * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Split data
    train_hidden_states = hidden_states_array[train_indices]
    train_remaining_tokens = remaining_tokens_array[train_indices]
    train_token_metadata = token_metadata_array[train_indices]

    val_hidden_states = hidden_states_array[val_indices]
    val_remaining_tokens = remaining_tokens_array[val_indices]
    val_token_metadata = token_metadata_array[val_indices]

    # Save splits
    print(f"Saving train ({len(train_indices)}) and val ({len(val_indices)}) splits...")
    np.save(output_dir / "train_hidden_states.npy", train_hidden_states)
    np.save(output_dir / "train_remaining_tokens.npy", train_remaining_tokens)
    np.save(output_dir / "train_token_metadata.npy", train_token_metadata)

    np.save(output_dir / "val_hidden_states.npy", val_hidden_states)
    np.save(output_dir / "val_remaining_tokens.npy", val_remaining_tokens)
    np.save(output_dir / "val_token_metadata.npy", val_token_metadata)

    # Save additional metadata if provided
    if additional_metadata:
        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(additional_metadata, f, indent=2)

    print(f"Saved to {output_dir}")
    print(f"Train: {train_hidden_states.shape}, Val: {val_hidden_states.shape}")