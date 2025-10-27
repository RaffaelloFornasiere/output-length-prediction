"""
Generate data for predictable task: "print X times the word Y"
Extracts hidden states (last token, last layer) at each generation step.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

# Prompt templates
COUNTS = list(range(5, 6))  # X: how many times to print
WORDS = ["hello", "world", "cat", "dog", "python", "test", "apple", "blue", "sun", "code"]


def generate_prompt(count: int, word: str) -> str:
    """Generate a predictable task prompt."""
    prompt = f"""<|start_header_id|>system<|end_header_id|>
print {count} times the word "{word}". only this, no extra text
<|eot_id|><|start_header _id|>assistant<|end_header_id|>"""
    return prompt


def load_model():
    """Load model and tokenizer."""
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "mps" else torch.float32,
        device_map=DEVICE,
        token=HF_TOKEN,
    )
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_hidden_states(model, tokenizer, prompt: str):
    """
    Generate text and extract hidden states at each step.
    Returns lists of hidden states, remaining tokens, and token metadata.
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs.input_ids.shape[1]

    # Generate with hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,  # Deterministic
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get total generated tokens
    generated_ids = outputs.sequences[0][input_length:]
    total_generated_tokens = len(generated_ids)

    # Extract hidden states: outputs.hidden_states is tuple of tuples
    # Structure: (step_0, step_1, ..., step_N)
    # Each step: (layer_0, layer_1, ..., layer_L)
    hidden_states = []
    remaining_tokens_list = []
    token_metadata = []

    for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
        # Get last layer
        last_layer = step_hidden_states[-1]  # Shape: (batch=1, seq_len, hidden_dim)

        # Get last token's hidden state
        last_token_state = last_layer[0, -1, :].cpu().numpy()  # Shape: (hidden_dim,)

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

    return hidden_states, remaining_tokens_list, token_metadata


def main():
    """Generate dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model()

    # Storage
    all_hidden_states = []
    all_remaining_tokens = []
    all_token_metadata = []

    # Generate all combinations of counts and words
    prompts: list[str] = []
    for count in COUNTS:
        for word in WORDS:
            prompts.append(generate_prompt(count, word))

    print(f"Generating {len(prompts)} prompts (all combinations)...")

    for prompt in tqdm(prompts):
        try:
            hidden_states, remaining_tokens, token_metadata = extract_hidden_states(model, tokenizer, prompt)
            all_hidden_states.extend(hidden_states)
            all_remaining_tokens.extend(remaining_tokens)
            all_token_metadata.extend(token_metadata)

        except Exception as e:
            print(f"\nError on prompt '{prompt}': {e}")
            continue

    # Convert to numpy arrays
    print(f"\nProcessing {len(all_hidden_states)} data points...")
    hidden_states_array = np.array(all_hidden_states)
    remaining_tokens_array = np.array(all_remaining_tokens)

    # Create structured array for token metadata
    token_metadata_array = np.array(
        [(m['token_id'], m['token_text']) for m in all_token_metadata],
        dtype=[('token_id', 'i4'), ('token_text', 'U100')]  # U100 = Unicode string up to 100 chars
    )

    # split into train/val
    indices = np.arange(len(hidden_states_array))

    # 90/10 split
    split_idx = int(0.9 * len(indices))
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
    np.save(OUTPUT_DIR / "train_hidden_states.npy", train_hidden_states)
    np.save(OUTPUT_DIR / "train_remaining_tokens.npy", train_remaining_tokens)
    np.save(OUTPUT_DIR / "train_token_metadata.npy", train_token_metadata)

    np.save(OUTPUT_DIR / "val_hidden_states.npy", val_hidden_states)
    np.save(OUTPUT_DIR / "val_remaining_tokens.npy", val_remaining_tokens)
    np.save(OUTPUT_DIR / "val_token_metadata.npy", val_token_metadata)

    print(f"Saved to {OUTPUT_DIR}")
    print(f"Train: {train_hidden_states.shape}, Val: {val_hidden_states.shape}")


if __name__ == "__main__":
    main()