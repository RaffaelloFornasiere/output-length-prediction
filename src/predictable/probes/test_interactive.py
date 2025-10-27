"""
Interactive testing script for linear probe.
Loads trained probe and Llama 3.2 3B, generates tokens step-by-step,
and shows predicted remaining tokens at each generation step.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from predictable.probes.linear import LinearProbe

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROBE_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


def find_best_probe() -> Path:
    """Find the best trained probe based on R² score from filename."""
    probe_files = list(PROBE_DIR.glob("linear_*.pt"))

    if not probe_files:
        raise FileNotFoundError(f"No probe models found in {PROBE_DIR}")

    best_probe = None
    best_r2 = -float('inf')

    for probe_file in probe_files:
        # Parse R² from filename: linear_e50_bs64_lr0.001_mae19.00_r20.79.pt
        try:
            parts = probe_file.stem.split('_')
            r2_part = [p for p in parts if p.startswith('r2')][0]
            r2_value = float(r2_part[2:])

            if r2_value > best_r2:
                best_r2 = r2_value
                best_probe = probe_file
        except (IndexError, ValueError):
            continue

    if best_probe is None:
        raise ValueError("Could not parse R² from any probe filenames")

    print(f"Auto-selected best probe: {best_probe.name} (R²={best_r2:.4f})")
    return best_probe


def load_models(probe_path: str):
    """Load both the language model and the trained probe."""
    print(f"Loading Llama 3.2 3B on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # Keep float32 for MPS compatibility
        device_map=DEVICE,
        token=HF_TOKEN,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading probe from {probe_path}...")
    # Infer hidden_dim from model config
    hidden_dim = model.config.hidden_size
    probe = LinearProbe(hidden_dim).to(DEVICE)
    probe.load_state_dict(torch.load(probe_path, map_location=DEVICE, weights_only=True))
    probe.eval()

    return model, tokenizer, probe


def generate_with_predictions(model, tokenizer, probe, prompt: str, max_new_tokens: int = 200):
    """
    Generate tokens using model.generate() and show predictions at each step.
    Exactly matches the training data generation approach.
    """
    print(f"\n{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs.input_ids.shape[1]

    # Generate with hidden states - SAME AS TRAINING
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Deterministic - SAME AS TRAINING
            do_sample=False,  # SAME AS TRAINING
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get generated tokens
    generated_ids = outputs.sequences[0][input_length:]
    total_generated_tokens = len(generated_ids)

    print(f"{'Step':<6} | {'Token':<20} | {'Actual Remaining':<18} | {'Predicted Remaining':<20} | {'Error':<10}")
    print("-" * 80)

    # Iterate through each generation step - SAME AS TRAINING
    for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
        # Get last layer's hidden state
        last_layer = step_hidden_states[-1]  # Shape: (batch=1, seq_len, hidden_dim)
        last_token_hidden = last_layer[0, -1, :]  # Shape: (hidden_dim,)

        # Predict remaining tokens with probe
        predicted_remaining = probe(last_token_hidden.unsqueeze(0)).item()

        # Calculate actual remaining tokens
        tokens_generated_so_far = step_idx + 1
        actual_remaining = total_generated_tokens - tokens_generated_so_far

        # Get the token that was generated at this step
        token_id = generated_ids[step_idx]
        token_text = tokenizer.decode(token_id)

        # Calculate error
        error = abs(predicted_remaining - actual_remaining)

        print(f"{step_idx+1:<6} | {token_text:<20} | {actual_remaining:<18} | {predicted_remaining:<20.2f} | {error:<10.2f}")

    # Show full generation
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n{'='*80}")
    print("Full Generated Output:")
    print(f"{'='*80}")
    print(full_output)
    print(f"\nTotal tokens generated: {total_generated_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Interactively test linear probe predictions")
    parser.add_argument("--probe", type=str, default=None, help="Path to trained probe model (.pt file). If not provided, uses best model based on R²")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt (if not provided, will ask interactively)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Get probe path
    if args.probe:
        probe_path = args.probe
    else:
        probe_path = find_best_probe()

    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter prompt: ")

    # Load models
    model, tokenizer, probe = load_models(probe_path)

    # Generate with predictions
    generate_with_predictions(model, tokenizer, probe, prompt, args.max_tokens)


if __name__ == "__main__":
    main()
