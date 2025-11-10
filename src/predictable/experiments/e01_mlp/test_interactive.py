"""
Interactive testing script for MLP probe.
Loads trained probe and Llama 3.2 3B, generates tokens step-by-step,
and shows predicted remaining tokens at each generation step.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from predictable.experiments.test_utils import find_best_probe, generate_with_predictions
from predictable.experiments.e01_mlp.train import MLPProbe, parse_hidden_dims

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROBE_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


def parse_probe_config(probe_path: Path) -> tuple:
    """Parse hidden_dims and dropout from probe filename.

    Returns:
        (hidden_dims, dropout)

    Example filename: mlp_log_h512_256_d0.1_e100_bs64_lr0.001_spear0.988_r20.97.pt
    Hidden dims are between 'h' and 'd' parts: h512_256 -> [512, 256]
    """
    filename = probe_path.stem
    parts = filename.split('_')

    # Find indices of h and d markers
    h_idx = next(i for i, p in enumerate(parts) if p.startswith('h'))
    d_idx = next(i for i, p in enumerate(parts) if p.startswith('d'))

    # Extract hidden dimensions between h and d
    # First part: remove 'h' prefix from h512 -> 512
    # Remaining parts: all numeric parts between h and d
    hidden_parts = [parts[h_idx][1:]]  # Remove 'h' prefix
    hidden_parts.extend(parts[h_idx + 1:d_idx])  # Add all parts until 'd'

    # Join with commas and parse
    hidden_str = ','.join(hidden_parts)
    hidden_dims = parse_hidden_dims(hidden_str)

    # Parse dropout (e.g., d0.1)
    dropout_part = parts[d_idx]
    dropout = float(dropout_part[1:])

    return hidden_dims, dropout


def load_models(probe_path: Path):
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
    # Infer input_dim from model config
    input_dim = model.config.hidden_size

    # Parse probe architecture from filename
    hidden_dims, dropout = parse_probe_config(probe_path)
    print(f"Probe architecture: hidden_dims={hidden_dims}, dropout={dropout}")

    probe = MLPProbe(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(DEVICE)
    probe.load_state_dict(torch.load(probe_path, map_location=DEVICE, weights_only=True))
    probe.eval()

    return model, tokenizer, probe


def main():
    parser = argparse.ArgumentParser(description="Interactively test MLP probe predictions")
    parser.add_argument("--probe", type=str, default=None, help="Path to trained probe model (.pt file). If not provided, uses best model based on R²")
    parser.add_argument("--prompt", type=str, default=None, help="User message content (will be formatted with chat template)")
    parser.add_argument("--raw_prompt", type=str, default=None, help="Raw pre-formatted prompt (bypasses chat template)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Get probe path
    if args.probe:
        probe_path = Path(args.probe)
    else:
        probe_path = find_best_probe(PROBE_DIR, pattern="mlp_*.pt")

    # Load models
    model, tokenizer, probe = load_models(probe_path)

    # Get prompt
    if args.raw_prompt:
        # Use raw prompt directly (for backward compatibility)
        prompt = args.raw_prompt
    elif args.prompt:
        # Apply chat template
        messages = [{"role": "system", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Interactive input
        user_content = input("Enter instruction: ")
        messages = [{"role": "system", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Generate with predictions
    generate_with_predictions(model, tokenizer, probe, prompt, args.max_tokens, DEVICE)


if __name__ == "__main__":
    main()
