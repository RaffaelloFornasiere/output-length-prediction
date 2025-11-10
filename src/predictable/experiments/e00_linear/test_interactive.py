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

from predictable.experiments.test_utils import find_best_probe, generate_with_predictions
from predictable.experiments.e00_linear.train import LinearProbe

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROBE_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")


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
    # Infer hidden_dim from model config
    hidden_dim = model.config.hidden_size
    probe = LinearProbe(hidden_dim).to(DEVICE)
    probe.load_state_dict(torch.load(probe_path, map_location=DEVICE, weights_only=True))
    probe.eval()

    # Detect if probe uses log-space (check filename)
    use_log = "log" in str(probe_path).lower()

    return model, tokenizer, probe, use_log


def main():
    parser = argparse.ArgumentParser(description="Interactively test linear probe predictions")
    parser.add_argument("--probe", type=str, default=None, help="Path to trained probe model (.pt file). If not provided, uses best model based on R²")
    parser.add_argument("--prompt", type=str, default=None, help="User message content (will be formatted with chat template)")
    parser.add_argument("--raw_prompt", type=str, default=None, help="Raw pre-formatted prompt (bypasses chat template)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Get probe path
    if args.probe:
        probe_path = Path(args.probe)
    else:
        probe_path = find_best_probe(PROBE_DIR, pattern="linear_*.pt")

    # Load models
    model, tokenizer, probe, use_log = load_models(probe_path)

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
    generate_with_predictions(model, tokenizer, probe, prompt, args.max_tokens, use_log, DEVICE)


if __name__ == "__main__":
    main()
