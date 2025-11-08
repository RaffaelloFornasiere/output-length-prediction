"""
Generate dataset 02: Same prompt with all hidden layers.
Uses a single prompt template (same as dataset 00) but extracts hidden states from ALL layers.
"""
from pathlib import Path
from tqdm import tqdm

from predictable.data_generation.utils import (
    generate_prompt,
    load_model,
    extract_hidden_states,
    save_dataset
)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Can be changed per dataset
OUTPUT_DIR = Path(__file__).parent / "data"

# Single prompt template used for all samples (same as dataset 00)
PROMPT_TEMPLATE = "Print exactly {count} repetitions of the token \"{word}\". Do not include anything else."

# Parameters to vary
COUNTS = list(range(5, 50))  # X: how many times to print
WORDS = ["hello", "world", "cat", "dog", "python", "test", "apple", "blue", "sun", "code"]


def main():
    """Generate dataset with same prompt template, extracting ALL hidden layers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_name=MODEL_NAME)

    # Storage for all data
    all_hidden_states = []
    all_remaining_tokens = []
    all_token_metadata = []

    # Generate all combinations of counts and words
    total_samples = len(COUNTS) * len(WORDS)
    print(f"Generating {total_samples} samples using prompt template:")
    print(f"  '{PROMPT_TEMPLATE}'")
    print(f"  Counts: {min(COUNTS)} to {max(COUNTS)}")
    print(f"  Words: {WORDS}")
    print(f"  Extracting ALL hidden layers")
    print()

    # Progress bar
    pbar = tqdm(total=total_samples, desc="Generating samples")

    for count in COUNTS:
        for word in WORDS:
            # Generate prompt using the single template
            prompt = generate_prompt(
                tokenizer,
                prompt_template=PROMPT_TEMPLATE,
                use_chat_template=True,
                count=count,
                word=word
            )

            try:
                # Extract hidden states from ALL layers
                result = extract_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=200,
                    temperature=0.0,
                    extract_all_layers=True  # Extract all layers instead of just last
                )

                # Collect data
                all_hidden_states.extend(result['hidden_states'])
                all_remaining_tokens.extend(result['remaining_tokens'])
                all_token_metadata.extend(result['token_metadata'])

            except Exception as e:
                print(f"\nError on count={count}, word='{word}': {e}")
                continue

            pbar.update(1)

    pbar.close()

    # Save dataset with metadata
    print(f"\nSaving dataset with {len(all_hidden_states)} tokens...")

    # Get layer information from the first result
    # We need to run one more extraction to get the layer info
    sample_prompt = generate_prompt(
        tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        use_chat_template=True,
        count=5,
        word="test"
    )
    sample_result = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompt=sample_prompt,
        max_new_tokens=200,
        temperature=0.0,
        extract_all_layers=True
    )

    layers_info = sample_result['layers_info']

    metadata = {
        "dataset_id": "02_same_prompt_multi_layers",
        "description": "Dataset generated using a single prompt template, extracting ALL hidden layers",
        "model_name": MODEL_NAME,
        "prompt_template": PROMPT_TEMPLATE,
        "counts_range": [min(COUNTS), max(COUNTS)],
        "words": WORDS,
        "total_samples": total_samples,
        "total_tokens": len(all_hidden_states),
        "layers_extracted": "all_layers",
        "num_layers": layers_info['num_layers'],
        "extracted_layer_indices": layers_info['extracted_layers'],
        "hidden_dim_per_layer": layers_info['hidden_dim_per_layer'],
        "total_hidden_dim": all_hidden_states[0].shape[0] if all_hidden_states else 0
    }

    save_dataset(
        output_dir=OUTPUT_DIR,
        hidden_states_list=all_hidden_states,
        remaining_tokens_list=all_remaining_tokens,
        token_metadata_list=all_token_metadata,
        train_val_split=0.9,
        additional_metadata=metadata
    )


if __name__ == "__main__":
    main()
