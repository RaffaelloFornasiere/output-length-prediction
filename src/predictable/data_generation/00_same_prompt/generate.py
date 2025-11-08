"""
Generate dataset 00: Same prompt for all samples.
Uses a single prompt template but varies the count and word parameters.
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

# Single prompt template used for all samples
PROMPT_TEMPLATE = "Print exactly {count} repetitions of the token \"{word}\". Do not include anything else."

# Parameters to vary
COUNTS = list(range(5, 50))  # X: how many times to print
WORDS = ["hello", "world", "cat", "dog", "python", "test", "apple", "blue", "sun", "code"]


def main():
    """Generate dataset with same prompt template."""
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
                # Extract hidden states
                result = extract_hidden_states(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=200,
                    temperature=0.0,
                    layers_to_extract=[-1]  # Last layer only
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
    metadata = {
        "dataset_id": "00_same_prompt",
        "description": "Dataset generated using a single prompt template",
        "model_name": MODEL_NAME,
        "prompt_template": PROMPT_TEMPLATE,
        "counts_range": [min(COUNTS), max(COUNTS)],
        "words": WORDS,
        "total_samples": total_samples,
        "total_tokens": len(all_hidden_states),
        "layers_extracted": "last_layer_only"
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