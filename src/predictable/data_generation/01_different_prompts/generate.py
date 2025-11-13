"""
Generate dataset 01: Different prompts for samples.
Uses multiple prompt templates randomly selected for each sample.
"""
import random
from pathlib import Path
from tqdm import tqdm

from predictable.data_generation.utils import (
    generate_prompt,
    load_model,
    extract_hidden_states,
    save_dataset
)
from predictable.data_generation.prompts import prompts

def check_correctness(generated_tokens: list, expected_word: str, expected_count: int, tokenizer) -> tuple[bool, dict]:
    """
    Verify that the generated output contains only consecutive repetitions of the expected word.
    Does not check if the count matches - only that the output is purely the target word repeated.

    Args:
        generated_tokens: List of token metadata dicts from extract_hidden_states result
        expected_word: The word that should be repeated
        expected_count: The originally requested count (for logging purposes)
        tokenizer: Tokenizer to decode tokens

    Returns:
        Tuple of (is_correct, info_dict) where info_dict contains diagnostic information
    """
    # Reconstruct the generated text from token metadata
    generated_text = ''.join([token['token_text'] for token in generated_tokens])

    # Strip whitespace and split by whitespace to count words
    words = generated_text.strip().split()

    # Check if ALL words are the expected word (allowing for case variations)
    # If there are any other words mixed in, it's incorrect
    all_words_match = all(expected_word.lower() in w.strip().lower() for w in words) if words else False

    # Count actual occurrences
    actual_count = len(words)

    # Only accept if all words match and there's at least one word
    is_correct = all_words_match and actual_count > 0

    info = {
        'expected_count': expected_count,
        'actual_count': actual_count,
        'expected_word': expected_word,
        'generated_text': generated_text,
        'total_words': len(words),
        'all_words_match': all_words_match
    }

    return is_correct, info

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Can be changed per dataset
OUTPUT_DIR = Path(__file__).parent / "data"

# Parameters to vary
COUNTS = list(range(10, 50))  # X: how many times to print
WORDS = ["hello", "world", "cat", "dog", "python", "test", "apple", "blue", "sun", "code"]


def main():
    """Generate dataset with different prompt templates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_name=MODEL_NAME)

    # Storage for all data
    all_hidden_states = []
    all_remaining_tokens = []
    all_token_metadata = []

    # Generate all combinations of counts and words
    total_samples = len(COUNTS) * len(WORDS)
    print(f"Generating {total_samples} samples using {len(prompts)} different prompt templates")
    print(f"  Counts: {min(COUNTS)} to {max(COUNTS)}")
    print(f"  Words: {WORDS}")
    print(f"  Number of unique prompt templates: {len(prompts)}")
    print()

    # Progress bar
    pbar = tqdm(total=total_samples, desc="Generating samples")

    # Track prompt usage for metadata
    prompt_usage = {i: 0 for i in range(len(prompts))}

    # Track correctness statistics
    correctness_stats = {
        'total_samples': 0,
        'correct_samples': 0,
        'incorrect_samples': 0,
        'mismatches': [],
        'count_variations': []  # Store (expected, actual) for statistics
    }

    for count in COUNTS:
        for word in WORDS:
            # Randomly select a prompt template
            prompt_idx = random.randrange(len(prompts))
            prompt_template = prompts[prompt_idx]
            prompt_usage[prompt_idx] += 1

            # Generate prompt using the selected template
            prompt = generate_prompt(
                tokenizer,
                prompt_template=prompt_template,
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

                # Verify correctness
                is_correct, info = check_correctness(
                    generated_tokens=result['token_metadata'],
                    expected_word=word,
                    expected_count=count,
                    tokenizer=tokenizer
                )

                correctness_stats['total_samples'] += 1

                # Track count variation (expected vs actual)
                correctness_stats['count_variations'].append({
                    'expected': info['expected_count'],
                    'actual': info['actual_count'],
                    'is_correct': is_correct
                })

                # Update progress bar regardless of correctness
                pbar.update(1)

                if is_correct:
                    correctness_stats['correct_samples'] += 1
                else:
                    correctness_stats['incorrect_samples'] += 1
                    # Log mismatch with full prompt and generated text
                    mismatch_info = {
                        'count': count,
                        'word': word,
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'expected_count': info['expected_count'],
                        'actual_count': info['actual_count'],
                        'generated_text': info['generated_text'],
                        'generated_text_preview': info['generated_text'][:100]  # First 100 chars for quick view
                    }
                    correctness_stats['mismatches'].append(mismatch_info)
                    continue

                # Collect data
                all_hidden_states.extend(result['hidden_states'])
                all_remaining_tokens.extend(result['remaining_tokens'])
                all_token_metadata.extend(result['token_metadata'])

            except Exception as e:
                print(f"\nError on count={count}, word='{word}': {e}")
                continue

    pbar.close()

    # Compute statistics on count variations
    variations = correctness_stats['count_variations']
    deviations = [v['actual'] - v['expected'] for v in variations]
    abs_deviations = [abs(d) for d in deviations]

    # Compute bins for deviations
    import numpy as np
    bins = [-float('inf'), -10, -5, -1, 0, 1, 5, 10, float('inf')]
    bin_labels = ['<-10', '-10 to -5', '-5 to -1', '-1', 'exact', '+1', '+1 to +5', '+5 to +10', '>+10']
    bin_counts = {label: 0 for label in bin_labels}

    for dev in deviations:
        for i in range(len(bins) - 1):
            if bins[i] < dev <= bins[i+1]:
                bin_counts[bin_labels[i]] += 1
                break

    # Print correctness summary
    print(f"\n{'='*70}")
    print("CORRECTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {correctness_stats['total_samples']}")
    print(f"Correct samples (pure repetitions): {correctness_stats['correct_samples']}")
    print(f"Incorrect samples (with extra text): {correctness_stats['incorrect_samples']}")
    if correctness_stats['total_samples'] > 0:
        accuracy = correctness_stats['correct_samples'] / correctness_stats['total_samples'] * 100
        print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")

    # Save dataset with metadata
    print(f"Saving dataset with {len(all_hidden_states)} tokens from {correctness_stats['correct_samples']} accepted samples...")
    metadata = {
        "dataset_id": "01_different_prompts",
        "description": "Dataset generated using multiple different prompt templates",
        "model_name": MODEL_NAME,
        "num_prompt_templates": len(prompts),
        "counts_range": [min(COUNTS), max(COUNTS)],
        "words": WORDS,
        "planned_samples": total_samples,
        "attempted_samples": correctness_stats['total_samples'],
        "accepted_samples": correctness_stats['correct_samples'],
        "rejected_samples": correctness_stats['incorrect_samples'],
        "total_tokens_in_dataset": len(all_hidden_states),
        "layers_extracted": "last_layer_only",
        "prompt_usage_distribution": prompt_usage,
        "correctness_stats": {
            "note": "Statistics below include ALL attempted samples (accepted + rejected)",
            "accuracy": correctness_stats['correct_samples'] / correctness_stats['total_samples'] if correctness_stats['total_samples'] > 0 else 0,
            "rejected_samples_with_prompts": correctness_stats['mismatches'],
            "count_variation_stats": {
                "note": "Computed over all attempts, not just accepted samples",
                "mean_deviation": float(np.mean(deviations)),
                "mean_absolute_deviation": float(np.mean(abs_deviations)),
                "std_deviation": float(np.std(deviations)),
                "min_deviation": int(min(deviations)),
                "max_deviation": int(max(deviations)),
                "deviation_bins": bin_counts
            },
            "all_variations": correctness_stats['count_variations']
        }
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