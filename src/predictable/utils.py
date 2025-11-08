"""
Shared utilities for the predictable package.
"""
from typing import List, Dict, Any


def apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    remove_date_lines: bool = True,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
) -> str:
    """
    Apply chat template to messages with optional date line removal.

    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content' keys
        remove_date_lines: If True, removes the date-related lines from chat template
        add_generation_prompt: Whether to add generation prompt
        tokenize: Whether to tokenize the output

    Returns:
        Formatted prompt string (or tokens if tokenize=True)

    Example:
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> prompt = apply_chat_template(tokenizer, messages)
    """
    if remove_date_lines:
        # Modify the chat template to remove date lines
        tmpl = tokenizer.chat_template
        tmpl = tmpl.replace('Cutting Knowledge Date: December 2023\\n', '')
        tmpl = tmpl.replace('{{- "Today Date: " + date_string + "\\n\\n" }}', '')
        tokenizer.chat_template = tmpl

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )

    return prompt
