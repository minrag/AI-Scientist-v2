"""
LLM utilities for model inference.

This module provides unified interfaces for interacting with LLMs.
All model configurations are loaded from config.yaml via model_type.
"""

import json
import re
from typing import Any

import anthropic
import backoff
import openai

from ai_scientist.utils.model_config import create_client, load_model_config
from ai_scientist.utils.token_tracker import track_token_usage


RETRYABLE_LLM_ERRORS = (Exception,)


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    RETRYABLE_LLM_ERRORS,
    max_tries=5,
)
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    return _get_batch_responses_from_llm_once(
        prompt,
        client,
        model,
        system_message,
        print_debug=print_debug,
        msg_history=msg_history,
        temperature=temperature,
        n_responses=n_responses,
    )


def _get_batch_responses_from_llm_once(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from LLM for the same input.

    Args:
        prompt: The user prompt/message
        client: OpenAI or Anthropic client instance
        model: The model name (from config)
        system_message: System message/instructions
        print_debug: Whether to print debug information
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate

    Returns:
        Tuple of (list of response contents, list of message histories)
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    # Check client type by inspecting the client
    is_anthropic = isinstance(client, anthropic.Anthropic)

    if is_anthropic:
        # Anthropic API - generate responses one at a time for multiple responses
        content = []
        new_msg_histories = []
        for _ in range(n_responses):
            c, hist = _get_response_from_llm_once(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=msg_history,
                temperature=temperature,
            )
            content.append(c)
            new_msg_histories.append(hist)
        new_msg_history = new_msg_histories
    else:
        # OpenAI API - can generate multiple responses in one call
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0] if isinstance(new_msg_history[0], list) else new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a single LLM call.

    Args:
        client: OpenAI or Anthropic client instance
        model: The model name (from config)
        temperature: Sampling temperature
        system_message: System message
        prompt: List of message dictionaries

    Returns:
        API response object
    """
    is_anthropic = isinstance(client, anthropic.Anthropic)

    if is_anthropic:
        return client.messages.create(
            model=model,
            temperature=temperature,
            system=system_message,
            messages=prompt,
        )

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *prompt,
        ],
        temperature=temperature,
        n=1,
    )


@backoff.on_exception(
    backoff.expo,
    RETRYABLE_LLM_ERRORS,
    max_tries=5,
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    return _get_response_from_llm_once(
        prompt,
        client,
        model,
        system_message,
        print_debug=print_debug,
        msg_history=msg_history,
        temperature=temperature,
    )


def _get_response_from_llm_once(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Get a single response from LLM.

    Args:
        prompt: The user prompt/message
        client: OpenAI or Anthropic client instance
        model: The model name (from config)
        system_message: System message/instructions
        print_debug: Whether to print debug information
        msg_history: Previous message history
        temperature: Sampling temperature

    Returns:
        Tuple of (response content, new message history)
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    is_anthropic = isinstance(client, anthropic.Anthropic)

    if is_anthropic:
        # Anthropic API
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    else:
        # OpenAI API
        new_msg_history = msg_history + [{"role": "user", "content": msg}]

        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )

        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON content from LLM output.

    Looks for JSON between ```json and ``` markers,
    or falls back to finding any JSON-like content.

    Args:
        llm_output: Raw LLM output string

    Returns:
        Parsed JSON as dict, or None if no valid JSON found
    """
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found
