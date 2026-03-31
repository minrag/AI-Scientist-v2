"""
VLM (Vision-Language Model) utilities for image understanding.

This module provides unified interfaces for interacting with VLMs.
All model configurations are loaded from config.yaml via model_type.
"""

import base64
import io
import json
import re
from typing import Any

import backoff
import openai
from PIL import Image

from ai_scientist.utils.model_config import create_client
from ai_scientist.utils.token_tracker import track_token_usage


RETRYABLE_VLM_ERRORS = (Exception,)


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    """Make a single VLM call.

    Args:
        client: OpenAI client instance
        model: The model name (from config)
        temperature: Sampling temperature
        system_message: System message
        prompt: List of message dictionaries (may include images)

    Returns:
        API response object
    """
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
    RETRYABLE_VLM_ERRORS,
    max_tries=5,
)
def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from vision-language model.

    Args:
        msg: Text message to send
        image_paths: Path(s) to image file(s)
        client: OpenAI client instance
        model: Name of model to use (from config)
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        max_images: Maximum number of images to include

    Returns:
        Tuple of (response content, new message history)
    """
    if msg_history is None:
        msg_history = []

    # Convert single image path to list for consistent handling
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Create content list starting with the text message
    content = [{"type": "text", "text": msg}]

    # Add each image to the content list
    for image_path in image_paths[:max_images]:
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )
    # Construct message with all images
    new_msg_history = msg_history + [{"role": "user", "content": content}]

    response = make_vlm_call(
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
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
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


@backoff.on_exception(
    backoff.expo,
    RETRYABLE_VLM_ERRORS,
    max_tries=5,
)
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from vision-language model for the same input.

    Args:
        msg: Text message to send
        image_paths: Path(s) to image file(s)
        client: OpenAI client instance
        model: Name of model to use (from config)
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate
        max_images: Maximum number of images to include

    Returns:
        Tuple of (list of response strings, list of message histories)
    """
    if msg_history is None:
        msg_history = []

    # Convert single image path to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Create content list with text and images
    content = [{"type": "text", "text": msg}]
    for image_path in image_paths[:max_images]:
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )

    # Construct message with all images
    new_msg_history = msg_history + [{"role": "user", "content": content}]

    # Get multiple responses
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        n=n_responses,
    )

    # Extract content from all responses
    contents = [r.message.content for r in response.choices]
    new_msg_histories = [
        new_msg_history + [{"role": "assistant", "content": c}] for c in contents
    ]

    if print_debug:
        # Just print the first response
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, new_msg_histories
