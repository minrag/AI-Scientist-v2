import base64
from typing import Any
import re
import json
import backoff
import openai
import os
from PIL import Image
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.config_loader import load_llm_config, get_model_config
from pathlib import Path


# Only the configured vision-language model key is recognised.
AVAILABLE_VLMS = ["vlm"]


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a single LLM API call using the provided client and model name.

    The ``model`` argument is expected to already be the API-level model name
    (as returned by :func:`create_client`). There is no special casing based on
    string content; configuration controls which model name is supplied.

    We trap ``JSONDecodeError`` from the underlying SDK and raise a concise
    runtime error so callers dealing with VLM responses can surface a clear
    message.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            n=1,
            stop=None,
        )
        if not resp:
            raise RuntimeError(
                "VLM API returned a falsy response (None or empty) for model %s" % model
            )
        return resp
    except json.decoder.JSONDecodeError as e:
        raise RuntimeError(
            "VLM client returned invalid/empty JSON response (model=%s)" % model
        ) from e


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    """Perform a VLM API call with no heuristic branching.

    The ``model`` parameter should already come from configuration and may be
    any valid API model string. This helper simply forwards the arguments.
    """
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *prompt,
        ],
        temperature=temperature,
    )


def prepare_vlm_prompt(msg, image_paths, max_images):
    pass


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
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
    """Get response from vision-language model."""
    if msg_history is None:
        msg_history = []

    # construct content list regardless of model name
    if isinstance(image_paths, str):
        image_paths = [image_paths]

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


# _load_llm_config 函数已被移动到 ai_scientist.utils.config_loader 模块
# 请使用 from ai_scientist.utils.config_loader import load_llm_config


def _get_model_config(model: str):
    """获取指定模型的配置（向后兼容性包装器）"""
    return get_model_config(model)


def create_client(model: str) -> tuple[Any, str]:
    """Create a client for a configured VLM model.

    Only the keys defined in llm_config.yaml under `models` are accepted
    (typically ``llm``, ``vlm`` and ``code``). The configuration must specify
    a ``client_type`` of either ``openai`` or ``anthropic`` and include a
    direct ``api_key`` entry. No environment variables are consulted and no
    built-in fallbacks are performed.
    """
    config = load_llm_config()
    if not config or "models" not in config or model not in config["models"]:
        raise ValueError(
            f"VLM 模型 '{model}' 未在配置文件中定义。请在 llm_config.yaml 中正确设置。"
        )
    model_config = config["models"][model]

    client_type = model_config.get("client_type", "").lower()
    if client_type not in ("openai", "anthropic"):
        raise ValueError(
            f"VLM 不支持的 client_type: {client_type} (模型: {model})，仅允许 openai 或 anthropic"
        )

    api_key = model_config.get("api_key")
    if not api_key:
        raise ValueError(f"VLM 模型 {model} 的配置必须包含 api_key")

    base_url = model_config.get("base_url")
    config_model_name = model_config.get("model_name", model)

    print(f"使用配置文件中的配置调用 VLM 模型 {model}")
    if client_type == "openai":
        if base_url:
            return openai.OpenAI(api_key=api_key, base_url=base_url), config_model_name
        return openai.OpenAI(api_key=api_key), config_model_name
    else:
        return anthropic.Anthropic(api_key=api_key), config_model_name


def extract_json_between_markers(llm_output: str) -> dict | None:
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
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
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
        model: Name of model to use
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate

    Returns:
        Tuple of (list of response strings, list of message histories)
    """
    if msg_history is None:
        msg_history = []

    # convert single image path to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # build a mixed text/image message payload
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

    new_msg_history = msg_history + [{"role": "user", "content": content}]

    # always forward, no special-case logic needed
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        n=n_responses,
        seed=0,
    )

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
