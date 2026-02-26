import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.config_loader import load_llm_config, get_model_config

import anthropic
import backoff
import openai
from pathlib import Path


# Only three model keys are supported; defined in llm_config.yaml under 'models'.
# llm: large language model
# vlm: vision-language / multimodal model
# code: code generation model
AVAILABLE_LLMS = ["llm", "vlm", "code"]


# Get N responses from a single message, used for ensembling.
@track_token_usage
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
    """Return multiple independent responses from the LLM.

    The provided ``model`` should already be the API-level model name from
    configuration.  No string sniffing or special cases are performed here;
    all client-specific dispatch is handled by :func:`get_response_from_llm`.
    This implementation simply calls that helper repeatedly and aggregates
    the results.
    """
    if msg_history is None:
        msg_history = []

    responses: list[str] = []
    histories: list[list[dict[str, Any]]] = []
    for _ in range(n_responses):
        resp, hist = get_response_from_llm(
            prompt,
            client,
            model,
            system_message,
            print_debug=print_debug,
            msg_history=msg_history,
            temperature=temperature,
        )
        responses.append(resp)
        histories.append(hist)

    if print_debug and histories:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(responses)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return responses, histories


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a single LLM request using client/model from configuration.

    The supplied ``model`` should already be the API model name derived from
    the configuration. No string-based heuristics are applied.

    This helper wraps the underlying client call in a tight try/except block
    so that if the underlying HTTP response is empty or otherwise unparsable
    (which the OpenAI/Anthropic SDK surfaces as ``JSONDecodeError`` during
    response.json()), we convert it into a more descriptive error with the
    request details.  Downstream code can catch ``RuntimeError`` if it needs
    to handle such failures specially.
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
                "LLM client returned a falsy response (None or empty)."
            )
        return resp
    except json.decoder.JSONDecodeError as e:
        # This indicates the HTTP response body could not be decoded as JSON,
        # typically because it was empty.  Include context for easier debugging.
        raise RuntimeError(
            "LLM API returned an invalid or empty response body when calling"
            f" model={model}. Please check your network/base_url and API key."
        ) from e


def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Get a single response from the configured LLM client.

    This helper is intentionally simple: it does **not** inspect the model
    string.  Instead it decides how to make the API call solely on whether the
    client object has an ``messages`` attribute (Anthropic-style) or not
    (OpenAI-style).  The ``model`` parameter is expected to be the API-level
    name already loaded from configuration.
    """
    msg = prompt
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # perform the actual API call and capture any failure modes.  Instead of
    # letting exceptions bubble out and halt the caller, we catch known
    # failure classes and convert them into an empty-response result.  This
    # keeps higher‑level loops (e.g. idea generation) running rather than
    # aborting the entire process.
    try:
        if hasattr(client, "messages"):
            # Anthropic client
            @backoff.on_exception(
                backoff.expo,
                anthropic.RateLimitError,
            )
            def call_anthropic():
                return client.messages.create(
                    model=model,
                    temperature=temperature,
                    system=system_message,
                    messages=new_msg_history,
                )
            response = call_anthropic()
            content = response.content[0].text
        else:
            # OpenAI-style client
            response = make_llm_call(
                client,
                model,
                temperature,
                system_message=system_message,
                prompt=new_msg_history,
            )
            # response should be a proper object with choices; if the call
            # succeeded but returned nothing, our earlier wrapper would have
            # already raised.
            try:
                content = response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(
                    "LLM response did not contain expected 'choices' field"
                ) from e
    except RuntimeError as rte:
        # this includes our own invalid/empty response errors from make_llm_call
        # Log the problem then re-raise so that the caller can decide how to
        # handle it (e.g. abort generation).  This prevents the outer loop from
        # repeatedly attempting to parse an empty string.
        print(f"LLM call error: {rte}")
        raise
    except Exception as e:
        # any other unexpected issue - log and propagate
        import traceback
        traceback.print_exc()
        print(f"Unexpected exception in get_response_from_llm: {e} (type {type(e)})")
        raise

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


# _load_llm_config 函数已被移动到 ai_scientist.utils.config_loader 模块
# 请使用 from ai_scientist.utils.config_loader import load_llm_config


def _get_model_config(model: str):
    """获取指定模型的配置（向后兼容性包装器）"""
    from ai_scientist.utils.config_loader import get_model_config
    return get_model_config(model)


def create_client(model) -> tuple[Any, str]:
    """Create client based strictly on llm_config.yaml entries.

    Only three top-level model keys are recognized (llm, vlm, code).
    Client types must be either "openai" or "anthropic" and an api_key
    must be provided directly in the configuration. Environment variables
    are no longer consulted and there is no built-in fallback logic for
    arbitrary model name patterns.
    """
    config = load_llm_config()
    if not config or "models" not in config or model not in config["models"]:
        raise ValueError(
            f"模型 '{model}' 未在配置文件中定义。请在 llm_config.yaml 中使用 llm、vlm 或 code 之一。"
        )
    model_config = config["models"][model]

    client_type = model_config.get("client_type", "").lower()
    if client_type not in ("openai", "anthropic"):
        raise ValueError(
            f"不支持的 client_type: {client_type} (模型: {model})，仅允许 openai 或 anthropic"
        )

    api_key = model_config.get("api_key")
    if not api_key:
        raise ValueError(f"模型 {model} 的配置必须包含 api_key")

    base_url = model_config.get("base_url")
    config_model_name = model_config.get("model_name", model)

    print(f"使用配置文件中的配置调用模型 {model}")

    if client_type == "openai":
        if base_url:
            return openai.OpenAI(api_key=api_key, base_url=base_url), config_model_name
        return openai.OpenAI(api_key=api_key), config_model_name
    else:  # anthropic
        return anthropic.Anthropic(api_key=api_key), config_model_name
