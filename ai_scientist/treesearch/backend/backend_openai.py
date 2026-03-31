import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

from ai_scientist.utils.model_config import load_model_config

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def get_ai_client(model_type: str, max_retries=2) -> openai.OpenAI:
    """Get OpenAI client configured from config.yaml.

    Args:
        model_type: Model type key from config.yaml (e.g., "llm", "vlm")
        max_retries: Maximum number of retries for API calls

    Returns:
        Configured OpenAI client instance
    """
    config = load_model_config(model_type)

    client_kwargs = {}
    if config["api_key"]:
        client_kwargs["api_key"] = config["api_key"]
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    client_kwargs["max_retries"] = max_retries

    return openai.OpenAI(**client_kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Query OpenAI API.

    Args:
        system_message: System message
        user_message: User message
        func_spec: Optional function specification for tool calling
        **model_kwargs: Additional model parameters including 'model', 'temperature'

    Returns:
        Tuple of (output, request_time, input_tokens, output_tokens, info)
    """
    model_type = model_kwargs.get("model")
    client = get_ai_client(model_type, max_retries=0)

    # Get actual model name from config
    config = load_model_config(model_type)
    model_name = config["model_name"]

    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    filtered_kwargs["model"] = model_name

    messages = opt_messages_to_list(system_message, user_message)
    if system_message is not None and user_message is None:
        messages = [{"role": "user", "content": system_message}]

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        ## tinging 模式不支持 tool_choice
        filtered_kwargs["extra_body"] =  {"enable_thinking": False}

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
