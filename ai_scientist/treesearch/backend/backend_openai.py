import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, base_url: str | None = None, api_key: str | None = None, max_retries=2) -> openai.OpenAI:
    """Return a generic OpenAI-compatible client.

    The caller can supply a ``base_url`` (for non-OpenAI endpoints) via
    keyword arguments; the model string itself is no longer examined.
    If ``api_key`` is provided, it will be used; otherwise the client will
    fall back to the OPENAI_API_KEY environment variable.
    """
    kwargs = {"max_retries": max_retries}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    return openai.OpenAI(**kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    # model_kwargs may include 'base_url', 'api_key' or other configuration fields
    client = get_ai_client(
        model_kwargs.get("model"),
        base_url=model_kwargs.get("base_url"),
        api_key=model_kwargs.get("api_key"),
        max_retries=0,
    )
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    # Remove client configuration parameters that should not be passed to the API call
    filtered_kwargs.pop("api_key", None)
    filtered_kwargs.pop("base_url", None)
    filtered_kwargs.pop("max_retries", None)

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

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
