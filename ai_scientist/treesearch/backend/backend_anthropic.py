import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic


ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)

def get_ai_client(model: str, base_url: str | None = None, api_key: str | None = None, max_retries=2) -> anthropic.Anthropic:
    """Return an Anthropic client.

    The caller can supply an ``api_key`` via keyword arguments;
    if not provided, the client will fall back to the ANTHROPIC_API_KEY environment variable.
    The ``base_url`` parameter is accepted for API consistency but currently ignored for Anthropic.
    """
    kwargs = {"max_retries": max_retries}

    if api_key is not None:
        kwargs["api_key"] = api_key
    # Anthropic client doesn't support base_url parameter
    # It's accepted for API consistency but ignored
    return anthropic.Anthropic(**kwargs)

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
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
    # Note: we intentionally do **not** inject a default ``max_tokens`` value
    # here.  The top-level code should refrain from specifying this parameter
    # when making queries so that the underlying API can use its own defaults.

    if func_spec is not None:
        raise NotImplementedError(
            "Anthropic does not support function calling for now."
        )

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    print(filtered_kwargs)

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
