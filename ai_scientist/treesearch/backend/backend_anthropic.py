import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

from ai_scientist.utils.model_config import load_model_config


# Extended exception list to catch more API errors
ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
    Exception,  # Catch all exceptions for debugging
)



def get_ai_client(model_type: str, max_retries=2) -> anthropic.Anthropic:
    """Get Anthropic client configured from config.yaml.

    Args:
        model_type: Model type key from config.yaml (e.g., "code")
        max_retries: Maximum number of retries for API calls

    Returns:
        Configured Anthropic client instance
    """
    config = load_model_config(model_type)

    client_kwargs = {}
    if config["api_key"]:
        client_kwargs["api_key"] = config["api_key"]
    client_kwargs["max_retries"] = max_retries

    # Set timeout to support long-running requests (>10 minutes)
    # Default is 600 seconds (10 minutes), set to 1 hours for long experiments
    client_kwargs["timeout"] = config.get("timeout", 3600)

    return anthropic.Anthropic(**client_kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Query Anthropic API.

    Args:
        system_message: System message
        user_message: User message
        func_spec: Optional function specification (not supported yet)
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

    # Ensure max_tokens is set (required by Anthropic API)
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = config.get("max_tokens", 64000)

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
