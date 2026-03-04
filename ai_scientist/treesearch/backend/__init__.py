from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from ai_scientist.utils.model_config import load_model_config


def get_ai_client(model_type: str, **model_kwargs):
    """
    Get the appropriate AI client based on model_type from config.yaml.

    Args:
        model_type (str): model type key from config.yaml (e.g., "llm", "code", "vlm")
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        Tuple of (client, model_name)
    """
    # Load model config to determine client type and get model name
    config = load_model_config(model_type)
    client_type = config["client_type"]
    model_name = config["model_name"]

    if client_type == "anthropic":
        client = backend_anthropic.get_ai_client(model_type=model_type, **model_kwargs)
    else:
        client = backend_openai.get_ai_client(model_type=model_type, **model_kwargs)

    return client, model_name


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message
        user_message (PromptType | None): Uncompiled user message
        model (str): model type key from config.yaml (e.g., "llm", "code", "vlm")
        temperature (float | None, optional): Temperature to sample at.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Determine backend based on client_type from config
    config = load_model_config(model)
    if config["client_type"] == "anthropic":
        query_func = backend_anthropic.query
    else:
        query_func = backend_openai.query

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
