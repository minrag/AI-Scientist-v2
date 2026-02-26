from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from ai_scientist.llm import _get_model_config

def get_ai_client(model: str, **model_kwargs):
    """Return an AI client based on configuration key or raw name.

    If ``model`` corresponds to one of the configured keys (llm/vlm/code),
    the client_type from the configuration is used to pick the backend.  For
    backward compatibility, raw API model strings will continue to use substring
    heuristics.
    """
    cfg = _get_model_config(model)
    if not cfg:
        raise ValueError(
            f"模型 '{model}' 未在配置文件中定义。请使用 llm、vlm 或 code 等键。"
        )
    client_type = cfg.get("client_type", "").lower()
    if client_type == "anthropic":
        return backend_anthropic.get_ai_client(model=model, **model_kwargs)
    else:
        return backend_openai.get_ai_client(model=model, **model_kwargs)

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
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # pass along temperature; configuration may override.  ``max_tokens``
    # is intentionally omitted so that callers do not specify it.
    if temperature is not None:
        model_kwargs["temperature"] = temperature

    # determine which query function to use via config
    cfg = _get_model_config(model)
    if not cfg:
        raise ValueError(
            f"模型 '{model}' 未在配置文件中定义。请使用 llm、vlm 或 code 等键。"
        )
    query_func = backend_anthropic.query if cfg.get("client_type", "").lower() == "anthropic" else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
