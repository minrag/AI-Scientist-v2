"""
Unified model configuration loader.

This module provides functions to load model configurations from config.yaml
and create API clients based on model_type rather than hardcoded model names.
"""

import os
from pathlib import Path
from typing import Any, Dict, Tuple
import openai
import anthropic


# Cache for loaded config to avoid repeated file reads
_config_cache: Dict[str, Any] = {}


def get_config_path() -> Path:
    """Get the path to config.yaml file.

    Searches in the following order:
    1. Current working directory
    2. Parent directory of this module
    3. AI_SCIENTIST_ROOT environment variable
    """
    # Try current working directory first
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    # Try parent directory of this module
    module_dir = Path(__file__).parent.parent.parent
    module_config = module_dir / "config.yaml"
    if module_config.exists():
        return module_config

    # Try AI_SCIENTIST_ROOT environment variable
    ai_scientist_root = os.environ.get("AI_SCIENTIST_ROOT")
    if ai_scientist_root:
        root_config = Path(ai_scientist_root) / "config.yaml"
        if root_config.exists():
            return root_config

    # Default to module directory
    return module_config


def load_config() -> Dict[str, Any]:
    """Load the entire config.yaml file.

    Returns:
        Dictionary containing the full configuration.
    """
    global _config_cache

    if "full_config" not in _config_cache:
        config_path = get_config_path()

        # Use OmegaConf if available, otherwise use yaml
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_path)
            _config_cache["full_config"] = OmegaConf.to_container(cfg, resolve=True)
        except ImportError:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    _config_cache["full_config"] = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "Either omegaconf or pyyaml is required to load config.yaml. "
                    "Install with: pip install omegaconf pyyaml"
                )

        # Setup environment variables from config
        setup_environment_variables()

    return _config_cache["full_config"]


def setup_environment_variables():
    """Set environment variables from config.yaml.

    This allows configuring environment variables like HF_ENDPOINT for HuggingFace mirror.
    """
    # Only set once at module load time
    if "env_configured" in _config_cache:
        return

    _config_cache["env_configured"] = True

    # Access the already-loaded config from cache
    config = _config_cache.get("full_config", {})
    env_config = config.get("environment", {})

    if env_config:
        for key, value in env_config.items():
            if value:  # Only set if value is not empty
                os.environ[key] = value
                print(f"Set {key}={value}")


def load_model_config(model_type: str) -> Dict[str, Any]:
    """Load configuration for a specific model type from config.yaml.

    Args:
        model_type: The model type key (e.g., 'llm', 'vlm', 'code', 'plot_aggregation',
                   'writeup', 'citation', 'small_model', 'review')

    Returns:
        Dictionary with keys: client_type, base_url, api_key, model_name

    Raises:
        ValueError: If model_type is not found in config.yaml
    """
    config = load_config()
    models = config.get("models", {})

    if model_type not in models:
        available_models = list(models.keys())
        raise ValueError(
            f"Model type '{model_type}' not found in config.yaml. "
            f"Available model types: {available_models}"
        )

    model_config = models[model_type]

    # Ensure all required fields are present
    required_fields = ["client_type", "model_name"]
    for field in required_fields:
        if field not in model_config:
            raise ValueError(
                f"Model '{model_type}' is missing required field: {field}"
            )

    # Return a copy with defaults for optional fields
    return {
        "client_type": model_config["client_type"],
        "base_url": model_config.get("base_url", None),
        "api_key": model_config.get("api_key", ""),
        "model_name": model_config["model_name"],
        "timeout": model_config.get("timeout", 1800),
    }


def create_client(model_type: str) -> Tuple[Any, str]:
    """Create an API client based on model_type from config.yaml.

    Args:
        model_type: The model type key from config.yaml

    Returns:
        Tuple of (client, model_name) where:
        - client: OpenAI or Anthropic client instance
        - model_name: The actual model name string to use in API calls

    Raises:
        ValueError: If model_type not found or client_type not supported
    """
    config = load_model_config(model_type)
    client_type = config["client_type"]
    api_key = config["api_key"]
    base_url = config["base_url"]
    model_name = config["model_name"]

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    if client_type == "openai":
        print(f"Using OpenAI API with model {model_name}.")
        return openai.OpenAI(**client_kwargs), model_name

    elif client_type == "anthropic":
        print(f"Using Anthropic API with model {model_name}.")
        return anthropic.Anthropic(**client_kwargs), model_name

    else:
        raise ValueError(
            f"Unsupported client_type '{client_type}' for model '{model_type}'. "
            f"Supported client types: 'openai', 'anthropic'"
        )


def get_available_model_types() -> list:
    """Get list of available model types from config.yaml.

    Returns:
        List of model type strings available in config.yaml
    """
    config = load_config()
    return list(config.get("models", {}).keys())
