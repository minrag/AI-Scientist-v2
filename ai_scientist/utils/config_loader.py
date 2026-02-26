"""统一的配置加载工具

提供从 llm_config.yaml 加载配置的公共方法，支持缓存和不同的配置获取需求。
"""

import os
from pathlib import Path
from typing import Any, Optional

# 配置缓存
_config_cache: Optional[dict] = None


def load_llm_config() -> Optional[dict]:
    """加载 LLM 配置文件

    从项目根目录的 llm_config.yaml 读取配置，支持 omegaconf（如果可用）或回退到 yaml。
    使用缓存避免重复读取文件。

    Returns:
        dict: 配置字典，如果文件不存在或加载失败则返回 None
    """
    global _config_cache

    # 如果已缓存，直接返回
    if _config_cache is not None:
        return _config_cache

    # 只从项目根目录的 llm_config.yaml 读取配置
    config_path = Path.cwd() / "llm_config.yaml"

    if not config_path.exists():
        print(f"警告: 配置文件 {config_path} 不存在")
        _config_cache = None
        return None

    try:
        # 尝试使用 omegaconf
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.load(str(config_path))
            config_dict = OmegaConf.to_container(config, resolve=True)
            _config_cache = config_dict
            return config_dict
        except ImportError:
            # 回退到 yaml
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            _config_cache = config_dict
            return config_dict
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        _config_cache = None
        return None


def get_model_config(model: str) -> Optional[dict]:
    """获取指定模型的配置

    Args:
        model: 模型键名，必须是 "llm"、"vlm" 或 "code" 之一

    Returns:
        dict: 模型配置字典，如果配置不存在则返回 None
    """
    config = load_llm_config()
    if config and "models" in config and model in config["models"]:
        return config["models"][model]
    return None


def get_semantic_scholar_api_key() -> Optional[str]:
    """获取 Semantic Scholar API 密钥

    首先尝试从配置文件的 semantic_scholar.api_key 读取，
    如果不存在或为空，则回退到环境变量 S2_API_KEY。

    Returns:
        str: API 密钥，如果不存在则返回 None
    """
    config = load_llm_config()

    # 首先从配置文件读取
    if config and "semantic_scholar" in config:
        api_key = config["semantic_scholar"].get("api_key", "").strip()
        if api_key:
            return api_key

    # 回退到环境变量
    return os.getenv("S2_API_KEY")


def clear_config_cache() -> None:
    """清除配置缓存，强制重新加载配置文件"""
    global _config_cache
    _config_cache = None