"""
Prompt Manager 模块
===================
用于从 prompt.yaml 文件中读取和管理所有提示词配置。

使用示例:
    from ai_scientist.utils.prompt_manager import get_prompt

    # 获取简单字符串提示词
    system_prompt = get_prompt('ideation.system_prompt')

    # 获取带格式化的提示词
    prompt = get_prompt('ideation.idea_generation',
                        workshop_description=desc,
                        prev_ideas_string=ideas)

    # 获取嵌套的字典配置
    spec = get_prompt('agent_manager.stage_config_spec')
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


def _get_prompt_file_path() -> str:
    """
    获取 prompt.yaml 文件的路径。

    按以下顺序查找：
    1. 当前工作目录
    2. 模块的父目录（项目根目录）

    Returns:
        prompt.yaml 文件的绝对路径

    Raises:
        FileNotFoundError: 当 prompt.yaml 文件不存在时
    """
    # 尝试当前工作目录
    cwd_config = Path.cwd() / "prompt.yaml"
    if cwd_config.exists():
        return str(cwd_config)

    # 尝试模块的父目录（项目根目录）
    module_dir = Path(__file__).parent.parent.parent
    module_config = module_dir / "prompt.yaml"
    if module_config.exists():
        return str(module_config)

    raise FileNotFoundError(
        f"prompt.yaml not found. Searched in:\n"
        f"  - {cwd_config}\n"
        f"  - {module_config}"
    )


_PROMPT_FILE_PATH = _get_prompt_file_path()


@lru_cache(maxsize=1)
def _load_prompt_file() -> Dict[str, Any]:
    """
    加载 prompt.yaml 文件并缓存结果。

    Returns:
        包含所有提示词配置的字典

    Raises:
        FileNotFoundError: 当 prompt.yaml 文件不存在时
        yaml.YAMLError: 当 YAML 文件解析失败时
    """
    with open(_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_nested_value(data: Dict[str, Any], keys: list) -> Any:
    """
    从嵌套字典中获取值。

    Args:
        data: 字典数据
        keys: 键名列表，用于逐层访问

    Returns:
        获取到的值

    Raises:
        KeyError: 当指定的键不存在时
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise KeyError(
                f"Key path '{'.'.join(keys)}' not found. "
                f"Failed at key: '{key}'"
            )
    return current


def get_prompt(prompt_path: str, **format_kwargs) -> Any:
    """
    从 prompt.yaml 中获取指定的提示词配置。

    这是获取提示词的主要方法，支持：
    - 点分隔的路径访问嵌套配置
    - 自动字符串格式化（如果提供了 format_kwargs）
    - 结果缓存以提高性能

    Args:
        prompt_path: 提示词在 YAML 中的路径，使用点分隔
                    例如：'ideation.system_prompt', 'review.neurips_form'
        **format_kwargs: 用于字符串格式化的可选参数

    Returns:
        提示词内容（字符串、字典、列表等）

    Raises:
        KeyError: 当指定的提示词路径不存在时
        FileNotFoundError: 当 prompt.yaml 文件不存在时
        yaml.YAMLError: 当 YAML 文件解析失败时

    Examples:
        # 获取简单的字符串提示词
        system_prompt = get_prompt('ideation.system_prompt')

        # 获取带格式化的提示词
        prompt = get_prompt(
            'ideation.idea_generation',
            workshop_description=desc,
            prev_ideas_string=ideas
        )

        # 获取嵌套的字典配置
        spec = get_prompt('agent_manager.stage_config_spec')
    """
    # 加载 YAML 文件
    prompt_data = _load_prompt_file()

    # 解析路径
    keys = prompt_path.split(".")

    # 获取值
    value = _get_nested_value(prompt_data, keys)

    # 如果是字符串且提供了格式化参数，则进行格式化
    if isinstance(value, str) and format_kwargs:
        try:
            return value.format(**format_kwargs)
        except KeyError as e:
            raise KeyError(
                f"Missing format parameter for key: {e}. "
                f"Available parameters: {list(format_kwargs.keys())}"
            )

    # 如果是字典且提供了格式化参数，递归格式化字典中的字符串
    elif isinstance(value, dict) and format_kwargs:
        return _format_dict(value, format_kwargs)

    return value


def _format_dict(data: Dict[str, Any], format_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归格式化字典中的所有字符串值。

    Args:
        data: 字典数据
        format_kwargs: 格式化参数

    Returns:
        格式化后的字典
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            # 只对包含占位符的字符串进行格式化，避免代码示例中的花括号导致错误
            placeholders = [k for k in format_kwargs.keys() if f'{{{k}}}' in value]
            if placeholders:
                try:
                    result[key] = value.format(**format_kwargs)
                except (KeyError, ValueError, IndexError):
                    # 如果格式化失败，保留原始值
                    # 代码中可能包含 { } 等特殊字符，导致 format() 失败
                    result[key] = value
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _format_dict(value, format_kwargs)
        elif isinstance(value, list):
            result[key] = [
                _format_dict(item, format_kwargs) if isinstance(item, dict)
                else _format_string_safe(item, format_kwargs) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _format_string_safe(value: str, format_kwargs: Dict[str, Any]) -> str:
    """
    安全地格式化字符串，只在包含占位符时才进行格式化。

    Args:
        value: 待格式化的字符串
        format_kwargs: 格式化参数

    Returns:
        格式化后的字符串或原始字符串
    """
    # 只对包含占位符的字符串进行格式化
    placeholders = [k for k in format_kwargs.keys() if f'{{{k}}}' in value]
    if placeholders:
        try:
            return value.format(**format_kwargs)
        except (KeyError, ValueError, IndexError):
            # 如果格式化失败，保留原始值
            return value
    return value


def get_all_prompts() -> Dict[str, Any]:
    """
    获取所有提示词配置。

    Returns:
        包含所有提示词的完整字典
    """
    return _load_prompt_file()


def list_prompt_paths(prefix: str = "") -> list:
    """
    列出所有可用的提示词路径。

    Args:
        prefix: 可选的前缀过滤，例如 'ideation.' 只列出 ideation 下的路径

    Returns:
        所有提示词路径的列表
    """
    prompt_data = _load_prompt_file()
    paths = []

    def _traverse(data, current_path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, dict):
                    _traverse(value, new_path)
                else:
                    paths.append(new_path)

    _traverse(prompt_data)

    if prefix:
        prefix = prefix.rstrip(".")
        paths = [p for p in paths if p.startswith(prefix)]

    return sorted(paths)


def reload_prompts():
    """
    重新加载提示词配置（清除缓存）。

    当 prompt.yaml 文件被修改后，调用此函数可以重新加载最新的配置。
    """
    _load_prompt_file.cache_clear()


# 为了方便使用，预定义一些常用的提示词路径常量
class PromptPaths:
    """
    提示词路径常量类

    提供了所有可用提示词路径的常量定义，便于在代码中引用。
    使用示例:
        from ai_scientist.utils.prompt_manager import PromptPaths
        prompt = get_prompt(PromptPaths.IDEATION_SYSTEM)
    """
    # Ideation 阶段
    IDEATION_SYSTEM = "ideation.system_prompt"
    IDEATION_IDEA_GENERATION = "ideation.idea_generation"
    IDEATION_IDEA_REFLECTION = "ideation.idea_reflection"

    # Writeup 阶段
    WRITEUP_CITATION_SYSTEM = "writeup.citation_system"
    WRITEUP_CITATION_FIRST = "writeup.citation_first"
    WRITEUP_CITATION_SECOND = "writeup.citation_second"
    WRITEUP_SYSTEM_TEMPLATE = "writeup.system_template"
    WRITEUP_PROMPT = "writeup.writeup_prompt"
    WRITEUP_REFLECTION = "writeup.reflection"

    # Review 阶段
    REVIEW_SYSTEM_BASE = "review.system_base"
    REVIEW_SYSTEM_NEG = "review.system_neg"
    REVIEW_SYSTEM_POS = "review.system_pos"
    REVIEW_NEURIPS_FORM = "review.neurips_form"
    REVIEW_REFLECTION = "review.reflection"
    REVIEW_META_REVIEWER_SYSTEM = "review.meta_reviewer_system"

    # VLM Review 阶段
    VLM_IMG_CAP_REF = "vlm_review.img_cap_ref"
    VLM_IMG_CAP_SELECTION = "vlm_review.img_cap_selection"
    VLM_IMG_REVIEW = "vlm_review.img_review"

    # Log Summarization 阶段
    LOG_REPORT_SUMMARIZER_SYSTEM = "log_summarization.report_summarizer_system"
    LOG_REPORT_SUMMARIZER = "log_summarization.report_summarizer"
    LOG_OUTPUT_FORMAT_CONTROL = "log_summarization.output_format_control"
    LOG_STAGE_AGGREGATE = "log_summarization.stage_aggregate"
    LOG_OVERALL_PLAN_SUMMARIZER = "log_summarization.overall_plan_summarizer"

    # Agent Manager 阶段
    AGENT_MANAGER_STAGE_CONFIG_SPEC = "agent_manager.stage_config_spec"
    AGENT_MANAGER_STAGE_PROGRESS_EVAL_SPEC = "agent_manager.stage_progress_eval_spec"
    AGENT_MANAGER_STAGE_COMPLETION_EVAL_SPEC = "agent_manager.stage_completion_eval_spec"
    AGENT_MANAGER_MAIN_STAGE_GOALS = "agent_manager.main_stage_goals"
    AGENT_MANAGER_TASK_DESC_TEMPLATE = "agent_manager.task_desc_template"

    # Metrics Parsing 阶段 (parallel_agent.py)
    METRICS_PARSING_PROMPT = "metrics_parsing_prompt"

    # Journal 阶段
    JOURNAL_NODE_SELECTION_SPEC = "journal.node_selection_spec"
    JOURNAL_NODE_SELECTION_PROMPT = "journal.node_selection_prompt"
    JOURNAL_SUMMARY_PROMPT = "journal.journal_summary_prompt"
    JOURNAL_SUMMARY_USER = "journal.journal_summary_user"
    JOURNAL_EXPERIMENT_NOTES_SUMMARY = "journal.experiment_notes_summary"

    # Journal2Report 阶段
    JOURNAL2REPORT_SYSTEM_PROMPT = "journal2report.system_prompt"
    JOURNAL2REPORT_CONTEXT_PROMPT = "journal2report.context_prompt"


# 模块级别的便捷函数
def get_ideation_prompt(name: str, **kwargs) -> Any:
    """获取 ideation 相关的提示词"""
    return get_prompt(f"ideation.{name}", **kwargs)


def get_writeup_prompt(name: str, **kwargs) -> Any:
    """获取 writeup 相关的提示词"""
    return get_prompt(f"writeup.{name}", **kwargs)


def get_review_prompt(name: str, **kwargs) -> Any:
    """获取 review 相关的提示词"""
    return get_prompt(f"review.{name}", **kwargs)


def get_vlm_review_prompt(name: str, **kwargs) -> Any:
    """获取 vlm_review 相关的提示词"""
    return get_prompt(f"vlm_review.{name}", **kwargs)


def get_log_summarization_prompt(name: str, **kwargs) -> Any:
    """获取 log_summarization 相关的提示词"""
    return get_prompt(f"log_summarization.{name}", **kwargs)


def get_agent_manager_prompt(name: str, **kwargs) -> Any:
    """获取 agent_manager 相关的提示词"""
    return get_prompt(f"agent_manager.{name}", **kwargs)


def get_journal_prompt(name: str, **kwargs) -> Any:
    """获取 journal 相关的提示词"""
    return get_prompt(f"journal.{name}", **kwargs)


def get_journal2report_prompt(name: str, **kwargs) -> Any:
    """获取 journal2report 相关的提示词"""
    return get_prompt(f"journal2report.{name}", **kwargs)
