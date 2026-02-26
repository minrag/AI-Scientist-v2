from .base_tool import BaseTool
from .semantic_scholar import SemanticScholarSearchTool, search_for_papers as search_for_papers_s2
from .open_alex import OpenAlexSearchTool, search_for_papers as search_for_papers_oa
from ai_scientist.utils.config_loader import get_default_academic_search_tool

__all__ = [
    "BaseTool",
    "SemanticScholarSearchTool",
    "OpenAlexSearchTool",
    "get_default_search_tool",
    "search_for_papers",
]


def get_default_search_tool():
    """根据配置返回默认的学术搜索工具实例"""
    default_tool = get_default_academic_search_tool()
    if default_tool == "open_alex":
        return OpenAlexSearchTool()
    else:
        return SemanticScholarSearchTool()


def search_for_papers(query, result_limit=10):
    """根据配置使用默认的学术搜索工具搜索论文"""
    default_tool = get_default_academic_search_tool()
    if default_tool == "open_alex":
        return search_for_papers_oa(query, result_limit)
    else:
        return search_for_papers_s2(query, result_limit)