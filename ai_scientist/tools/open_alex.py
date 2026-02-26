import warnings
import time
from typing import Dict, List, Optional, Union

import backoff
import requests
import pyalex
from pyalex import Works

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.config_loader import get_open_alex_config


def inverted_index_to_text(inverted_index: Dict) -> str:
    """将OpenAlex的abstract_inverted_index转换为纯文本

    Args:
        inverted_index: 倒排索引字典，格式为{"word": [position1, position2, ...]}

    Returns:
        重建后的摘要文本，如果转换失败则返回空字符串
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    try:
        # 创建一个位置到单词的映射
        position_to_word = {}
        for word, positions in inverted_index.items():
            if isinstance(positions, list):
                for pos in positions:
                    if isinstance(pos, int):
                        position_to_word[pos] = word

        if not position_to_word:
            return ""

        # 找到最大位置
        max_position = max(position_to_word.keys())

        # 重建文本
        text_parts = []
        for i in range(max_position + 1):
            if i in position_to_word:
                text_parts.append(position_to_word[i])
            elif text_parts:  # 在单词之间添加空格
                # 这里我们简单处理，实际可能需要更复杂的逻辑
                pass

        return " ".join(text_parts)
    except Exception:
        # 如果转换失败，返回空字符串
        return ""


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class OpenAlexSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.config = get_open_alex_config()

        # 配置PyAlex
        self._configure_pyalex()

        if not self.config.get("email"):
            warnings.warn(
                "No OpenAlex email found. Requests may be rate-limited. "
                "Set the email in llm_config.yaml under 'open_alex' section."
            )

    def _configure_pyalex(self):
        """配置PyAlex库"""
        api_key = self.config.get("api_key")
        email = self.config.get("email")

        if api_key:
            pyalex.config.api_key = api_key

        if email:
            pyalex.config.email = email

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None

        try:
            # 使用PyAlex进行搜索
            # 使用字典解包语法传递带点号的字段名
            results = Works().filter(**{"default.search": query}).get(per_page=self.max_results)

            papers = []
            for work in results:
                # 将PyAlex工作对象转换为字典
                paper_dict = dict(work)

                # 尝试获取摘要
                abstract = self._get_abstract_from_work(work)
                if abstract:
                    paper_dict["abstract"] = abstract

                papers.append(paper_dict)

            if not papers:
                return None

            # 按引用数降序排序
            papers.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
            return papers

        except Exception as e:
            print(f"Error searching OpenAlex with PyAlex: {e}")
            return None

    def _get_abstract_from_work(self, work) -> Optional[str]:
        """从PyAlex工作对象中获取摘要"""
        try:
            # 首先检查是否有直接的abstract字段
            if "abstract" in work and work["abstract"]:
                return work["abstract"]

            # 检查是否有abstract_inverted_index并尝试转换
            if "abstract_inverted_index" in work and work["abstract_inverted_index"]:
                inverted_index = work["abstract_inverted_index"]
                if inverted_index and isinstance(inverted_index, dict):
                    abstract = inverted_index_to_text(inverted_index)
                    if abstract:
                        return abstract

            return None
        except Exception:
            return None

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            # 提取作者信息
            authors = []
            for authorship in paper.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name", "Unknown")
                authors.append(author_name)
            authors_str = ", ".join(authors) if authors else "Unknown authors"

            # 提取期刊/会议信息
            venue = "Unknown Venue"
            primary_location = paper.get("primary_location")
            if primary_location and isinstance(primary_location, dict):
                source = primary_location.get("source")
                if source and isinstance(source, dict):
                    venue = source.get("display_name", venue)

            # 构建论文信息
            title = paper.get("title", "Unknown Title")
            year = paper.get("publication_year", "Unknown Year")
            citations = paper.get("cited_by_count", "N/A")
            doi = paper.get("doi", "")
            abstract = paper.get("abstract", "")

            paper_info = f"""{i + 1}: {title}. {authors_str}. {venue}, {year}.
Number of citations: {citations}"""

            if doi:
                paper_info += f"\nDOI: {doi}"

            if abstract:
                # 限制摘要长度
                abstract_preview = abstract[:500] + "..." if len(abstract) > 500 else abstract
                paper_info += f"\nAbstract: {abstract_preview}"
            else:
                paper_info += "\nAbstract: Not available"

            paper_strings.append(paper_info)

        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    on_backoff=on_backoff,
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    """独立函数，用于直接搜索OpenAlex论文"""
    config = get_open_alex_config()

    # 配置PyAlex
    api_key = config.get("api_key")
    email = config.get("email")

    if api_key:
        pyalex.config.api_key = api_key
    if email:
        pyalex.config.email = email

    if not query:
        return None

    try:
        # 使用PyAlex进行搜索
        results = Works().filter(**{"default.search": query}).get(per_page=result_limit)

        papers = []
        for work in results:
            paper_dict = dict(work)

            # 尝试获取摘要
            if "abstract_inverted_index" in work and work["abstract_inverted_index"]:
                inverted_index = work["abstract_inverted_index"]
                if inverted_index and isinstance(inverted_index, dict):
                    abstract = inverted_index_to_text(inverted_index)
                    if abstract:
                        paper_dict["abstract"] = abstract

            papers.append(paper_dict)

        if not papers:
            return None

        # 按引用数降序排序
        papers.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
        return papers

    except Exception as e:
        print(f"Error searching OpenAlex with PyAlex: {e}")
        return None