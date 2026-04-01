import requests
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union
from datetime import datetime

import backoff

from ai_scientist.tools.base_tool import BaseTool


# Cache for base URL to avoid repeated file reads
_arxiv_base_url_cache: Optional[str] = None


def get_arxiv_base_url() -> str:
    """Get arXiv API base URL from config.yaml."""
    global _arxiv_base_url_cache

    if _arxiv_base_url_cache is not None:
        return _arxiv_base_url_cache

    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        _arxiv_base_url_cache = config.get("arxiv", {}).get(
            "base_url",
            "http://export.arxiv.org/api/query"
        )
        return _arxiv_base_url_cache
    except Exception:
        _arxiv_base_url_cache = "http://export.arxiv.org/api/query"
        return _arxiv_base_url_cache


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class ArxivSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchArxiv",
        description: str = (
            "Search for relevant preprints using arXiv. "
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

        base_url = get_arxiv_base_url()
        papers = _search_arxiv(base_url, query, self.max_results)

        if not papers:
            return None

        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        """Format papers for display. Assumes papers are in normalized format."""
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = paper.get("authors", [])
            author_names = ", ".join([author.get("name", "Unknown") for author in authors])

            title = paper.get("title", "Unknown Title")
            venue = paper.get("venue", "Unknown Venue")
            year = paper.get("year", "Unknown Year")
            citation_count = paper.get("citationCount", "N/A")
            abstract = paper.get("abstract", "No abstract available.")

            paper_strings.append(
                f"""{i + 1}: {title}. {author_names}. {venue}, {year}.
Number of citations: {citation_count}
Abstract: {abstract}"""
            )
        return "\n\n".join(paper_strings)


def _search_arxiv(base_url: str, query: str, max_results: int) -> Optional[List[Dict]]:
    """Search arXiv and return a list of papers in normalized format."""
    # arXiv API uses ATOM format
    # Search query can include: title, abstract, author, category, etc.
    search_query = f"all:{query}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    rsp = requests.get(base_url, params=params)
    print(f"arXiv Search Status Code: {rsp.status_code}")
    rsp.raise_for_status()

    try:
        root = ET.fromstring(rsp.text)
    except ET.ParseError as e:
        print(f"arXiv XML parse error: {e}")
        return None

    # Define Atom namespace
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []
    for entry in root.findall("atom:entry", ns):
        paper = _parse_arxiv_entry(entry, ns)
        if paper:
            papers.append(paper)

    return papers if papers else None


def _parse_arxiv_entry(entry, ns: Dict) -> Optional[Dict]:
    """Parse a single arXiv entry into normalized format (compatible with Semantic Scholar)."""
    # Title
    title_elem = entry.find("atom:title", ns)
    title = title_elem.text if title_elem is not None and title_elem.text else "Unknown Title"
    # Clean up title (arXiv titles may have extra whitespace)
    title = " ".join(title.split())

    # Authors - match Semantic Scholar format: [{name: str}, ...]
    authors = []
    for author_elem in entry.findall("atom:author", ns):
        name_elem = author_elem.find("atom:name", ns)
        if name_elem is not None and name_elem.text:
            authors.append({"name": name_elem.text})

    # Abstract
    abstract_elem = entry.find("atom:summary", ns)
    abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else "No abstract available."
    abstract = " ".join(abstract.split())  # Clean up whitespace

    # arXiv ID (stored in citationStyles for BibTeX)
    arxiv_id = ""
    pdf_url = ""
    for id_elem in entry.findall("atom:id", ns):
        if id_elem.text:
            url = id_elem.text
            if "arxiv.org" in url:
                # Extract arXiv ID from URL like https://arxiv.org/abs/2101.00123
                arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""
                pdf_url = url.replace("/abs/", "/pdf/")
                break

    # Published date / Year
    published_elem = entry.find("atom:published", ns)
    year = "Unknown Year"
    if published_elem is not None and published_elem.text:
        try:
            # Parse ISO 8601 date like 2021-01-01T00:00:00Z
            published_date = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
            year = str(published_date.year)
        except Exception:
            year = "Unknown Year"

    # Category / Venue
    categories = []
    for category_elem in entry.findall("atom:category", ns):
        term = category_elem.get("term", "")
        if term:
            categories.append(term)
    venue = categories[0] if categories else "Unknown Venue"

    # Generate BibTeX (in citationStyles to match Semantic Scholar format)
    cite_key = f"arxiv_{arxiv_id.replace('/', '_')}" if arxiv_id else f"paper_{hash(title) % 10000}"
    author_names_list = [a.get("name", "Unknown") for a in authors]
    bibtex_authors = " and ".join(author_names_list) if author_names_list else "Unknown"

    bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{bibtex_authors}}},
  journal={{{venue}}},
  year={{{year}}},
  eprint={{{arxiv_id}}},
  url={{{pdf_url if pdf_url else f'https://arxiv.org/abs/{arxiv_id}'}}},
  archivePrefix={{arXiv}}
}}"""

    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": "N/A",  # arXiv doesn't provide citation count directly
        "citationStyles": {
            "bibtex": bibtex
        },
    }


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
    """Standalone function to search for papers using arXiv API.

    Args:
        query: The search query string.
        result_limit: Maximum number of results to return.

    Returns:
        List of paper dictionaries in normalized format.
    """
    if not query:
        return None

    base_url = get_arxiv_base_url()
    papers = _search_arxiv(base_url, query, result_limit)

    time.sleep(0.5)  # Be nice to the API
    return papers
