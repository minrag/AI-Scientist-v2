import os
import requests
import time
import warnings
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


# Cache for API key to avoid repeated file reads
_open_alex_api_key_cache: Optional[str] = None


def get_open_alex_api_key() -> Optional[str]:
    """Get OpenAlex API key from config.yaml."""
    global _open_alex_api_key_cache

    if _open_alex_api_key_cache is not None:
        return _open_alex_api_key_cache

    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        api_key = config.get("open_alex", {}).get("api_key", "")
        _open_alex_api_key_cache = api_key if api_key else None
        return _open_alex_api_key_cache
    except Exception:
        # No fallback for OpenAlex
        _open_alex_api_key_cache = None
        return _open_alex_api_key_cache


def get_open_alex_base_url() -> str:
    """Get OpenAlex API base URL from config.yaml."""
    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        return config.get("open_alex", {}).get("base_url", "https://api.openalex.org/works")
    except Exception:
        return "https://api.openalex.org/works"


def get_default_search_tool_name() -> str:
    """Get the default search tool name from config.yaml."""
    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        default_tool = config.get("academic_search", {}).get("default_tool", "open_alex")
        # Map tool name to tool class name
        tool_mapping = {
            "open_alex": "SearchOpenAlex",
            "semantic_scholar": "SearchSemanticScholar",
            "pubmed": "SearchPubMed",
        }
        return tool_mapping.get(default_tool, "SearchOpenAlex")
    except Exception:
        return "SearchOpenAlex"


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
        self.OPEN_ALEX_API_KEY = get_open_alex_api_key()
        self.email = self._get_email_from_config()

        if not self.OPEN_ALEX_API_KEY:
            warnings.warn(
                "No OpenAlex API key found. Requests will be subject to rate limits. "
                "Set the api_key in config.yaml under open_alex section."
            )

    def _get_email_from_config(self) -> Optional[str]:
        """Get email from config.yaml for OpenAlex rate limiting."""
        try:
            from ai_scientist.utils.model_config import load_config
            config = load_config()
            return config.get("open_alex", {}).get("email", "")
        except Exception:
            return None

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

        # Build the API URL from config
        base_url = get_open_alex_base_url()

        # Build query parameters - use 'search' for full-text search
        # See: https://docs.openalex.org/how-to-use-the-api/get-lists-of-works/search-works
        params = {
            "search": query,
            "per_page": self.max_results,
            "sort": "cited_by_count:desc",  # Sort by citation count descending
        }

        # Add email for better rate limiting if available
        if self.email:
            params["mailto"] = self.email

        # Add API key if available (as a query parameter)
        if self.OPEN_ALEX_API_KEY:
            params["api_key"] = self.OPEN_ALEX_API_KEY

        headers = {
            "Accept": "application/json"
        }

        rsp = requests.get(
            base_url,
            headers=headers,
            params=params,
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()

        papers = results.get("results", [])
        if not papers:
            return None

        # Normalize papers to Semantic Scholar compatible format
        normalized_papers = [normalize_paper(paper) for paper in papers]
        return normalized_papers

    def format_papers(self, papers: List[Dict]) -> str:
        """Format papers for display. Assumes papers are in normalized format."""
        paper_strings = []
        for i, paper in enumerate(papers):
            # Extract authors from normalized format
            authors = paper.get("authors", [])
            author_names = ", ".join([author.get("name", "Unknown") for author in authors])

            # Extract other fields from normalized format
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


def normalize_paper(paper: Dict) -> Dict:
    """Normalize OpenAlex paper format to match Semantic Scholar format.

    This allows other modules to use a consistent field structure:
    - title: Paper title
    - authors: List of {name: str} dicts
    - venue: Venue/journal name
    - year: Publication year
    - abstract: Abstract text
    - citationCount: Citation count
    - citationStyles: Dict with bibtex string
    """
    # Extract authors - convert to Semantic Scholar format
    authorships = paper.get("authorships", [])
    authors = [
        {"name": authorship.get("author", {}).get("display_name", "Unknown")}
        for authorship in authorships
    ]

    # Extract venue
    primary_location = paper.get("primary_location", {})
    source = primary_location.get("source", {})
    venue = source.get("display_name", "Unknown Venue") if source else "Unknown Venue"

    # Extract year
    year = paper.get("publication_year", "Unknown Year")

    # Extract title
    title = paper.get("title", paper.get("display_name", "Unknown Title"))

    # Extract citation count
    citation_count = paper.get("cited_by_count", 0)

    # Extract abstract
    abstract = paper.get("abstract", "")
    if not abstract:
        # Try OpenAlex inverted index format
        abstract_inverted = paper.get("abstract_inverted_index", {})
        if abstract_inverted:
            abstract_words = []
            for word, indices in abstract_inverted.items():
                for idx in indices:
                    abstract_words.append((idx, word))
            abstract_words.sort(key=lambda x: x[0])
            abstract = " ".join([word for _, word in abstract_words])

    # Generate simple BibTeX if available
    doi = paper.get("doi", "")
    cite_key = doi.replace("https://doi.org/", "").replace("/", "_").replace(".", "_") if doi else f"paper_{hash(title) % 10000}"

    # Build author string for bibtex
    author_names_list = [author.get("name", "Unknown") for author in authors]
    if len(author_names_list) > 0:
        # Format: "Last1, First1 and Last2, First2 and ..."
        bibtex_authors = " and ".join(author_names_list)
    else:
        bibtex_authors = "Unknown"

    bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{bibtex_authors}}},
  journal={{{venue}}},
  year={{{year}}},
  doi={{{doi}}}
}}"""

    return {
        "title": title,
        "authors": authors,
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": citation_count,
        "citationStyles": {
            "bibtex": bibtex
        },
    }


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
    """Standalone function to search for papers using OpenAlex API.

    Args:
        query: The search query string.
        result_limit: Maximum number of results to return.

    Returns:
        List of paper dictionaries in Semantic Scholar compatible format.
    """
    open_alex_api_key = get_open_alex_api_key()
    base_url = get_open_alex_base_url()
    email = None

    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        email = config.get("open_alex", {}).get("email", "")
    except Exception:
        pass

    if not query:
        return None

    # Build query parameters - use 'search' for full-text search
    params = {
        "search": query,
        "per_page": result_limit,
        "sort": "cited_by_count:desc",
    }

    # Add email for better rate limiting if available
    if email:
        params["mailto"] = email

    # Add API key if available
    if open_alex_api_key:
        params["api_key"] = open_alex_api_key

    headers = {
        "Accept": "application/json"
    }

    rsp = requests.get(
        base_url,
        headers=headers,
        params=params,
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(f"Response Content: {rsp.text[:500]}")
    rsp.raise_for_status()
    results = rsp.json()

    papers = results.get("results", [])
    if not papers:
        return None

    # Normalize papers to Semantic Scholar compatible format
    normalized_papers = [normalize_paper(paper) for paper in papers]

    time.sleep(0.5)  # Be nice to the API
    return normalized_papers
