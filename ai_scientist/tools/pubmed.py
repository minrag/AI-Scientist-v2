import requests
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


# Cache for API key to avoid repeated file reads
_pubmed_api_key_cache: Optional[str] = None


def get_pubmed_api_key() -> Optional[str]:
    """Get PubMed API key from config.yaml."""
    global _pubmed_api_key_cache

    if _pubmed_api_key_cache is not None:
        return _pubmed_api_key_cache

    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        api_key = config.get("pubmed", {}).get("api_key", "")
        _pubmed_api_key_cache = api_key if api_key else None
        return _pubmed_api_key_cache
    except Exception:
        _pubmed_api_key_cache = None
        return _pubmed_api_key_cache


def get_pubmed_base_url() -> str:
    """Get PubMed API base URL from config.yaml."""
    try:
        from ai_scientist.utils.model_config import load_config
        config = load_config()
        return config.get("pubmed", {}).get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    except Exception:
        return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class PubMedSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchPubMed",
        description: str = (
            "Search for relevant biomedical literature using PubMed. "
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
        self.PUBMED_API_KEY = get_pubmed_api_key()

        if not self.PUBMED_API_KEY:
            warnings.warn(
                "No PubMed API key found. Requests will be subject to rate limits (3 req/sec). "
                "Set the api_key in config.yaml under pubmed section."
            )

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

        base_url = get_pubmed_base_url()
        pmids = _esearch(base_url, query, self.max_results, self.PUBMED_API_KEY)
        if not pmids:
            return None

        papers = _efetch(base_url, pmids, self.PUBMED_API_KEY)
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


def _esearch(base_url: str, query: str, max_results: int, api_key: Optional[str]) -> Optional[List[str]]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{base_url}/esearch.fcgi"
    rsp = requests.get(url, params=params)
    print(f"PubMed ESearch Status Code: {rsp.status_code}")
    rsp.raise_for_status()

    results = rsp.json()
    id_list = results.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return None

    return id_list


def _efetch(base_url: str, pmids: List[str], api_key: Optional[str]) -> Optional[List[Dict]]:
    """Fetch paper details from PubMed using PMIDs and return normalized papers."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{base_url}/efetch.fcgi"
    rsp = requests.get(url, params=params)
    print(f"PubMed EFetch Status Code: {rsp.status_code}")
    rsp.raise_for_status()

    try:
        root = ET.fromstring(rsp.text)
    except ET.ParseError as e:
        print(f"PubMed XML parse error: {e}")
        return None

    papers = []
    for article in root.findall(".//PubmedArticle"):
        paper = _parse_pubmed_article(article)
        if paper:
            papers.append(paper)

    return papers if papers else None


def _parse_pubmed_article(article) -> Optional[Dict]:
    """Parse a single PubmedArticle XML element into normalized format."""
    medline = article.find("MedlineCitation")
    if medline is None:
        return None

    art = medline.find("Article")
    if art is None:
        return None

    # Title
    title_elem = art.find("ArticleTitle")
    title = title_elem.text if title_elem is not None and title_elem.text else "Unknown Title"

    # Authors
    authors = []
    author_list = art.find("AuthorList")
    if author_list is not None:
        for author_elem in author_list.findall("Author"):
            last = author_elem.find("LastName")
            fore = author_elem.find("ForeName")
            if last is not None and last.text:
                name = last.text
                if fore is not None and fore.text:
                    name = f"{fore.text} {last.text}"
                authors.append({"name": name})

    # Abstract
    abstract_elem = art.find("Abstract")
    abstract = ""
    if abstract_elem is not None:
        abstract_parts = []
        for text_elem in abstract_elem.findall("AbstractText"):
            if text_elem.text:
                label = text_elem.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {text_elem.text}")
                else:
                    abstract_parts.append(text_elem.text)
        abstract = " ".join(abstract_parts)

    # Journal / Venue
    journal = art.find("Journal")
    venue = "Unknown Venue"
    if journal is not None:
        journal_title = journal.find("Title")
        if journal_title is not None and journal_title.text:
            venue = journal_title.text

    # Year
    year = "Unknown Year"
    pub_date = None
    if journal is not None:
        pub_date = journal.find("JournalIssue/PubDate")
    if pub_date is not None:
        year_elem = pub_date.find("Year")
        if year_elem is not None and year_elem.text:
            year = year_elem.text

    # PMID
    pmid_elem = medline.find("PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""

    # DOI
    doi = ""
    article_id_list = article.find("PubmedData/ArticleIdList")
    if article_id_list is not None:
        for aid in article_id_list.findall("ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text
                break

    # Generate BibTeX
    cite_key = f"pmid_{pmid}" if pmid else f"paper_{hash(title) % 10000}"
    author_names_list = [a.get("name", "Unknown") for a in authors]
    bibtex_authors = " and ".join(author_names_list) if author_names_list else "Unknown"

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
        "abstract": abstract if abstract else "No abstract available.",
        "citationCount": "N/A",
        "citationStyles": {
            "bibtex": bibtex
        },
    }


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
    """Standalone function to search for papers using PubMed API.

    Args:
        query: The search query string.
        result_limit: Maximum number of results to return.

    Returns:
        List of paper dictionaries in normalized format.
    """
    if not query:
        return None

    api_key = get_pubmed_api_key()
    base_url = get_pubmed_base_url()

    pmids = _esearch(base_url, query, result_limit, api_key)
    if not pmids:
        return None

    papers = _efetch(base_url, pmids, api_key)

    time.sleep(0.5)  # Be nice to the API
    return papers
