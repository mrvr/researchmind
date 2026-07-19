"""
connectors/crossref_connector.py
Crossref API — free, keyless, DOI/metadata search across essentially all
publishers (including IEEE, ACM, and AAS journals). Metadata + abstract
only — Crossref does not serve full text.
"""

import requests

from connectors.base import CandidatePaper, SearchConnector
from config import CROSSREF_API_URL, SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("CrossrefConnector")


def _strip_jats(abstract: str) -> str:
    """Crossref abstracts are JATS XML fragments — strip tags for plain text."""
    import re
    return re.sub(r"<[^>]+>", " ", abstract or "").strip()


class CrossrefConnector(SearchConnector):
    source_name = "Crossref"

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        params = {
            "query": query,
            "rows": max_results,
            "select": "title,abstract,author,DOI,URL,published,container-title",
        }
        try:
            resp = requests.get(CROSSREF_API_URL, params=params, timeout=SEARCH_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Crossref search failed for '{query}': {e}")
            return []

        candidates = []
        for item in data.get("message", {}).get("items", []):
            titles = item.get("title") or []
            title = titles[0].strip() if titles else ""
            if not title:
                continue

            authors = []
            for a in item.get("author", []) or []:
                name = " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
                if name:
                    authors.append(name)

            year = None
            date_parts = (item.get("published") or {}).get("date-parts")
            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            candidates.append(CandidatePaper(
                title=title,
                source=self.source_name,
                url=item.get("URL", ""),
                abstract=_strip_jats(item.get("abstract", "")),
                authors=authors,
                year=year,
                doi=item.get("DOI", ""),
                pdf_url="",
                content_level="abstract_only",
            ))
        return candidates
