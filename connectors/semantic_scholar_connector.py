"""
connectors/semantic_scholar_connector.py
Semantic Scholar Academic Graph API — free, generous rate limits without a
key (higher limits with SEMANTIC_SCHOLAR_API_KEY). Broad cross-publisher
coverage — indexes most IEEE/ACM/journal metadata as a fallback.
"""

import requests

from connectors.base import CandidatePaper, SearchConnector
from config import SEMANTIC_SCHOLAR_API_URL, SEMANTIC_SCHOLAR_API_KEY, SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("SemanticScholarConnector")


class SemanticScholarConnector(SearchConnector):
    source_name = "Semantic Scholar"

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,externalIds,openAccessPdf,url",
        }
        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

        try:
            resp = requests.get(
                SEMANTIC_SCHOLAR_API_URL, params=params, headers=headers,
                timeout=SEARCH_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Semantic Scholar search failed for '{query}': {e}")
            return []

        candidates = []
        for item in data.get("data", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            authors = [a.get("name", "") for a in (item.get("authors") or [])]
            external_ids = item.get("externalIds") or {}
            oa_pdf = (item.get("openAccessPdf") or {}).get("url", "")

            candidates.append(CandidatePaper(
                title=title,
                source=self.source_name,
                url=item.get("url", ""),
                abstract=(item.get("abstract") or "").strip(),
                authors=[a for a in authors if a],
                year=item.get("year"),
                doi=external_ids.get("DOI", ""),
                pdf_url=oa_pdf,
                content_level="full_text" if oa_pdf else "abstract_only",
            ))
        return candidates
