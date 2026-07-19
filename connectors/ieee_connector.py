"""
connectors/ieee_connector.py
IEEE Xplore API — official, requires a free developer key (IEEE_API_KEY).
Returns metadata + abstracts only; IEEE does not serve full text without
a subscription. Auto-disables when no key is configured.
"""

import requests

from connectors.base import CandidatePaper, SearchConnector
from config import IEEE_API_URL, IEEE_API_KEY, SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("IEEEConnector")


class IEEEConnector(SearchConnector):
    source_name = "IEEE Xplore"

    def is_configured(self) -> bool:
        return bool(IEEE_API_KEY)

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        if not self.is_configured():
            log.warning("IEEE_API_KEY not set — skipping IEEE Xplore search")
            return []

        params = {
            "apikey": IEEE_API_KEY,
            "querytext": query,
            "max_records": max_results,
            "format": "json",
        }
        try:
            resp = requests.get(IEEE_API_URL, params=params, timeout=SEARCH_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"IEEE Xplore search failed for '{query}': {e}")
            return []

        candidates = []
        for item in data.get("articles", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            authors = [
                a.get("full_name", "")
                for a in (item.get("authors") or {}).get("authors", [])
            ]

            candidates.append(CandidatePaper(
                title=title,
                source=self.source_name,
                url=item.get("html_url", "") or item.get("pdf_url", ""),
                abstract=(item.get("abstract") or "").strip(),
                authors=[a for a in authors if a],
                year=item.get("publication_year"),
                doi=item.get("doi", ""),
                pdf_url="",   # paywalled — abstract only without a subscription
                content_level="abstract_only",
            ))
        return candidates
