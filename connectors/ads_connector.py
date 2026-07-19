"""
connectors/ads_connector.py
NASA Astrophysics Data System API — the canonical index for AAS journals
(ApJ, AJ, ApJL). Official, free, requires a bearer token (ADS_API_TOKEN).
Auto-disables when no token is configured.
"""

import requests

from connectors.base import CandidatePaper, SearchConnector
from config import ADS_API_URL, ADS_API_TOKEN, SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("ADSConnector")


class ADSConnector(SearchConnector):
    source_name = "NASA ADS (AAS)"

    def is_configured(self) -> bool:
        return bool(ADS_API_TOKEN)

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        if not self.is_configured():
            log.warning("ADS_API_TOKEN not set — skipping NASA ADS search")
            return []

        params = {
            "q": query,
            "rows": max_results,
            "fl": "title,abstract,author,year,doi,bibcode,identifier",
        }
        headers = {"Authorization": f"Bearer {ADS_API_TOKEN}"}

        try:
            resp = requests.get(
                ADS_API_URL, params=params, headers=headers, timeout=SEARCH_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"NASA ADS search failed for '{query}': {e}")
            return []

        candidates = []
        for doc in data.get("response", {}).get("docs", []):
            titles = doc.get("title") or []
            title = titles[0].strip() if titles else ""
            if not title:
                continue

            bibcode = doc.get("bibcode", "")
            doi_list = doc.get("doi") or []
            year_raw = str(doc.get("year", "")).strip()

            candidates.append(CandidatePaper(
                title=title,
                source=self.source_name,
                url=f"https://ui.adsabs.harvard.edu/abs/{bibcode}" if bibcode else "",
                abstract=(doc.get("abstract") or "").strip(),
                authors=doc.get("author", []) or [],
                year=int(year_raw) if year_raw.isdigit() else None,
                doi=doi_list[0] if doi_list else "",
                pdf_url="",
                content_level="abstract_only",
            ))
        return candidates
