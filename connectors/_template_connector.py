"""
connectors/_template_connector.py
Copy this file to add a new source from config.EXTRA_SEARCH_SOURCES.

Steps:
  1. Copy this file to connectors/<key>_connector.py (key must match an
     EXTRA_SEARCH_SOURCES entry in config.py).
  2. Rename the class, set `source_name`.
  3. Implement search() against EXTRA_SEARCH_SOURCES[key]["base_url"].
  4. Register the class in connectors/registry.py's CONNECTOR_CLASSES.
  5. Flip "enabled": True for that key in config.EXTRA_SEARCH_SOURCES.

Do not do this for an entry marked access="scrape_only" without first
reviewing that source's Terms of Service — that's the same problem that
ruled out direct Google Scholar / ACM Digital Library connectors.
"""

from connectors.base import CandidatePaper, SearchConnector
from config import EXTRA_SEARCH_SOURCES
from utils.logger import get_logger

log = get_logger("TemplateConnector")


class TemplateConnector(SearchConnector):
    source_key = "pubmed"          # must match an EXTRA_SEARCH_SOURCES key
    source_name = "Template Source"

    def is_configured(self) -> bool:
        entry = EXTRA_SEARCH_SOURCES.get(self.source_key, {})
        if not entry.get("enabled"):
            return False
        key_env = entry.get("api_key_env")
        if key_env:
            import os
            return bool(os.getenv(key_env))
        return True

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        if not self.is_configured():
            return []
        # Call EXTRA_SEARCH_SOURCES[self.source_key]["base_url"], map the
        # response into CandidatePaper(title, authors, abstract, url,
        # source, year, doi) and return the list. See arxiv_connector.py /
        # crossref_connector.py for worked examples.
        raise NotImplementedError
