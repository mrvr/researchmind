"""
connectors/registry.py
Fans a set of search queries out across every configured connector in
parallel, caches results on disk, and dedupes the merged candidate list.
"""

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from connectors.base import CandidatePaper
from connectors.arxiv_connector import ArxivConnector
from connectors.semantic_scholar_connector import SemanticScholarConnector
from connectors.crossref_connector import CrossrefConnector
from connectors.ieee_connector import IEEEConnector
from connectors.ads_connector import ADSConnector
from config import SEARCH_MAX_RESULTS_PER_SOURCE
from utils import search_cache
from utils.logger import get_logger

log = get_logger("ConnectorRegistry")

# Fixed set of implemented connectors. Sources listed only in
# config.EXTRA_SEARCH_SOURCES have no connector class yet — see
# connectors/_template_connector.py to add one.
CONNECTOR_CLASSES = [
    ArxivConnector,
    SemanticScholarConnector,
    CrossrefConnector,
    IEEEConnector,
    ADSConnector,
]


def available_sources() -> dict[str, bool]:
    """Report which connectors are usable right now (used by GET /api/similarity/sources)."""
    result = {}
    for cls in CONNECTOR_CLASSES:
        instance = cls()
        result[instance.source_name] = instance.is_configured()
    return result


def _search_one(connector, query: str, max_results: int) -> list[CandidatePaper]:
    cached = search_cache.get(connector.source_name, query)
    if cached is not None:
        return [CandidatePaper(**c) for c in cached]

    try:
        results = connector.search(query, max_results=max_results)
    except Exception as e:
        log.error(f"{connector.source_name} raised on query '{query}': {e}")
        return []

    search_cache.set(connector.source_name, query, [asdict(c) for c in results])
    return results


def run_all(
    queries: list[str],
    max_results_per_source: int = SEARCH_MAX_RESULTS_PER_SOURCE,
    max_workers: int = 6,
) -> list[CandidatePaper]:
    """
    Run every query against every configured connector in parallel,
    merge, and dedupe the results.
    """
    connectors = [cls() for cls in CONNECTOR_CLASSES]
    active = [c for c in connectors if c.is_configured()]
    skipped = [c.source_name for c in connectors if not c.is_configured()]
    if skipped:
        log.info(f"Skipping unconfigured connectors: {', '.join(skipped)}")

    tasks = [(connector, query) for connector in active for query in queries]
    all_candidates: list[CandidatePaper] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_search_one, connector, query, max_results_per_source): (connector, query)
            for connector, query in tasks
        }
        for future in as_completed(futures):
            connector, query = futures[future]
            try:
                results = future.result()
                all_candidates.extend(results)
            except Exception as e:
                log.error(f"{connector.source_name} task failed for '{query}': {e}")

    return _dedupe(all_candidates)


def _dedupe(candidates: list[CandidatePaper]) -> list[CandidatePaper]:
    seen: dict[str, CandidatePaper] = {}
    for c in candidates:
        key = c.dedupe_key()
        existing = seen.get(key)
        # Prefer the entry with the most content (full text > abstract > blank)
        if existing is None or (
            existing.content_level != "full_text" and c.content_level == "full_text"
        ):
            seen[key] = c
    return list(seen.values())
