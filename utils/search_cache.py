"""
utils/search_cache.py
On-disk, TTL-based cache for external search connector results — avoids
redundant calls against rate-limited free APIs when the same paper is
re-checked or the same query hits multiple connectors.
"""

import hashlib
import json
import time
from pathlib import Path

from config import SEARCH_CACHE_DIR, SEARCH_CACHE_TTL_HOURS
from utils.logger import get_logger

log = get_logger("SearchCache")

_TTL_SECONDS = SEARCH_CACHE_TTL_HOURS * 3600


def _cache_path(connector_name: str, query: str) -> Path:
    key = hashlib.sha256(f"{connector_name}::{query}".encode("utf-8")).hexdigest()
    return SEARCH_CACHE_DIR / f"{key}.json"


def get(connector_name: str, query: str) -> list[dict] | None:
    """Return cached candidate dicts for this connector+query, or None if
    there's no fresh entry."""
    path = _cache_path(connector_name, query)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"Corrupt cache entry {path.name}, ignoring: {e}")
        return None

    if time.time() - payload.get("cached_at", 0) > _TTL_SECONDS:
        return None

    return payload.get("candidates", [])


def set(connector_name: str, query: str, candidates: list[dict]) -> None:
    """Persist candidate dicts for this connector+query."""
    path = _cache_path(connector_name, query)
    payload = {"cached_at": time.time(), "connector": connector_name, "query": query, "candidates": candidates}
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        log.warning(f"Failed to write cache entry {path.name}: {e}")
