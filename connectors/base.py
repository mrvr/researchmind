"""
connectors/base.py
Shared interface every search connector implements, and the candidate
result shape they all return.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CandidatePaper:
    """A single search result from an external academic source."""

    title:        str
    source:       str                  # connector name, e.g. "arXiv"
    url:          str        = ""
    abstract:     str        = ""
    authors:      list[str]  = field(default_factory=list)
    year:         Optional[int] = None
    doi:          str        = ""
    pdf_url:      str        = ""       # open-access full text, if any
    content_level: str       = "abstract_only"   # "full_text" | "abstract_only"

    def dedupe_key(self) -> str:
        """Key used by the registry to merge duplicate candidates across sources."""
        if self.doi:
            return f"doi:{self.doi.strip().lower()}"
        normalized = " ".join(self.title.lower().split())
        return f"title:{normalized}"


class SearchConnector:
    """
    Base class for all search connectors.

    Subclasses set `source_name` and implement `search()`. `is_configured()`
    lets the registry skip connectors that need an API key/token that isn't
    set, without treating that as an error.
    """

    source_name: str = "unknown"

    def is_configured(self) -> bool:
        """Return True if this connector has everything it needs to run."""
        return True

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        """Run one search query and return candidate papers. Must not raise —
        catch and log connector-specific errors, returning [] on failure."""
        raise NotImplementedError
