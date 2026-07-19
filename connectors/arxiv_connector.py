"""
connectors/arxiv_connector.py
arXiv API — free, keyless, Atom XML response. Full-text PDFs are open
access, so candidates from this connector get content_level="full_text".
"""

import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

from connectors.base import CandidatePaper, SearchConnector
from config import ARXIV_API_URL, SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("ArxivConnector")

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(SearchConnector):
    source_name = "arXiv"

    def search(self, query: str, max_results: int = 10) -> list[CandidatePaper]:
        url = (
            f"{ARXIV_API_URL}?search_query=all:{quote(query)}"
            f"&start=0&max_results={max_results}"
        )
        try:
            resp = requests.get(url, timeout=SEARCH_TIMEOUT_SEC)
            resp.raise_for_status()
            return self._parse(resp.text)
        except Exception as e:
            log.error(f"arXiv search failed for '{query}': {e}")
            return []

    def _parse(self, xml_text: str) -> list[CandidatePaper]:
        candidates = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            log.error(f"arXiv response parse error: {e}")
            return []

        for entry in root.findall(f"{_ATOM_NS}entry"):
            title = (entry.findtext(f"{_ATOM_NS}title") or "").strip().replace("\n", " ")
            if not title:
                continue
            summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip().replace("\n", " ")
            arxiv_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()

            authors = [
                (a.findtext(f"{_ATOM_NS}name") or "").strip()
                for a in entry.findall(f"{_ATOM_NS}author")
            ]

            published = entry.findtext(f"{_ATOM_NS}published") or ""
            year = int(published[:4]) if published[:4].isdigit() else None

            pdf_url = ""
            for link in entry.findall(f"{_ATOM_NS}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break

            candidates.append(CandidatePaper(
                title=title,
                source=self.source_name,
                url=arxiv_id,
                abstract=summary,
                authors=[a for a in authors if a],
                year=year,
                doi="",
                pdf_url=pdf_url,
                content_level="full_text" if pdf_url else "abstract_only",
            ))
        return candidates
