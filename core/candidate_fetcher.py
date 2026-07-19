"""
core/candidate_fetcher.py
Resolves each candidate's content: downloads open-access full text where
a PDF URL is available, falls back to the abstract already returned by
the connector.
"""

import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

from connectors.base import CandidatePaper
import processors.document_processor as doc_proc
from config import SEARCH_TIMEOUT_SEC
from utils.logger import get_logger

log = get_logger("CandidateFetcher")

MAX_PDF_BYTES = 25 * 1024 * 1024  # 25MB safety cap


def fetch_content(candidate: CandidatePaper) -> str:
    """
    Return the best available text for a candidate: downloaded full text
    if a PDF URL is present and fetchable, otherwise the abstract. Mutates
    candidate.content_level to reflect what was actually obtained.
    """
    if candidate.pdf_url:
        text = _try_download_pdf(candidate.pdf_url)
        if text.strip():
            candidate.content_level = "full_text"
            return text

    candidate.content_level = "abstract_only"
    return candidate.abstract


def _try_download_pdf(pdf_url: str) -> str:
    tmp_path = None
    try:
        resp = requests.get(pdf_url, timeout=SEARCH_TIMEOUT_SEC, stream=True)
        resp.raise_for_status()

        content = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_PDF_BYTES:
                log.warning(f"PDF exceeds {MAX_PDF_BYTES:,} bytes, truncating fetch: {pdf_url}")
                break

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(bytes(content))
            tmp_path = Path(tmp.name)

        result = doc_proc.process(tmp_path)
        return result.get("content", "") if not result.get("error") else ""

    except Exception as e:
        log.warning(f"Could not fetch/extract PDF from {pdf_url}: {e}")
        return ""

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def fetch_all(candidates: list[CandidatePaper], max_workers: int = 6) -> list[tuple[CandidatePaper, str]]:
    """Fetch content for each candidate in parallel, returning (candidate, text) pairs."""
    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        texts = list(executor.map(fetch_content, candidates))

    return list(zip(candidates, texts))
