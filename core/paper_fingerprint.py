"""
core/paper_fingerprint.py
Extracts a searchable fingerprint (title, abstract, search queries) from
an uploaded research paper's text. Uses the LLM when available for
better accuracy, falls back to heuristics otherwise so the Similarity
Finder still works without Ollama running.
"""

from config import (
    PAPER_TITLE_ABSTRACT_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT,
    SEARCH_QUERIES_PER_PAPER,
)
from utils.logger import get_logger

log = get_logger("PaperFingerprint")

PAPER_START_CHARS = 3000   # enough to cover title + abstract on most papers
_SECTION_HEADINGS = ("introduction", "keywords", "1.", "1 ", "index terms")


def _heuristic_title(text: str) -> str:
    """Fallback title guess: first non-trivial line of the extracted text."""
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and not line.lower().startswith(("abstract", "keywords")):
            return line[:200]
    return "Untitled paper"


def _heuristic_abstract(text: str) -> str:
    """Fallback abstract guess: text after an 'Abstract' heading, else the first long paragraph."""
    lower = text.lower()
    idx = lower.find("abstract")
    if idx != -1:
        start = idx + len("abstract")
        snippet = text[start:start + 1500].strip(" :\n")
        for heading in _SECTION_HEADINGS:
            cut = snippet.lower().find(f"\n{heading}")
            if cut != -1:
                snippet = snippet[:cut]
        if snippet.strip():
            return snippet.strip()[:1500]

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    return paragraphs[0][:1500] if paragraphs else text[:1500]


def _parse_title_abstract(raw: str) -> dict:
    result = {"title": "", "abstract": ""}
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("TITLE:"):
            result["title"] = line.split(":", 1)[1].strip()
        elif upper.startswith("ABSTRACT:"):
            result["abstract"] = line.split(":", 1)[1].strip()
        elif result["abstract"] and line.strip():
            result["abstract"] += " " + line.strip()   # continuation lines
    return result


def extract_title_abstract(
    paper_text: str,
    llm_client=None,
    title_override: str = "",
) -> dict:
    """Returns {"title": str, "abstract": str}."""
    title = title_override.strip() or None

    if llm_client:
        try:
            prompt = PAPER_TITLE_ABSTRACT_PROMPT.format(paper_start=paper_text[:PAPER_START_CHARS])
            raw = llm_client.generate(prompt)
            parsed = _parse_title_abstract(raw)
            return {
                "title": title or parsed.get("title") or _heuristic_title(paper_text),
                "abstract": parsed.get("abstract") or _heuristic_abstract(paper_text),
            }
        except Exception as e:
            log.warning(f"LLM title/abstract extraction failed, using heuristics: {e}")

    return {
        "title": title or _heuristic_title(paper_text),
        "abstract": _heuristic_abstract(paper_text),
    }


def _heuristic_keyword_query(abstract: str) -> str:
    """No-LLM fallback: string together a handful of longer words from the abstract."""
    words = [w.strip(".,;:()[]") for w in abstract.split() if len(w) > 6]
    return " ".join(words[:6])


def generate_search_queries(
    title: str,
    abstract: str,
    domain_hint: str = "",
    llm_client=None,
) -> list[str]:
    """
    Returns a list of search query strings: the raw title (exact-match)
    plus a few LLM-generated queries (or a heuristic keyword query if no
    LLM is available).
    """
    queries = [title] if title else []
    n_llm_queries = max(SEARCH_QUERIES_PER_PAPER - 1, 1)

    if llm_client:
        try:
            prompt = SEARCH_QUERY_GENERATION_PROMPT.format(
                n=n_llm_queries,
                title=title,
                abstract=abstract[:1500],
                domain_hint=domain_hint or "none provided",
            )
            raw = llm_client.generate(prompt)
            for line in raw.splitlines():
                line = line.strip(" -*•\t")
                if line and len(line) > 3:
                    queries.append(line)
        except Exception as e:
            log.warning(f"LLM query generation failed, using heuristic keywords: {e}")

    if len(queries) <= 1:
        for q in (_heuristic_keyword_query(abstract), domain_hint):
            if q:
                queries.append(q)

    seen = set()
    deduped = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q)

    return deduped[:SEARCH_QUERIES_PER_PAPER]
