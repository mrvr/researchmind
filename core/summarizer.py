"""
core/summarizer.py
Merges all processor outputs and generates the final structured
research summary using the local LLM.

This is the final stage of the pipeline — it:
  1. Builds a rich combined context from all processed files
  2. Optionally retrieves most relevant chunks from ChromaDB
  3. Calls the local LLM to produce the structured summary
  4. Returns a clean structured result dict
"""

from utils.logger import get_logger
from config import SUMMARY_PROMPT_TEMPLATE

log = get_logger("Summarizer")

# Max characters to send to LLM (to stay within context window)
MAX_CONTEXT_CHARS = 12_000


def _truncate(text: str, max_chars: int, label: str = "") -> str:
    if len(text) <= max_chars:
        return text
    log.warning(f"Truncating {label} from {len(text):,} to {max_chars:,} chars")
    return text[:max_chars] + f"\n[... truncated — original length {len(text):,} chars]"


def build_combined_content(
    processor_results: list[dict],
    vector_store=None,
    user_context: str = "",
) -> str:
    """
    Merge all processor outputs into a single structured text block.

    Each file type contributes differently:
    - text / document → full content
    - audio           → transcription + LLM summary
    - video           → transcription + LLM gist
    - spreadsheet     → statistical description + LLM insights

    Args:
        processor_results: List of result dicts from all processors
        vector_store:      VectorStore instance (optional — for semantic retrieval)
        user_context:      Research context provided by the user

    Returns:
        Combined multi-source text string
    """
    sections = []
    per_source_budget = MAX_CONTEXT_CHARS // max(len(processor_results), 1)

    for res in processor_results:
        if res.get("error"):
            sections.append(f"=== {res['source']} ===\n[Processing failed: {res['error']}]")
            continue

        source  = res["source"]
        ftype   = res["type"]
        header  = f"=== SOURCE: {source} (type: {ftype.upper()}) ==="

        if ftype in ("text", "document"):
            content = res.get("content", "")
            body = _truncate(content, per_source_budget, source)

        elif ftype == "audio":
            summary = res.get("summary", "")
            transcript_snippet = _truncate(res.get("transcription", ""), 800, f"{source}:transcript")
            body = f"[AUDIO SUMMARY]\n{summary}\n\n[TRANSCRIPT SNIPPET]\n{transcript_snippet}"

        elif ftype == "video":
            gist = res.get("gist", "")
            meta = res.get("metadata", {})
            meta_str = ""
            if meta.get("duration_sec"):
                m, s = divmod(int(meta["duration_sec"]), 60)
                meta_str = f"Duration: {m}m {s}s"
            transcript_snippet = _truncate(res.get("transcription", ""), 800, f"{source}:transcript")
            body = f"[VIDEO METADATA] {meta_str}\n[VIDEO GIST]\n{gist}\n\n[TRANSCRIPT SNIPPET]\n{transcript_snippet}"

        elif ftype == "spreadsheet":
            insights = res.get("insights", "")
            cols     = ", ".join(res.get("columns", [])[:20])
            shape    = res.get("shape", [0, 0])
            body = (
                f"[DATASET] {shape[0]:,} rows × {shape[1]} columns\n"
                f"[COLUMNS] {cols}\n\n"
                f"[DATA INSIGHTS]\n{insights}"
            )
        else:
            body = str(res)

        sections.append(f"{header}\n{body}")

    # If vector store available, add semantically relevant extra context
    if vector_store and user_context:
        log.info("Retrieving semantically relevant chunks from ChromaDB ...")
        relevant = vector_store.query(user_context, n_results=6)
        if relevant:
            rag_parts = [f"  [{r['source']}] {r['text'][:300]}" for r in relevant]
            sections.append(
                "=== SEMANTICALLY RELEVANT PASSAGES (ChromaDB retrieval) ===\n"
                + "\n\n".join(rag_parts)
            )

    return "\n\n" + ("\n\n" + "─"*60 + "\n\n").join(sections)


def generate_final_summary(
    combined_content: str,
    user_context: str,
    llm_client,
) -> dict:
    """
    Call the local LLM with the combined content and generate the
    final structured research summary.

    Returns:
        {
            "topic":       str,
            "overview":    str,
            "keyPoints":   list[str],
            "methodology": str,
            "dataInsights":str,
            "gaps":        str,
            "keywords":    list[str],
            "raw":         str        ← full LLM response
        }
    """
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        user_context=user_context or "No specific context provided.",
        combined_content=_truncate(combined_content, MAX_CONTEXT_CHARS, "combined_content"),
    )

    log.info("Calling LLM for final summary generation ...")
    raw_response = llm_client.generate(prompt)
    log.info(f"  ✓ LLM response received ({len(raw_response):,} chars)")

    return _parse_summary(raw_response)


def _parse_summary(raw: str) -> dict:
    """
    Parse the LLM's structured text response into a clean dict.
    Tries to extract each section by heading keyword.
    Falls back gracefully if parsing fails.
    """
    result = {
        "topic":        "",
        "overview":     "",
        "keyPoints":    [],
        "methodology":  "",
        "dataInsights": "",
        "gaps":         "",
        "keywords":     [],
        "raw":          raw,
    }

    lines = raw.split("\n")
    current_section = None
    buffer = []

    def _flush():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        buffer = []
        return text

    section_map = {
        "main topic":        "topic",
        "overview":          "overview",
        "key points":        "keyPoints",
        "methodology":       "methodology",
        "data insights":     "dataInsights",
        "research gaps":     "gaps",
        "keywords":          "keywords",
    }

    for line in lines:
        lower = line.lower().strip()
        matched = None
        for keyword, key in section_map.items():
            if keyword in lower and (line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "#", "**", "=", "-", "KEY", "MAIN", "OVERVIEW", "METH", "DATA", "RES", "KEY"))):
                matched = key
                break

        if matched:
            if current_section and buffer:
                _store_section(result, current_section, _flush())
            current_section = matched
            # Grab inline text (e.g. "1. MAIN TOPIC: Deep Learning")
            inline = line.split(":", 1)[-1].strip() if ":" in line else ""
            if inline:
                buffer.append(inline)
        else:
            buffer.append(line)

    if current_section and buffer:
        _store_section(result, current_section, _flush())

    # If parsing produced nothing meaningful, use raw as overview
    if not result["overview"] and not result["topic"]:
        result["overview"] = raw
        result["topic"]    = "See overview for details"

    return result


def _store_section(result: dict, key: str, text: str):
    """Store parsed section text into the result dict."""
    if key in ("keyPoints", "keywords"):
        # Parse as bullet list
        items = []
        for line in text.split("\n"):
            line = line.strip().lstrip("•-*·▸▪0123456789. ").strip()
            if line and len(line) > 3:
                items.append(line)
        result[key] = items or [text] if text else []
    else:
        result[key] = text
