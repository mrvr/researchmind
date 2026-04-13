"""
main.py — ResearchMind Pipeline Orchestrator

This is the central entry point that:
  1. Validates inputs
  2. Routes each file to the correct processor
  3. Stores text chunks in ChromaDB
  4. Merges all outputs
  5. Generates the final LLM summary
  6. Returns a structured result dict

Can be used as:
  - A library (imported by api.py)
  - A CLI tool (run directly: python main.py --files ...)
"""

import argparse
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Ensure the project root is always on sys.path ─────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

from utils.logger import get_logger
from utils.file_router import route_files
from core.llm_client import LLMClient
from core.vector_store import VectorStore
from core.summarizer import build_combined_content, generate_final_summary
import processors.text_processor        as text_proc
import processors.document_processor    as doc_proc
import processors.audio_processor       as audio_proc
import processors.video_processor       as video_proc
import processors.spreadsheet_processor as sheet_proc

log = get_logger("Pipeline")


def run_pipeline(
    filepaths: list[str | Path],
    user_context: str = "",
    reset_vector_store: bool = True,
    parallel_workers: int = 2,
) -> dict:
    """
    Execute the full ResearchMind pipeline.

    Args:
        filepaths:           List of paths to uploaded files
        user_context:        Optional research focus text from the user
        reset_vector_store:  Whether to clear ChromaDB before this run
        parallel_workers:    Number of threads for parallel processing

    Returns:
        {
            "status":          "success" | "error",
            "topic":           str,
            "overview":        str,
            "keyPoints":       list[str],
            "methodology":     str,
            "dataInsights":    str,
            "gaps":            str,
            "keywords":        list[str],
            "sourceFiles":     list[str],
            "processingLog":   list[str],        ← per-file status messages
            "error":           str | None,
            "raw_llm":         str               ← full LLM response
        }
    """
    filepaths   = [Path(f) for f in filepaths]
    proc_log    = []
    all_results = []

    log.info("=" * 60)
    log.info(f"ResearchMind Pipeline — {len(filepaths)} file(s)")
    log.info("=" * 60)

    # ── Step 1: Validate files ─────────────────────────────────────────────────
    valid_files = []
    for fp in filepaths:
        if not fp.exists():
            msg = f"File not found: {fp}"
            log.error(msg); proc_log.append(f"❌ {msg}")
        else:
            valid_files.append(fp)

    if not valid_files:
        return _error_result("No valid files to process.", proc_log)

    # ── Step 2: Check LLM availability ────────────────────────────────────────
    llm = LLMClient()
    if not llm.is_available():
        log.warning("LLM not available — summaries will use raw transcriptions only")
        proc_log.append("⚠️  Ollama LLM not available — using raw text only")
        llm = None  # Pipeline continues without LLM (partial output)

    # ── Step 3: Initialise Vector Store ───────────────────────────────────────
    store = VectorStore()
    if reset_vector_store:
        store.reset()
        proc_log.append("🗄  ChromaDB collection reset for new session")

    # ── Step 4: Route files ────────────────────────────────────────────────────
    groups = route_files(valid_files)
    for category, files in groups.items():
        if files and category != "unsupported":
            log.info(f"  {category.upper()}: {[f.name for f in files]}")
    if groups.get("unsupported"):
        for fp in groups["unsupported"]:
            msg = f"Unsupported file type skipped: {fp.name}"
            log.warning(msg); proc_log.append(f"⚠️  {msg}")

    # ── Step 5: Process files (parallel for independent types) ────────────────
    tasks = []

    for fp in groups.get("text", []):
        tasks.append(("text", fp))
    for fp in groups.get("document", []):
        tasks.append(("document", fp))
    for fp in groups.get("audio", []):
        tasks.append(("audio", fp))
    for fp in groups.get("video", []):
        tasks.append(("video", fp))
    for fp in groups.get("spreadsheet", []):
        tasks.append(("spreadsheet", fp))

    def _process_task(task):
        category, fp = task
        try:
            if category == "text":
                return text_proc.process(fp)
            elif category == "document":
                return doc_proc.process(fp)
            elif category == "audio":
                return audio_proc.process(fp, llm)
            elif category == "video":
                return video_proc.process(fp, llm)
            elif category == "spreadsheet":
                return sheet_proc.process(fp, llm)
        except Exception as e:
            log.error(f"Unhandled error processing {fp.name}: {e}")
            return {"source": fp.name, "type": category, "error": str(e), "content": ""}

    log.info(f"Processing {len(tasks)} file(s) with {parallel_workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_map = {executor.submit(_process_task, t): t for t in tasks}
        for future in as_completed(future_map):
            result = future.result()
            if result:
                all_results.append(result)
                src = result["source"]
                if result.get("error"):
                    proc_log.append(f"❌ {src}: {result['error']}")
                else:
                    chars = result.get("char_count", 0)
                    proc_log.append(f"✅ {src}: {chars:,} chars extracted")

    if not all_results:
        return _error_result("All files failed to process.", proc_log)

    # ── Step 6: Store text in ChromaDB ─────────────────────────────────────────
    log.info("Storing content in ChromaDB ...")
    for res in all_results:
        if res.get("error"):
            continue
        # Pick the best text field per type
        if res["type"] in ("text", "document"):
            text_to_store = res.get("content", "")
        elif res["type"] == "audio":
            text_to_store = res.get("transcription", "") + "\n" + res.get("summary", "")
        elif res["type"] == "video":
            text_to_store = res.get("transcription", "") + "\n" + res.get("gist", "")
        elif res["type"] == "spreadsheet":
            text_to_store = res.get("description", "") + "\n" + res.get("insights", "")
        else:
            text_to_store = str(res)

        if text_to_store.strip():
            chunks_added = store.add_document(
                source_name=res["source"],
                text=text_to_store,
                metadata={"type": res["type"]},
            )
            proc_log.append(f"🗄  {res['source']}: {chunks_added} chunks stored in ChromaDB")

    # ── Step 7: Build combined content for LLM ────────────────────────────────
    log.info("Building combined content for summarization ...")
    combined = build_combined_content(
        processor_results=all_results,
        vector_store=store if user_context else None,
        user_context=user_context,
    )

    # ── Step 8: Generate final summary ────────────────────────────────────────
    if llm:
        summary = generate_final_summary(combined, user_context, llm)
    else:
        # Fallback: return combined content without LLM summarization
        summary = {
            "topic":        "LLM unavailable — raw extraction only",
            "overview":     combined[:2000],
            "keyPoints":    [],
            "methodology":  "",
            "dataInsights": "",
            "gaps":         "",
            "keywords":     [],
            "raw":          combined,
        }
        proc_log.append("⚠️  Final summary skipped (LLM unavailable)")

    log.info("Pipeline complete ✓")
    proc_log.append(f"✅ Pipeline complete — {len(all_results)} file(s) processed")

    return {
        "status":        "success",
        "topic":         summary.get("topic", ""),
        "overview":      summary.get("overview", ""),
        "keyPoints":     summary.get("keyPoints", []),
        "methodology":   summary.get("methodology", ""),
        "dataInsights":  summary.get("dataInsights", ""),
        "gaps":          summary.get("gaps", ""),
        "keywords":      summary.get("keywords", []),
        "sourceFiles":   [r["source"] for r in all_results],
        "processingLog": proc_log,
        "error":         None,
        "raw_llm":       summary.get("raw", ""),
    }


def _error_result(message: str, proc_log: list) -> dict:
    return {
        "status": "error", "topic": "", "overview": "", "keyPoints": [],
        "methodology": "", "dataInsights": "", "gaps": "", "keywords": [],
        "sourceFiles": [], "processingLog": proc_log,
        "error": message, "raw_llm": "",
    }


# ── CLI Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResearchMind — AI Research Summary Pipeline")
    parser.add_argument("--files",   nargs="+", required=True, help="Paths to input files")
    parser.add_argument("--context", default="",              help="Research focus context")
    parser.add_argument("--workers", type=int, default=2,     help="Parallel processing workers")
    parser.add_argument("--no-reset", action="store_true",    help="Keep existing ChromaDB data")
    args = parser.parse_args()

    result = run_pipeline(
        filepaths=args.files,
        user_context=args.context,
        reset_vector_store=not args.no_reset,
        parallel_workers=args.workers,
    )

    print("\n" + "="*60)
    print("RESEARCH SUMMARY OUTPUT")
    print("="*60)
    print(json.dumps(result, indent=2))
