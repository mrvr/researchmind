"""
api.py — FastAPI server for ResearchMind

Connects the frontend webapp (research_summary_app.html) to the
Python pipeline via a REST endpoint.

Endpoints:
  POST /api/summarize    ← main endpoint (called by frontend's callBackendAPI())
  GET  /api/health       ← check Ollama + dependencies
  GET  /api/models       ← list available Ollama models

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

# ── Ensure the project root is always on sys.path ─────────────────────────────
# Uses os.path.abspath + __file__ — works even when uvicorn imports this module
# from a different working directory (e.g. running from parent folder).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from main import run_pipeline
from core.llm_client import LLMClient
from utils.logger import get_logger

log = get_logger("API")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ResearchMind API",
    description="Local AI research summary pipeline",
    version="1.0.0",
)

# Allow requests from the local frontend (any origin on localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Localhost-only deployment — open is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Main Summarize Endpoint ────────────────────────────────────────────────────
@app.post("/api/summarize")
async def summarize(
    files:   list[UploadFile] = File(..., description="Uploaded research files"),
    context: str              = Form("",  description="Optional research focus context"),
):
    """
    Receive uploaded files, run the full pipeline, return structured summary.

    This is the endpoint called by the frontend's callBackendAPI() function.
    Replace the mock in research_summary_app.html with:

        const response = await fetch('http://localhost:8000/api/summarize', {
            method: 'POST',
            body: formData
        });
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    log.info(f"Received {len(files)} file(s) | context: '{context[:80]}'")

    # Save uploaded files to a temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="researchmind_"))
    saved_paths = []

    try:
        for upload in files:
            dest = temp_dir / upload.filename
            with dest.open("wb") as f:
                content = await upload.read()
                f.write(content)
            saved_paths.append(dest)
            log.info(f"  Saved: {upload.filename} ({len(content):,} bytes)")

        # Run the pipeline
        result = run_pipeline(
            filepaths=saved_paths,
            user_context=context,
            reset_vector_store=True,
        )

        return JSONResponse(content=result)

    except Exception as e:
        log.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        log.info(f"  Cleaned up temp dir: {temp_dir}")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """
    Check the status of all pipeline dependencies.
    Useful for the frontend to verify the backend is ready.
    """
    llm = LLMClient()
    ollama_ok = llm.is_available()

    # Check optional dependencies
    dep_status = {}
    for pkg, label in [
        ("fitz",            "PyMuPDF (PDF)"),
        ("docx",            "python-docx (DOCX)"),
        ("pptx",            "python-pptx (PPTX)"),
        ("odf",             "odfpy (ODT)"),
        ("faster_whisper",  "faster-whisper (Audio/Video)"),
        ("pandas",          "pandas (Spreadsheets)"),
        ("chromadb",        "ChromaDB (Vector DB)"),
        ("sentence_transformers", "sentence-transformers (Embeddings)"),
    ]:
        try:
            __import__(pkg)
            dep_status[label] = "✓ installed"
        except ImportError:
            dep_status[label] = "✗ missing"

    # Check FFmpeg
    import shutil as sh
    dep_status["FFmpeg (Video)"] = "✓ installed" if sh.which("ffmpeg") else "✗ not found in PATH"

    return {
        "status":       "ok" if ollama_ok else "degraded",
        "ollama":       "✓ running" if ollama_ok else "✗ not running",
        "model":        llm.model,
        "dependencies": dep_status,
    }


# ── List Models ────────────────────────────────────────────────────────────────
@app.get("/api/models")
async def list_models():
    """Return available Ollama models."""
    llm    = LLMClient()
    models = llm.list_models()
    return {"models": models, "current": llm.model}


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name":    "ResearchMind API",
        "version": "1.0.0",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/api/health",
    }
