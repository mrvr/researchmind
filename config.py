"""
config.py — Central configuration for ResearchMind pipeline.
Optimised for NVIDIA GPU (CUDA) + Ollama on Ubuntu.
Edit this file to tune models, paths, and processing parameters.
"""

import os
import subprocess
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DB_PATH  = BASE_DIR / "chroma_db"
TEMP_DIR        = BASE_DIR / "tmp"

CHROMA_DB_PATH.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── GPU Auto-Detection ─────────────────────────────────────────────────────────
def _detect_gpu() -> dict:
    """
    Probe the NVIDIA GPU and return key specs.
    Falls back gracefully if nvidia-smi is unavailable.

    Returns:
        {
            "available": bool,
            "name": str,
            "vram_mb": int,
            "cuda_version": str,
            "driver_version": str,
        }
    """
    info = {"available": False, "name": "N/A", "vram_mb": 0,
            "cuda_version": "N/A", "driver_version": "N/A"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().split("\n")[0]          # first GPU only
        parts = [p.strip() for p in out.split(",")]
        info["available"]      = True
        info["name"]           = parts[0] if len(parts) > 0 else "Unknown"
        info["vram_mb"]        = int(parts[1]) if len(parts) > 1 else 0
        info["driver_version"] = parts[2] if len(parts) > 2 else "N/A"

        # CUDA version from nvcc or nvidia-smi header
        try:
            cuda_raw = subprocess.check_output(
                ["nvidia-smi"], text=True, timeout=5
            )
            for line in cuda_raw.splitlines():
                if "CUDA Version" in line:
                    info["cuda_version"] = line.split("CUDA Version:")[-1].strip().split()[0]
                    break
        except Exception:
            pass

    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass
    return info

GPU_INFO = _detect_gpu()

# ── Whisper Model Auto-Selection ───────────────────────────────────────────────
# "Let the pipeline decide" → choose based on available VRAM:
#   ≥ 10 GB VRAM  →  large-v3  (best accuracy, ~3GB VRAM)
#   ≥  5 GB VRAM  →  medium    (great balance,  ~1.5GB VRAM)
#   ≥  2 GB VRAM  →  small     (fast + good,    ~500MB VRAM)
#   <  2 GB VRAM  →  base      (minimal,        ~200MB VRAM)
#   No GPU        →  base on CPU (int8, fast enough)
def _auto_whisper_model(vram_mb: int, gpu_available: bool) -> tuple[str, str, str]:
    """Returns (model_size, device, compute_type)."""
    if not gpu_available:
        return "base", "cpu", "int8"
    if vram_mb >= 10_000:
        return "large-v3", "cuda", "float16"
    elif vram_mb >= 5_000:
        return "medium",   "cuda", "float16"
    elif vram_mb >= 2_000:
        return "small",    "cuda", "float16"
    else:
        return "base",     "cuda", "float16"

_env_model = os.getenv("WHISPER_MODEL")           # manual override via env var
if _env_model:
    WHISPER_MODEL_SIZE   = _env_model
    WHISPER_DEVICE       = "cuda" if GPU_INFO["available"] else "cpu"
    WHISPER_COMPUTE_TYPE = "float16" if GPU_INFO["available"] else "int8"
else:
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = _auto_whisper_model(
        GPU_INFO["vram_mb"], GPU_INFO["available"]
    )

WHISPER_LANGUAGE = None    # None = auto-detect language; or e.g. "en", "de"

# ── Ollama LLM ─────────────────────────────────────────────────────────────────
# Ollama automatically uses the NVIDIA GPU — no extra config needed.
# Recommended models by VRAM:
#   ≥ 24 GB  →  mistral:7b-instruct  or  llama3:8b  (best quality)
#   ≥ 12 GB  →  mistral:7b-instruct  or  phi3:medium
#   ≥  6 GB  →  phi3:mini  or  gemma2:2b
#   <  6 GB  →  tinyllama  (fallback)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "mistral")
OLLAMA_TIMEOUT  = 300      # seconds — safe for large models on first load

# ── Embeddings (sentence-transformers, runs on GPU automatically) ──────────────
# CUDA-aware: sentence-transformers detects GPU via PyTorch automatically.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # ~90MB; fast on GPU, fine on CPU too

# ── ChromaDB Collection ────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "researchmind_docs"

# ── Text Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000    # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# ── LLM Prompt Templates ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research analyst. Your task is to analyze
research content and produce structured, insightful summaries suitable for
academic paper writing. Be precise, objective, and highlight key findings,
methodologies, and research gaps."""

SUMMARY_PROMPT_TEMPLATE = """
You are analyzing research content from multiple sources.

RESEARCH CONTEXT PROVIDED BY USER:
{user_context}

CONTENT FROM ALL SOURCES:
{combined_content}

Based on the above, produce a structured research summary with the following sections:

1. MAIN TOPIC: One concise sentence identifying the central research topic.
2. OVERVIEW: 2-3 paragraphs summarizing the core content across all sources.
3. KEY POINTS: 5-8 bullet points of the most important findings, facts, or arguments.
4. METHODOLOGY (if applicable): How was the research conducted?
5. DATA INSIGHTS (if spreadsheet data was analyzed): Key statistics and trends found.
6. RESEARCH GAPS & SUGGESTIONS: What is missing? What should be investigated further?
7. KEYWORDS: 8-10 relevant academic keywords for this topic.

Be specific. Avoid vague language. Cite content types (e.g., "The video lecture discusses...", "The PDF paper states...").
"""

AUDIO_SUMMARY_PROMPT = """
The following is a transcription of an audio recording.
Summarize the key points discussed, the main topic, and any notable conclusions.
Keep the summary concise (3-5 sentences).

TRANSCRIPTION:
{transcription}

SUMMARY:"""

VIDEO_GIST_PROMPT = """
The following is a transcription of a video recording.
Extract: (1) the main topic, (2) key discussion points, (3) any conclusions or recommendations.
Be concise and factual.

TRANSCRIPTION:
{transcription}

GIST:"""

SPREADSHEET_ANALYSIS_PROMPT = """
You are a data analyst. The following is a statistical description of a dataset.
Identify: (1) what the data is about, (2) key patterns or trends, (3) notable outliers or insights,
(4) what conclusions can be drawn for research purposes.

DATASET DESCRIPTION:
{data_description}

ANALYSIS:"""

# ── Similarity Finder (plagiarism check) ───────────────────────────────────────
# Searches external academic sources for content overlap with an uploaded paper.
# The only stage of the app that makes external network calls — see
# documents/similarity-finder-architecture.md for the full design.

SEARCH_CACHE_DIR       = BASE_DIR / "search_cache"
SEARCH_CACHE_DIR.mkdir(exist_ok=True)
SEARCH_CACHE_TTL_HOURS = 168        # 1 week

SEARCH_MAX_RESULTS_PER_SOURCE = 10
SEARCH_TIMEOUT_SEC            = 15
SEARCH_QUERIES_PER_PAPER      = 4   # LLM-generated queries + 1 raw-title query

PLAGIARISM_SIMILARITY_THRESHOLD = 0.60   # >= this → flagged as chance of plagiarism
SIMILARITY_SEMANTIC_WEIGHT      = 0.70   # embedding cosine similarity
SIMILARITY_LEXICAL_WEIGHT       = 0.30   # word n-gram Jaccard overlap
SIMILARITY_TOP_K_CHUNK_PAIRS    = 5      # best-matching chunk pairs averaged per candidate

# Official/free APIs — always attempted.
ARXIV_API_URL            = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API_URL         = "https://api.crossref.org/works"

# Official APIs gated behind a key/token — the connector auto-disables (with
# a log warning) when the corresponding env var is unset.
IEEE_API_URL   = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
IEEE_API_KEY   = os.getenv("IEEE_API_KEY", "")

ADS_API_URL    = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_API_TOKEN  = os.getenv("ADS_API_TOKEN", "")

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # optional, raises rate limit

# Placeholder registry for additional, not-yet-wired-up sources — see
# documents/similarity-finder-architecture.md "Placeholder for additional
# sources". Adding a row here does nothing on its own: a matching
# connectors/<key>_connector.py must exist AND "enabled" must be True.
EXTRA_SEARCH_SOURCES = {
    "pubmed":        {"name": "PubMed / NCBI E-utilities",       "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",         "api_key_env": None,                "access": "official_api", "enabled": False},
    "core":          {"name": "CORE (core.ac.uk)",               "base_url": "https://api.core.ac.uk/v3/search/works/",               "api_key_env": "CORE_API_KEY",      "access": "official_api", "enabled": False},
    "openalex":      {"name": "OpenAlex",                        "base_url": "https://api.openalex.org/works",                        "api_key_env": None,                "access": "official_api", "enabled": False},
    "dblp":          {"name": "DBLP (CS bibliography)",          "base_url": "https://dblp.org/search/publ/api",                      "api_key_env": None,                "access": "official_api", "enabled": False},
    "doaj":          {"name": "DOAJ (Directory of OA Journals)", "base_url": "https://doaj.org/api/search/articles/",                 "api_key_env": None,                "access": "official_api", "enabled": False},
    "springer":      {"name": "Springer Link",                   "base_url": "https://api.springernature.com/meta/v2/",               "api_key_env": "SPRINGER_API_KEY",  "access": "official_api", "enabled": False},
    "sciencedirect": {"name": "Elsevier ScienceDirect",          "base_url": "https://api.elsevier.com/content/search/sciencedirect", "api_key_env": "ELSEVIER_API_KEY",  "access": "official_api", "enabled": False},
    "ssrn":          {"name": "SSRN",                            "base_url": "https://papers.ssrn.com/sol3/",                         "api_key_env": None,                "access": "scrape_only",  "enabled": False},
    "researchgate":  {"name": "ResearchGate",                    "base_url": "https://www.researchgate.net/search",                   "api_key_env": None,                "access": "scrape_only",  "enabled": False},
    "jstor":         {"name": "JSTOR",                           "base_url": "https://www.jstor.org/action/doBasicSearch",            "api_key_env": None,                "access": "scrape_only",  "enabled": False},
}

PAPER_TITLE_ABSTRACT_PROMPT = """
The following is the beginning of a research paper's extracted text.
Identify: (1) the paper's title, (2) its abstract (or the first
substantial paragraph if no abstract section exists).

Respond in exactly this format:
TITLE: <title text>
ABSTRACT: <abstract text>

PAPER TEXT:
{paper_start}
"""

SEARCH_QUERY_GENERATION_PROMPT = """
You are helping search academic databases for papers related to the one below.
Generate {n} concise, targeted search queries (3-8 words each) that would
surface closely related or overlapping research. One per line, no numbering,
no extra commentary.

TITLE: {title}
ABSTRACT: {abstract}
DOMAIN HINT: {domain_hint}

QUERIES:"""
