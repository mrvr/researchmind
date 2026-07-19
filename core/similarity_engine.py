"""
core/similarity_engine.py
Scores a candidate paper against the uploaded paper using a blend of
embedding-based semantic similarity (chunk-level cosine) and lexical
n-gram overlap (word-shingle Jaccard) — see documents/
similarity-finder-architecture.md for the rationale.
"""

import numpy as np

from core.vector_store import _chunk_text as chunk_text
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    SIMILARITY_SEMANTIC_WEIGHT,
    SIMILARITY_LEXICAL_WEIGHT,
    SIMILARITY_TOP_K_CHUNK_PAIRS,
)
from utils.logger import get_logger

log = get_logger("SimilarityEngine")

_model = None

_EMPTY_SCORE = {"semanticScore": 0.0, "lexicalScore": 0.0, "blendedScore": 0.0, "matchedExcerpts": []}


def _load_model(device: str):
    """Load the model on `device`, verified with a real encode() call — see
    core/vector_store.py's _load_embedding_function() for why this can't
    just trust device availability. Raises on failure so the caller falls back."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device != "cpu":
        model.encode(["gpu warm-up check"])
    return model


def _get_model():
    """Lazily load the shared sentence-transformers model (same one vector_store uses),
    falling back to CPU if the configured GPU device can't actually run it."""
    global _model
    if _model is None:
        device = EMBEDDING_DEVICE
        log.info(f"Loading embedding model '{EMBEDDING_MODEL}' on {device} for similarity scoring...")
        try:
            _model = _load_model(device)
        except Exception as e:
            if device == "cpu":
                raise
            log.warning(f"Embedding model failed on {device} ({e}) — falling back to CPU")
            _model = _load_model("cpu")
            device = "cpu"
        log.info(f"  ✓ Similarity embedding model running on {device}")
    return _model


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def _word_ngrams(text: str, n: int = 5) -> set[tuple]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _lexical_overlap(text_a: str, text_b: str, n: int = 5) -> float:
    grams_a = _word_ngrams(text_a, n)
    grams_b = _word_ngrams(text_b, n)
    if not grams_a or not grams_b:
        return 0.0
    intersection = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return intersection / union if union else 0.0


def score_candidate(paper_text: str, candidate_text: str) -> dict:
    """
    Returns:
        {
            "semanticScore": float,   # 0-1, embedding cosine similarity
            "lexicalScore":  float,   # 0-1, word n-gram Jaccard overlap
            "blendedScore":  float,   # weighted combination — the reported % match
            "matchedExcerpts": [{"paperSnippet", "candidateSnippet", "score"}, ...]
        }
    """
    if not candidate_text.strip() or not paper_text.strip():
        return dict(_EMPTY_SCORE)

    paper_chunks = chunk_text(paper_text)
    candidate_chunks = chunk_text(candidate_text)
    if not paper_chunks or not candidate_chunks:
        return dict(_EMPTY_SCORE)

    try:
        model = _get_model()
        paper_emb = np.asarray(model.encode(paper_chunks))
        candidate_emb = np.asarray(model.encode(candidate_chunks))
    except Exception as e:
        log.error(f"Embedding failed, falling back to lexical-only score: {e}")
        lexical_score = _lexical_overlap(paper_text, candidate_text)
        return {
            "semanticScore": 0.0,
            "lexicalScore": round(lexical_score, 4),
            "blendedScore": round(lexical_score, 4),
            "matchedExcerpts": [],
        }

    sim_matrix = _cosine_sim_matrix(paper_emb, candidate_emb)

    k = min(SIMILARITY_TOP_K_CHUNK_PAIRS, sim_matrix.size)
    flat_indices = np.argsort(sim_matrix.ravel())[::-1][:k]
    semantic_score = float(sim_matrix.ravel()[flat_indices].mean()) if k else 0.0

    lexical_score = _lexical_overlap(paper_text, candidate_text)

    blended = (
        SIMILARITY_SEMANTIC_WEIGHT * semantic_score
        + SIMILARITY_LEXICAL_WEIGHT * lexical_score
    )

    matched_excerpts = []
    p_indices, c_indices = np.unravel_index(flat_indices, sim_matrix.shape)
    for p_idx, c_idx in zip(p_indices, c_indices):
        matched_excerpts.append({
            "paperSnippet": paper_chunks[p_idx][:300],
            "candidateSnippet": candidate_chunks[c_idx][:300],
            "score": round(float(sim_matrix[p_idx, c_idx]), 4),
        })

    return {
        "semanticScore": round(semantic_score, 4),
        "lexicalScore": round(lexical_score, 4),
        "blendedScore": round(min(blended, 1.0), 4),
        "matchedExcerpts": matched_excerpts[:3],
    }
