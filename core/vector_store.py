"""
core/vector_store.py
ChromaDB integration — stores document chunks as embeddings,
enables semantic retrieval for context-aware summarization.

Uses sentence-transformers for local embedding generation (no API key needed).
"""

from pathlib import Path
from typing import Optional
from utils.logger import get_logger
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

log = get_logger("VectorStore")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       Full text string
        chunk_size: Maximum characters per chunk
        overlap:    Characters to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end   = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


class VectorStore:
    """
    Manages a ChromaDB collection for the ResearchMind pipeline.

    Typical usage:
        store = VectorStore()
        store.add_document("paper.pdf", full_text, metadata={"type": "document"})
        results = store.query("neural networks in EEG", n_results=5)
        store.reset()  # clear before each new session
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_PATH,
        collection_name: str    = CHROMA_COLLECTION_NAME,
        embedding_model: str    = EMBEDDING_MODEL,
    ):
        self.persist_dir     = Path(persist_dir)
        self.collection_name = collection_name
        self._client         = None
        self._collection     = None
        self._embedding_fn   = None
        self._embedding_model_name = embedding_model

    # ── Lazy init ──────────────────────────────────────────────────────────────

    def _init(self):
        """Lazily initialize ChromaDB client and embedding function."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        log.info(f"Initialising ChromaDB at {self.persist_dir} ...")
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        log.info(f"Loading embedding model '{self._embedding_model_name}' ...")
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model_name
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"  ✓ Collection '{self.collection_name}' ready "
                 f"({self._collection.count()} existing chunks)")

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_document(
        self,
        source_name: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Chunk a text and add all chunks to ChromaDB.

        Args:
            source_name: File name (used as metadata and ID prefix)
            text:        Full extracted text
            metadata:    Additional metadata dict (e.g. {"type": "audio"})

        Returns:
            Number of chunks added
        """
        self._init()

        chunks = _chunk_text(text)
        if not chunks:
            log.warning(f"No chunks generated from: {source_name}")
            return 0

        ids, documents, metadatas = [], [], []
        base_meta = {"source": source_name, **(metadata or {})}

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_name}::{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({**base_meta, "chunk_index": i, "total_chunks": len(chunks)})

        # Upsert so re-running doesn't duplicate
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        log.info(f"  ✓ Stored {len(chunks)} chunks from '{source_name}'")
        return len(chunks)

    def query(self, query_text: str, n_results: int = 10) -> list[dict]:
        """
        Semantic search against the stored chunks.

        Returns:
            List of dicts: [{"text": str, "source": str, "chunk_index": int, "distance": float}]
        """
        self._init()

        count = self._collection.count()
        if count == 0:
            log.warning("Vector store is empty — no results to return")
            return []

        n_results = min(n_results, count)
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":        doc,
                "source":      meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "distance":    round(dist, 4),
                "type":        meta.get("type", "unknown"),
            })

        return output

    def get_all_texts_by_source(self) -> dict[str, str]:
        """
        Retrieve all stored chunks grouped by source file.
        Useful for building the final combined content for summarization.

        Returns:
            {"filename.pdf": "full reconstructed text", ...}
        """
        self._init()

        count = self._collection.count()
        if count == 0:
            return {}

        # Fetch all in batches
        all_results = self._collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        # Group by source, sort by chunk_index
        grouped: dict[str, list[tuple[int, str]]] = {}
        for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
            source = meta.get("source", "unknown")
            idx    = meta.get("chunk_index", 0)
            grouped.setdefault(source, []).append((idx, doc))

        # Sort chunks and join
        return {
            source: "\n".join(chunk for _, chunk in sorted(chunks))
            for source, chunks in grouped.items()
        }

    def reset(self):
        """Delete and recreate the collection (use at start of new session)."""
        self._init()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"  ✓ Collection '{self.collection_name}' reset")

    def count(self) -> int:
        """Return total number of chunks stored."""
        self._init()
        return self._collection.count()
