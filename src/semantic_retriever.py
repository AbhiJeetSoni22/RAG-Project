"""Semantic Retrieval Module using Sentence Transformers for Phase 5.

This module indexes document chunks using dense semantic embeddings from
Sentence Transformers (default: all-MiniLM-L6-v2), encodes user questions into
the same vector space, computes cosine similarities, and ranks passages by semantic
relevance while preserving all Phase 2 metadata and filtering below a similarity threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, Any] = {}


class SemanticRetrievalError(Exception):
    """Raised when an unrecoverable error occurs during semantic retrieval."""
    pass


def get_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Retrieve or initialize a cached SentenceTransformer model instance.

    Ensures the transformer model is loaded once in memory and reused across queries.

    Args:
        model_name: HuggingFace model identifier (default: 'all-MiniLM-L6-v2').

    Returns:
        SentenceTransformer instance.

    Raises:
        SemanticRetrievalError: If sentence-transformers is not installed or model loading fails.
    """
    if SentenceTransformer is None:
        raise SemanticRetrievalError(
            "The 'sentence-transformers' package is not installed. "
            "Please install it via 'pip install sentence-transformers'."
        )

    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        try:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as exc:
            raise SemanticRetrievalError(
                f"Failed to load SentenceTransformer model '{model_name}': {exc}"
            ) from exc

    return _MODEL_CACHE[model_name]


class SemanticRetriever:
    """Dense Semantic passage retriever powered by Sentence Transformers.

    Generates normalized sentence embeddings for all document chunks once upon
    initialization, caches the embedding matrix in memory, and calculates
    cosine similarity against question embeddings for deterministic Top-K ranking.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model: Optional[Any] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        use_original_text: bool = True,
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Semantic Retriever with document chunks.

        Args:
            chunks: List of chunk dictionaries containing 'original_text' and/or 'cleaned_text'.
            model: Optional pre-loaded SentenceTransformer instance (e.g. from st.cache_resource).
            model_name: Model name if loading via cache (default: 'all-MiniLM-L6-v2').
            use_original_text: If True, encodes 'original_text' (preserves casing, punctuation,
                               and sentence structure which sentence transformers excel at).
                               Falls back to 'cleaned_text' if original_text is missing.
            chunk_embeddings: Optional precomputed/cached numpy array of normalized chunk embeddings.
                              If supplied, skips re-encoding chunk texts.
        """
        self.chunks = list(chunks) if chunks else []
        self.model_name = model_name
        self.use_original_text = use_original_text

        # Load or bind model
        if model is not None:
            self.model = model
        else:
            self.model = get_sentence_transformer(model_name)

        # Extract text representations for chunks
        self.corpus: list[str] = []
        for c in self.chunks:
            if self.use_original_text:
                text = (c.get("original_text") or c.get("cleaned_text", "")).strip()
            else:
                text = (c.get("cleaned_text") or c.get("original_text", "")).strip()
            self.corpus.append(text)

        # Store or generate chunk embeddings
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.is_indexed = False

        if not self.corpus or all(not doc for doc in self.corpus):
            return

        if chunk_embeddings is not None:
            if len(chunk_embeddings) != len(self.chunks):
                raise SemanticRetrievalError(
                    f"Dimension mismatch: chunk_embeddings count ({len(chunk_embeddings)}) "
                    f"does not match chunks count ({len(self.chunks)})."
                )
            self.chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
            self.is_indexed = True
            return

        try:
            # Generate normalized L2 embeddings so cosine similarity equals dot product
            embeddings = self.model.encode(
                self.corpus,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.chunk_embeddings = np.asarray(embeddings, dtype=np.float32)
            self.is_indexed = True
        except Exception as exc:
            raise SemanticRetrievalError(f"Failed to generate chunk embeddings: {exc}") from exc

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Generate a normalized embedding vector for the user's question.

        Args:
            query: User's question string.

        Returns:
            1D numpy array representing the query embedding, or None if query is empty.
        """
        if not query or not query.strip():
            return None

        clean_query = query.strip()
        try:
            embedding = self.model.encode(
                clean_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embedding, dtype=np.float32)
        except Exception as exc:
            raise SemanticRetrievalError(f"Failed to encode query: {exc}") from exc

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.10,
    ) -> list[dict[str, Any]]:
        """Search document chunks using semantic embedding cosine similarity.

        Args:
            query: The user question.
            top_k: Maximum number of top passages to return (default 5).
            min_similarity: Minimum cosine similarity threshold (default 0.10).

        Returns:
            Ranked list of chunk dictionaries with similarity_score descending.
            Each dictionary retains all Phase 2 metadata.
        """
        if (
            not self.chunks
            or not self.is_indexed
            or self.chunk_embeddings is None
            or len(self.chunk_embeddings) == 0
        ):
            return []

        if not query or not query.strip():
            return []

        # Check for query with no alphanumeric characters
        if not any(c.isalnum() for c in query):
            return []

        if top_k <= 0:
            return []

        effective_k = min(top_k, len(self.chunks))

        query_vec = self.encode_query(query)
        if query_vec is None:
            return []

        # Cosine similarity for unit-normalized vectors: dot product
        # query_vec shape: (D,), chunk_embeddings shape: (N, D)
        try:
            similarities = np.dot(self.chunk_embeddings, query_vec).flatten()
        except Exception:
            # Fallback to sklearn cosine_similarity
            similarities = cosine_similarity(
                query_vec.reshape(1, -1), self.chunk_embeddings
            ).flatten()

        # Sort indices descending by similarity score
        sorted_indices = np.argsort(similarities)[::-1]

        results: list[dict[str, Any]] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            # Filter below minimum similarity threshold
            if score < min_similarity:
                continue

            chunk = self.chunks[idx].copy()
            # Store numeric similarity score rounded to 4 decimals for consistency
            chunk["similarity_score"] = round(score, 4)
            results.append(chunk)

            if len(results) >= effective_k:
                break

        return results


def retrieve_semantic_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.10,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Functional interface for semantic retrieval with sentence transformer embeddings.

    Args:
        chunks: List of chunk dictionaries.
        query: User's question.
        top_k: Number of chunks to retrieve.
        min_similarity: Minimum similarity score threshold.
        model: Pre-loaded SentenceTransformer instance (optional).
        model_name: Name of the model to use if model is None.

    Returns:
        List of matching chunk dictionaries ranked by similarity_score.
    """
    retriever = SemanticRetriever(chunks=chunks, model=model, model_name=model_name)
    return retriever.search(query=query, top_k=top_k, min_similarity=min_similarity)
