"""Retrieval Engine module using TF-IDF and Cosine Similarity for Phase 3.

This module indexes document chunks using scikit-learn's TfidfVectorizer,
transforms incoming user questions consistently with the chunk representation,
computes cosine similarity, and ranks document passages by relevance while
preserving all Phase 2 metadata and filtering below a similarity threshold.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import clean_chunk_text


class RetrievalError(Exception):
    """Raised when an unrecoverable error occurs during retrieval."""
    pass


class TFIDFRetriever:
    """TF-IDF and Cosine Similarity based passage retriever.

    Fits a scikit-learn TfidfVectorizer on the document chunks corpus once,
    transforms the chunks into a TF-IDF document matrix, and compares incoming
    user queries against the corpus using cosine similarity.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
    ) -> None:
        """Initialize the retriever with a collection of chunks from Phase 2.

        Args:
            chunks: List of chunk dictionaries containing 'cleaned_text' and 'original_text'.
            ngram_range: Range of n-values for n-grams (default: unigrams and bigrams).
            min_df: Minimum document frequency for vocabulary.
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        """
        self.chunks = list(chunks) if chunks else []
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        # Extract retrieval corpus from cleaned_text (fallback to original_text)
        self.corpus: list[str] = [
            (c.get("cleaned_text") or c.get("original_text", "")).strip()
            for c in self.chunks
        ]

        if not self.corpus or all(not doc for doc in self.corpus):
            self.vectorizer: Optional[TfidfVectorizer] = None
            self.chunk_matrix = None
            self.is_fitted = False
            return

        try:
            # Fit vectorizer on document corpus once
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                sublinear_tf=self.sublinear_tf,
                norm="l2",
            )
            self.chunk_matrix = self.vectorizer.fit_transform(self.corpus)
            self.is_fitted = True
        except Exception as exc:
            raise RetrievalError(f"Failed to fit TF-IDF vectorizer on chunk corpus: {exc}") from exc

    def preprocess_query(self, query: str) -> str:
        """Normalize the user's question consistently with document preprocessing.

        Uses the same clean_chunk_text pipeline (punctuation filtering, lowercase,
        stop-word removal preserving question interrogatives).

        Args:
            query: Raw user query string.

        Returns:
            Preprocessed query string.
        """
        if not query or not query.strip():
            return ""
        return clean_chunk_text(query.strip())

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.10,
    ) -> list[dict[str, Any]]:
        """Search the document chunks for passages relevant to the query.

        Args:
            query: The user question.
            top_k: Maximum number of top results to return (default 5).
            min_similarity: Minimum cosine similarity threshold (default 0.10).

        Returns:
            Ranked list of chunk dictionaries with similarity_score descending.
            Each dictionary retains all Phase 2 metadata.
        """
        if not self.chunks or not self.is_fitted or self.vectorizer is None or self.chunk_matrix is None:
            return []

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # Bound top_k to number of chunks
        effective_k = min(top_k, len(self.chunks))

        # Consistent question preprocessing
        cleaned_query = self.preprocess_query(query)
        if not cleaned_query:
            # Query was only punctuation or empty after filtering
            return []

        # Transform query using the ALREADY FITTED vectorizer (never re-fit)
        query_vec = self.vectorizer.transform([cleaned_query])

        # If query contains no known vocabulary terms, query_vec will be all zeros
        if query_vec.nnz == 0:
            return []

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.chunk_matrix).flatten()

        # Sort indices descending by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        results: list[dict[str, Any]] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            # Filter by minimum similarity threshold
            if score < min_similarity:
                continue

            chunk = self.chunks[idx].copy()
            # Store numeric similarity score (e.g. 0.8742)
            chunk["similarity_score"] = round(score, 4)
            results.append(chunk)

            if len(results) >= effective_k:
                break

        return results


def retrieve_relevant_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.10,
) -> list[dict[str, Any]]:
    """Functional interface to retrieve relevant chunks for a question.

    Args:
        chunks: List of chunk dictionaries.
        query: User's question.
        top_k: Number of chunks to retrieve.
        min_similarity: Minimum similarity score threshold.

    Returns:
        List of matching chunk dictionaries ranked by similarity_score.
    """
    retriever = TFIDFRetriever(chunks)
    return retriever.search(query=query, top_k=top_k, min_similarity=min_similarity)
