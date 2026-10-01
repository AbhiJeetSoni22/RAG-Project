"""Hybrid Retrieval Module for Phase 7.

This module combines lexical similarity from TF-IDF (Phase 3) and dense semantic
embedding similarity from Sentence Transformers (Phase 5) into a unified hybrid
retrieval engine. It features deterministic min-max score normalization, candidate
union pooling, weighted score fusion with configurable alpha, and full chunk metadata
preservation, feeding directly into the Phase 4 Extractive QA Engine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union
import numpy as np

from src.retriever import TFIDFRetriever, RetrievalError
from src.semantic_retriever import (
    SemanticRetriever,
    SemanticRetrievalError,
    DEFAULT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class HybridRetrievalError(Exception):
    """Raised when an unrecoverable error occurs during hybrid retrieval."""
    pass


class HybridRetriever:
    """Hybrid passage retriever combining lexical TF-IDF and dense Semantic embeddings.

    Reuses existing TFIDFRetriever and SemanticRetriever instances without duplicating
    vectorization or embedding matrices. Computes normalized scores across a candidate
    union pool and applies weighted score fusion:
        HybridScore = α * NormalizedTFIDF + (1 - α) * NormalizedSemantic
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        alpha: float = 0.5,
        tfidf_retriever: Optional[TFIDFRetriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        model: Optional[Any] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize the Hybrid Retriever.

        Args:
            chunks: List of document chunk dictionaries from Phase 2.
            alpha: Weight for lexical TF-IDF score in [0.0, 1.0]. Default is 0.5 (equal weight).
                   alpha = 1.0 corresponds to purely TF-IDF retrieval.
                   alpha = 0.0 corresponds to purely Semantic retrieval.
            tfidf_retriever: Optional pre-initialized TFIDFRetriever instance.
            semantic_retriever: Optional pre-initialized SemanticRetriever instance.
            model: Optional pre-loaded SentenceTransformer instance if initializing semantic retriever.
            model_name: Model identifier if initializing semantic retriever (default: 'all-MiniLM-L6-v2').

        Raises:
            ValueError: If alpha is not between 0.0 and 1.0.
            HybridRetrievalError: If initializing either underlying retriever fails.
        """
        self.chunks = list(chunks) if chunks else []
        self.alpha = self._validate_alpha(alpha)

        # Reuse or initialize TF-IDF retriever
        if tfidf_retriever is not None:
            self.tfidf_retriever = tfidf_retriever
        else:
            try:
                self.tfidf_retriever = TFIDFRetriever(self.chunks)
            except Exception as exc:
                raise HybridRetrievalError(f"Failed to initialize TFIDFRetriever: {exc}") from exc

        # Reuse or initialize Semantic retriever
        if semantic_retriever is not None:
            self.semantic_retriever = semantic_retriever
        else:
            try:
                self.semantic_retriever = SemanticRetriever(
                    chunks=self.chunks,
                    model=model,
                    model_name=model_name,
                )
            except Exception as exc:
                raise HybridRetrievalError(f"Failed to initialize SemanticRetriever: {exc}") from exc

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        """Validate that alpha is a float between 0.0 and 1.0 inclusive.

        Args:
            alpha: Weight parameter.

        Returns:
            Validated float value.

        Raises:
            ValueError: If alpha < 0.0 or alpha > 1.0 or not numeric.
        """
        try:
            val = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Alpha must be a numeric value between 0.0 and 1.0, got: {alpha}") from exc

        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0 inclusive, got: {val}")
        return val

    @staticmethod
    def normalize_scores(
        scores: Union[Sequence[float], dict[str, float]],
    ) -> Union[list[float], dict[str, float]]:
        """Normalize scores using min-max scaling into the [0.0, 1.0] range.

        Formula:
            normalized_score = (score - min_score) / (max_score - min_score)

        Edge Cases:
            - If scores is empty: returns empty list/dict.
            - If max_score == min_score: avoids division-by-zero.
              If max_score > 0, returns 1.0 for all scores (tied positive similarity).
              If max_score == 0, returns 0.0 for all scores (zero similarity).

        Args:
            scores: Sequence of float scores, or a mapping of {chunk_id: score}.

        Returns:
            Normalized scores matching the input structure (list or dict).
        """
        if isinstance(scores, dict):
            if not scores:
                return {}
            vals = list(scores.values())
            min_val = min(vals)
            max_val = max(vals)

            if max_val == min_val:
                fill_val = 1.0 if max_val > 0.0 else 0.0
                return {k: fill_val for k in scores}

            diff = max_val - min_val
            return {k: (v - min_val) / diff for k, v in scores.items()}

        # Sequence of floats
        if not scores:
            return []
        min_val = min(scores)
        max_val = max(scores)

        if max_val == min_val:
            fill_val = 1.0 if max_val > 0.0 else 0.0
            return [fill_val for _ in scores]

        diff = max_val - min_val
        return [(s - min_val) / diff for s in scores]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.10,
        alpha: Optional[float] = None,
        candidate_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search document chunks using weighted hybrid retrieval.

        Pipeline:
        1. Query candidate retrieval from both TF-IDF and Semantic engines.
        2. Candidate union: pool unique chunks retrieved by either engine.
        3. Missing score handling: assign raw score 0.0 for candidates absent from an engine.
        4. Deterministic min-max normalization across the candidate pool for both score channels.
        5. Weighted score fusion: HybridScore = α * NormTFIDF + (1 - α) * NormSemantic.
        6. Relevance thresholding: filter chunks below min_similarity.
        7. Final Top-K ranking by descending hybrid_score with metadata preservation.

        Args:
            query: User's question string.
            top_k: Maximum number of top passages to return (default: 5).
            min_similarity: Minimum similarity score threshold (default: 0.10).
            alpha: Optional query-level override for alpha weight.
            candidate_k: Number of candidates to request from each underlying retriever.
                         Defaults to top_k if not specified.

        Returns:
            Ranked list of chunk dictionaries with similarity_score and explainability metrics.
        """
        if not self.chunks:
            return []

        if not query or not query.strip():
            return []

        # Validate alphanumeric content in query
        if not any(c.isalnum() for c in query):
            return []

        if top_k <= 0:
            return []

        eff_alpha = self.alpha if alpha is None else self._validate_alpha(alpha)
        k_candidates = top_k if candidate_k is None else max(1, candidate_k)
        effective_k = min(top_k, len(self.chunks))

        # Step 1: Retrieve candidate sets from both underlying engines
        try:
            tfidf_candidates = self.tfidf_retriever.search(
                query=query,
                top_k=k_candidates,
                min_similarity=min_similarity,
            )
        except Exception as exc:
            logger.warning(f"TF-IDF retrieval failed during hybrid search: {exc}")
            tfidf_candidates = []

        try:
            semantic_candidates = self.semantic_retriever.search(
                query=query,
                top_k=k_candidates,
                min_similarity=min_similarity,
            )
        except Exception as exc:
            logger.warning(f"Semantic retrieval failed during hybrid search: {exc}")
            semantic_candidates = []

        # If both retrievers returned no candidates, return empty list
        if not tfidf_candidates and not semantic_candidates:
            return []

        # Step 2 & 3: Candidate Union & Missing Score Handling
        # Pool all unique chunks retrieved by either engine.
        candidate_pool: dict[str, dict[str, Any]] = {}
        raw_tfidf: dict[str, float] = {}
        raw_semantic: dict[str, float] = {}

        for c in tfidf_candidates:
            cid = c["chunk_id"]
            candidate_pool[cid] = c.copy()
            raw_tfidf[cid] = float(c.get("similarity_score", 0.0))

        for c in semantic_candidates:
            cid = c["chunk_id"]
            if cid not in candidate_pool:
                candidate_pool[cid] = c.copy()
            raw_semantic[cid] = float(c.get("similarity_score", 0.0))

        # Assign 0.0 for candidates missing from either retriever's candidate set
        for cid in candidate_pool:
            if cid not in raw_tfidf:
                raw_tfidf[cid] = 0.0
            if cid not in raw_semantic:
                raw_semantic[cid] = 0.0

        # Step 4: Deterministic Min-Max Score Normalization across the candidate pool
        norm_tfidf = self.normalize_scores(raw_tfidf)
        norm_semantic = self.normalize_scores(raw_semantic)

        # Step 5 & 8: Weighted Fusion & Metadata Preservation
        scored_candidates: list[dict[str, Any]] = []
        for cid, chunk in candidate_pool.items():
            t_raw = raw_tfidf[cid]
            s_raw = raw_semantic[cid]
            t_norm = norm_tfidf[cid]
            s_norm = norm_semantic[cid]

            # Weighted fusion formula
            hybrid_score = (eff_alpha * t_norm) + ((1.0 - eff_alpha) * s_norm)

            # Step 7: Apply similarity threshold
            if hybrid_score < min_similarity:
                continue

            chunk["tfidf_score"] = round(t_raw, 4)
            chunk["semantic_score"] = round(s_raw, 4)
            chunk["normalized_tfidf_score"] = round(t_norm, 4)
            chunk["normalized_semantic_score"] = round(s_norm, 4)
            chunk["hybrid_score"] = round(hybrid_score, 4)
            # Maintain backward compatibility with ExtractiveQAEngine and Phase 3/5 APIs
            chunk["similarity_score"] = round(hybrid_score, 4)

            scored_candidates.append(chunk)

        # Sort descending by hybrid_score
        # Tie-breaker: prefer higher semantic score, then higher tfidf score
        scored_candidates.sort(
            key=lambda c: (
                c["hybrid_score"],
                c["semantic_score"],
                c["tfidf_score"],
            ),
            reverse=True,
        )

        return scored_candidates[:effective_k]


def retrieve_hybrid_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.10,
    alpha: float = 0.5,
    tfidf_retriever: Optional[TFIDFRetriever] = None,
    semantic_retriever: Optional[SemanticRetriever] = None,
) -> list[dict[str, Any]]:
    """Functional interface for hybrid retrieval combining TF-IDF and Semantic embeddings.

    Args:
        chunks: List of chunk dictionaries from Phase 2.
        query: User's question string.
        top_k: Number of top chunks to retrieve.
        min_similarity: Minimum hybrid similarity threshold.
        alpha: Lexical weight in [0.0, 1.0] (default: 0.5).
        tfidf_retriever: Pre-initialized TFIDFRetriever (optional).
        semantic_retriever: Pre-initialized SemanticRetriever (optional).

    Returns:
        List of ranked chunk dictionaries with hybrid_score and metadata.
    """
    retriever = HybridRetriever(
        chunks=chunks,
        alpha=alpha,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )
    return retriever.search(query=query, top_k=top_k, min_similarity=min_similarity)
