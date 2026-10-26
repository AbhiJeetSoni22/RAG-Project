"""Retrieval Based Question Answering - Core NLP Package.

Uses PEP 562 lazy loading for heavy dependencies (PyTorch, SentenceTransformers)
to ensure lightning-fast startup for lightweight utilities and tests.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.pdf_processor import extract_text_from_pdf, PDFProcessingError
from src.text_processor import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    process_page_data,
    calculate_document_stats,
    TextProcessingError,
)
from src.chunker import (
    chunk_document,
    chunk_page,
    split_text_into_chunks,
    clean_chunk_text,
    validate_chunks,
    calculate_chunk_stats,
    ChunkingError,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MIN_CHUNK_SIZE,
)
from src.retriever import (
    TFIDFRetriever,
    retrieve_relevant_chunks,
    RetrievalError,
)
from src.qa_engine import (
    ExtractiveQAEngine,
    answer_question,
    QAError,
)

# Heavy symbols that will be lazily loaded upon access
_LAZY_MODULE_MAP: dict[str, str] = {
    # Phase 5: Semantic Retrieval (Sentence Transformers / PyTorch)
    "SemanticRetriever": "src.semantic_retriever",
    "retrieve_semantic_chunks": "src.semantic_retriever",
    "SemanticRetrievalError": "src.semantic_retriever",
    # Phase 7: Hybrid Retrieval
    "HybridRetriever": "src.hybrid_retriever",
    "retrieve_hybrid_chunks": "src.hybrid_retriever",
    "HybridRetrievalError": "src.hybrid_retriever",
    # Phase 6: Evaluator
    "compute_hit_at_k": "src.evaluator",
    "compute_precision_at_k": "src.evaluator",
    "compute_recall_at_k": "src.evaluator",
    "compute_reciprocal_rank": "src.evaluator",
    "QueryEvaluationResult": "src.evaluator",
    "CategoryMetric": "src.evaluator",
    "EvaluationSummary": "src.evaluator",
    "ComparisonResult": "src.evaluator",
    "evaluate_retriever": "src.evaluator",
    "compare_retrieval_methods": "src.evaluator",
    "load_evaluation_dataset": "src.evaluator",
    "validate_dataset_against_chunks": "src.evaluator",
    "export_results_to_csv": "src.evaluator",
    "export_summary_to_csv": "src.evaluator",
    "EvaluationError": "src.evaluator",
}


def __getattr__(name: str) -> Any:
    """Lazily load heavy modules upon attribute access."""
    if name in _LAZY_MODULE_MAP:
        mod = importlib.import_module(_LAZY_MODULE_MAP[name])
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Phase 1: PDF extraction & preprocessing
    "extract_text_from_pdf",
    "PDFProcessingError",
    "clean_text",
    "tokenize_text",
    "remove_stopwords",
    "process_page_data",
    "calculate_document_stats",
    "TextProcessingError",
    # Phase 2: Chunking & Retrieval preparation
    "chunk_document",
    "chunk_page",
    "split_text_into_chunks",
    "clean_chunk_text",
    "validate_chunks",
    "calculate_chunk_stats",
    "ChunkingError",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MIN_CHUNK_SIZE",
    # Phase 3: Retrieval Engine
    "TFIDFRetriever",
    "retrieve_relevant_chunks",
    "RetrievalError",
    # Phase 4: Question Answering
    "ExtractiveQAEngine",
    "answer_question",
    "QAError",
    # Phase 5: Semantic Retrieval Enhancement
    "SemanticRetriever",
    "retrieve_semantic_chunks",
    "SemanticRetrievalError",
    # Phase 7: Hybrid Retrieval
    "HybridRetriever",
    "retrieve_hybrid_chunks",
    "HybridRetrievalError",
    # Phase 6: IR Evaluator
    "compute_hit_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_reciprocal_rank",
    "QueryEvaluationResult",
    "CategoryMetric",
    "EvaluationSummary",
    "ComparisonResult",
    "evaluate_retriever",
    "compare_retrieval_methods",
    "load_evaluation_dataset",
    "validate_dataset_against_chunks",
    "export_results_to_csv",
    "export_summary_to_csv",
    "EvaluationError",
]
