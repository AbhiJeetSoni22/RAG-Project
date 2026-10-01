"""Retrieval Based Question Answering - Core NLP Package."""

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

from src.semantic_retriever import (
    SemanticRetriever,
    retrieve_semantic_chunks,
    SemanticRetrievalError,
)
from src.hybrid_retriever import (
    HybridRetriever,
    retrieve_hybrid_chunks,
    HybridRetrievalError,
)

from src.evaluator import (
    compute_hit_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    QueryEvaluationResult,
    CategoryMetric,
    EvaluationSummary,
    ComparisonResult,
    evaluate_retriever,
    compare_retrieval_methods,
    load_evaluation_dataset,
    validate_dataset_against_chunks,
    export_results_to_csv,
    export_summary_to_csv,
    EvaluationError,
)

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
    # Phase 6: Retrieval Evaluation
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

