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
]
