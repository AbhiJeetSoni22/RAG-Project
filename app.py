"""Streamlit application for Retrieval Based Question Answering (Phases 1, 2, 3 & 4).

Phase 1: Document Ingestion & Text Preprocessing
Phase 2: Text Chunking & Retrieval Data Preparation
Phase 3: Retrieval Engine (TF-IDF + Cosine Similarity)
Phase 4: Question Answering (Extractive, Retrieval-Grounded QA)
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pdf_processor import extract_text_from_pdf, PDFProcessingError
from src.text_processor import (
    process_page_data,
    calculate_document_stats,
    TextProcessingError,
)
from src.chunker import (
    chunk_document,
    calculate_chunk_stats,
    validate_chunks,
    ChunkingError,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.retriever import (
    TFIDFRetriever,
    RetrievalError,
)
from src.qa_engine import (
    ExtractiveQAEngine,
    QAError,
)
from src.semantic_retriever import (
    SemanticRetriever,
    get_sentence_transformer,
    SemanticRetrievalError,
    DEFAULT_MODEL_NAME,
)
from src.hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalError,
)
from src.evaluator import (
    load_evaluation_dataset,
    validate_dataset_against_chunks,
    compare_retrieval_methods,
    export_results_to_csv,
    export_summary_to_csv,
    EvaluationError,
)


# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_cached_transformer(model_name: str = DEFAULT_MODEL_NAME):
    """Load and cache the SentenceTransformer model once across user sessions."""
    return get_sentence_transformer(model_name)


@st.cache_data(show_spinner=False)
def get_cached_chunk_embeddings(_model, chunk_texts: tuple[str, ...]):
    """Generate and cache normalized dense embeddings for document chunks.

    Only recomputes when chunk text contents change. Subsequent question
    searches instantly reuse this cached embedding matrix.
    """
    return _model.encode(
        list(chunk_texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

# Configure Streamlit page
st.set_page_config(
    page_title="Retrieval Based Question Answering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling, clean hierarchy, and premium readability
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .badge-phase {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    .status-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .status-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.45rem;
        font-size: 0.92rem;
        color: #0f172a;
        font-weight: 500;
    }

    .status-item:last-child {
        margin-bottom: 0;
    }

    .status-check {
        color: #10b981;
        font-size: 1.15rem;
        margin-right: 0.6rem;
        font-weight: bold;
    }

    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.95rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Phase 4 Answer Cards */
    .answer-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        border-radius: 0.75rem;
        padding: 1.35rem 1.5rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(22, 163, 74, 0.08);
    }

    .answer-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #15803d;
        margin-bottom: 0.5rem;
    }

    .answer-text {
        font-size: 1.18rem;
        font-weight: 600;
        color: #14532d;
        line-height: 1.6;
    }

    .no-answer-card {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }

    .no-answer-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #b91c1c;
        margin-bottom: 0.35rem;
    }

    .no-answer-msg {
        font-size: 1.05rem;
        font-weight: 600;
        color: #991b1b;
        margin-bottom: 0.25rem;
    }

    .source-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem 0.75rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
    }

    /* Result Card Styling */
    .result-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.75rem;
        padding: 1.15rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .result-card:hover {
        border-color: #94a3b8;
        box-shadow: 0 4px 8px -1px rgba(0, 0, 0, 0.08);
    }

    .result-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.85rem;
    }

    .rank-pill {
        background: #0f172a;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }

    .similarity-pill {
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }

    .similarity-high {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .similarity-med {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }

    .similarity-low {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    .meta-tag {
        background: #f8fafc;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 0.375rem;
        border: 1px solid #e2e8f0;
    }

    .passage-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 0.95rem 1.15rem;
        font-size: 0.95rem;
        line-height: 1.55;
        color: #1e293b;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }

    .chunk-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .chunk-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .chunk-id-tag {
        background: #eff6ff;
        color: #1d4ed8;
        font-family: 'Fira Code', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 0.55rem;
        border-radius: 0.375rem;
        border: 1px solid #bfdbfe;
    }

    .chunk-meta-pill {
        background: #f1f5f9;
        color: #475569;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 0.375rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    """Render page title, badge, and subtitle."""
    st.markdown('<span class="badge-phase">Phase 7 &bull; Hybrid Retrieval (ACTIVE)</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Retrieval Based Question Answering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a PDF, retrieve passages via Hybrid Retrieval (TF-IDF + Semantic Embeddings), extract grounded answers, and objectively benchmark retrieval performance across standard IR metrics.</div>',
        unsafe_allow_html=True,
    )



def render_sidebar() -> tuple[int, int, str, float, int, float, int]:
    """Render sidebar controls for configurable chunking, retrieval, and QA parameters."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Configuration")

        st.markdown("#### 🧩 Phase 2: Chunking")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Target maximum character length per chunk. Paragraphs larger than this will be split by sentence boundaries.",
            key="cfg_chunk_size",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=min(250, chunk_size - 10),
            value=min(DEFAULT_CHUNK_OVERLAP, chunk_size // 5),
            step=10,
            help="Number of characters carried over from the end of a chunk into the start of the next chunk to preserve context at boundaries.",
            key="cfg_chunk_overlap",
        )

        st.markdown("---")
        st.markdown("#### 🔍 Phase 3, 5 & 7: Retrieval")
        retrieval_method = st.radio(
            "Retrieval Method",
            options=["Hybrid", "Semantic Embeddings", "TF-IDF"],
            index=0,
            help="Select retrieval engine: Hybrid (TF-IDF + Semantic Embeddings), Semantic Embeddings (Sentence Transformers all-MiniLM-L6-v2), or Lexical TF-IDF.",
            key="cfg_retrieval_method",
        )

        hybrid_alpha = 0.50
        if retrieval_method == "Hybrid":
            hybrid_alpha = st.slider(
                "Hybrid Lexical Weight (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Weight α for lexical TF-IDF score vs (1 - α) for Semantic embedding score (0.0 = purely Semantic, 1.0 = purely TF-IDF).",
                key="cfg_hybrid_alpha",
            )

        top_k = st.slider(
            "Top-K Retrieved Passages",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Maximum number of relevant passages to retrieve for answer extraction.",
            key="cfg_top_k",
        )

        min_similarity = st.slider(
            "Minimum Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            help="Chunks with cosine similarity below this threshold will be excluded as non-relevant.",
            key="cfg_min_similarity",
        )

        st.markdown("---")
        st.markdown("#### 💡 Phase 4: Answer Extraction")
        max_sentences = st.slider(
            "Max Answer Sentences",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            help="Maximum number of consecutive sentences to extract for the answer span.",
            key="cfg_max_sentences",
        )

        st.markdown("---")
        st.markdown("#### ℹ️ Grounding Principle")
        st.markdown(
            """
            * **Strategy:** Extractive / Retrieval-grounded
            * **Principle:** Retrieve first $\\rightarrow$ Answer only from evidence
            * **Retriever:** Hybrid, Semantic Embeddings, or TF-IDF
            * **Safety:** Never answers if no evidence exists in the document
            * **No LLM / Hallucinations:** 100% deterministic NLP extraction
            """
        )

    return chunk_size, chunk_overlap, retrieval_method, hybrid_alpha, top_k, min_similarity, max_sentences


def main() -> None:
    """Main application loop."""
    chunk_size, chunk_overlap, retrieval_method, hybrid_alpha, top_k, min_similarity, max_sentences = render_sidebar()
    render_header()

    # Upload Section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file to extract, clean, tokenize, chunk, and perform Question Answering.",
    )

    if uploaded_file is None:
        st.info("👋 Please upload a PDF document above to start document ingestion and question answering.")
        return

    # Check for empty file upload (0 bytes)
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("⚠️ The uploaded file is empty (0 bytes). Please upload a valid PDF document.")
        return

    # Phase 1: Ingestion & Text Processing
    try:
        with st.spinner("Step 1: Extracting and preprocessing text page-by-page..."):
            raw_pages = extract_text_from_pdf(io.BytesIO(file_bytes))
            num_pages = len(raw_pages)

            if num_pages == 0:
                st.error("⚠️ The PDF contains 0 pages.")
                return

            processed_pages = process_page_data(raw_pages, remove_punctuation_in_filter=False)
            doc_stats = calculate_document_stats(processed_pages)

    except PDFProcessingError as p_err:
        logger.error(f"PDF extraction error: {p_err}", exc_info=True)
        st.error(f"❌ PDF Processing Error: {p_err}")
        return
    except TextProcessingError as t_err:
        logger.error(f"NLP processing error: {t_err}", exc_info=True)
        st.error(f"❌ Text Preprocessing Error: {t_err}")
        return
    except Exception as exc:
        logger.error(f"Unexpected ingestion error: {exc}", exc_info=True)
        st.error(f"❌ An unexpected error occurred while processing the PDF: {exc}")
        return

    # Check if any text was extracted
    if doc_stats["pages_with_text"] == 0:
        st.warning(
            "⚠️ No extractable text was found in this document. "
            "It may contain scanned images, drawings, or non-extractable text without an OCR layer."
        )
        return

    # Phase 2: Document Chunking
    try:
        with st.spinner("Step 2: Partitioning text into semantic chunks with overlap..."):
            chunks = chunk_document(
                processed_pages=processed_pages,
                source=uploaded_file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            is_valid, validation_issues = validate_chunks(chunks)
            if not is_valid:
                st.warning(f"⚠️ Chunk validation warnings: {', '.join(validation_issues)}")

            chunk_stats = calculate_chunk_stats(chunks, processed_pages)

    except ChunkingError as c_err:
        logger.error(f"Chunking error: {c_err}", exc_info=True)
        st.error(f"❌ Chunking Error: {c_err}")
        return
    except Exception as exc:
        logger.error(f"Unexpected chunking error: {exc}", exc_info=True)
        st.error(f"❌ An unexpected error occurred during chunking: {exc}")
        return

    # Phase 3, 5 & 7: Index Document Chunks for TF-IDF, Semantic, and Hybrid Retrieval
    try:
        with st.spinner("Step 3: Indexing document chunks for TF-IDF, Semantic, and Hybrid retrieval..."):
            tfidf_retriever = TFIDFRetriever(chunks)

            # Initialize Semantic Retriever with cached SentenceTransformer and cached embeddings
            transformer_model = load_cached_transformer()
            chunk_texts_tuple = tuple(
                (c.get("original_text") or c.get("cleaned_text", "")).strip() for c in chunks
            )
            cached_chunk_embs = get_cached_chunk_embeddings(transformer_model, chunk_texts_tuple)
            semantic_retriever = SemanticRetriever(
                chunks=chunks,
                model=transformer_model,
                chunk_embeddings=cached_chunk_embs,
            )

            # Initialize Hybrid Retriever reusing existing instances
            hybrid_retriever = HybridRetriever(
                chunks=chunks,
                alpha=hybrid_alpha,
                tfidf_retriever=tfidf_retriever,
                semantic_retriever=semantic_retriever,
            )
    except (RetrievalError, SemanticRetrievalError, HybridRetrievalError) as r_err:
        logger.error(f"Retriever initialization error: {r_err}", exc_info=True)
        st.error(f"❌ Retrieval Engine Error: {r_err}")
        return
    except Exception as exc:
        logger.error(f"Unexpected retriever error: {exc}", exc_info=True)
        st.error(f"❌ Failed to build retrieval engines: {exc}")
        return

    # Phase 4: Initialize QA Engine
    qa_engine = ExtractiveQAEngine()

    st.markdown("---")

    # Layout: Status Checklist & Document Information
    col_status, col_info = st.columns([1, 2])

    with col_status:
        st.markdown("### 📋 Processing Status")
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-item"><span class="status-check">✓</span> PDF uploaded ({uploaded_file.name})</div>
                <div class="status-item"><span class="status-check">✓</span> Text extracted ({doc_stats['total_pages']} pages)</div>
                <div class="status-item"><span class="status-check">✓</span> Text cleaned & normalized</div>
                <div class="status-item"><span class="status-check">✓</span> Tokenization completed</div>
                <div class="status-item"><span class="status-check">✓</span> Stop-word removal completed</div>
                <div class="status-item"><span class="status-check">✓</span> Text chunking completed ({len(chunks)} chunks)</div>
                <div class="status-item"><span class="status-check">✓</span> TF-IDF vectorizer indexed ({len(chunks)} passages)</div>
                <div class="status-item"><span class="status-check">✓</span> Semantic embeddings indexed ({len(chunks)} passages &bull; all-MiniLM-L6-v2)</div>
                <div class="status-item"><span class="status-check">✓</span> Hybrid retriever indexed ({len(chunks)} passages &bull; α={hybrid_alpha:.2f})</div>
                <div class="status-item"><span class="status-check">✓</span> <b>Extractive QA Engine ready</b></div>
                <div class="status-item"><span class="status-check">✓</span> <b>Retrieval Evaluation Framework ready</b></div>
            </div>

            """,
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown("### 📊 Document Overview")
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">File Name</div>
                    <div class="metric-val" style="font-size: 1.05rem; word-break: break-all;">{uploaded_file.name}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Pages</div>
                    <div class="metric-val">{doc_stats["total_pages"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pages with Text</div>
                    <div class="metric-val">{doc_stats["pages_with_text"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Extracted Chars</div>
                    <div class="metric-val">{doc_stats["total_characters"]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Tokens</div>
                    <div class="metric-val">{doc_stats["total_tokens"]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Indexed Chunks</div>
                    <div class="metric-val" style="color: #4f46e5;">{chunk_stats["total_chunks"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # PHASE 4: QUESTION ANSWERING SECTION
    # =========================================================================
    st.markdown("## 🤖 Question Answering")
    st.caption("Ask questions about the uploaded document. The system retrieves relevant passages and extracts factual answers strictly grounded in the document evidence.")

    col_method, col_query, col_btn = st.columns([2, 4, 1])

    with col_method:
        selected_method = st.selectbox(
            "Retrieval Method",
            options=["Hybrid", "Semantic Embeddings", "TF-IDF"],
            index=0 if retrieval_method == "Hybrid" else (1 if retrieval_method == "Semantic Embeddings" else 2),
            help="Choose which retrieval engine provides candidate passages to the extractive QA engine.",
            key="qa_retrieval_method_select",
        )

    with col_query:
        query_input = st.text_input(
            "Enter your question:",
            placeholder="e.g. How much attendance is required? Or: What is Natural Language Processing?",
            key="user_qa_question_input",
        )

    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("💬 Ask Question", type="primary", use_container_width=True)

    # Retrieval & QA Pipeline Execution
    if query_input:
        cleaned_q = query_input.strip()

        # Requirement 16: Handle empty / whitespace / punctuation-only question
        if not cleaned_q or not any(c.isalnum() for c in cleaned_q):
            st.warning("⚠️ Please enter a question containing valid words or alphanumeric keywords.")
        else:
            with st.spinner(f"Retrieving passages via {selected_method} and extracting grounded answer..."):
                # Select active retriever
                if selected_method == "Hybrid":
                    active_retriever = hybrid_retriever
                elif selected_method == "Semantic Embeddings":
                    active_retriever = semantic_retriever
                else:
                    active_retriever = tfidf_retriever

                # Retrieve top-k chunks
                retrieved_results = active_retriever.search(
                    query=cleaned_q,
                    top_k=top_k,
                    min_similarity=min_similarity,
                )

                # Phase 4: Extract answer from retrieved passages
                qa_result = qa_engine.answer(
                    question=cleaned_q,
                    retrieved_chunks=retrieved_results,
                    max_sentences=max_sentences,
                )

            # =================================================================
            # Requirement 11, 12, 13, 15: Answer Presentation
            # =================================================================
            if qa_result["found"]:
                st.markdown("### 💡 Answer")
                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-title">✓ Grounded Answer (Extracted from Document)</div>
                        <div class="answer-text">{qa_result['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Requirement 12: Source Attribution Section
                score_label = "Hybrid Score" if selected_method == "Hybrid" else "Retrieval Similarity"
                st.markdown(
                    f"""
                    <div class="source-container">
                        <span style="font-weight: 700; font-size: 0.82rem; color: #475569; text-transform: uppercase;">Source Attribution:</span>
                        <span class="meta-tag"><b>Retrieval Method:</b> {selected_method}</span>
                        <span class="meta-tag"><b>Document:</b> {qa_result['source']}</span>
                        <span class="meta-tag"><b>Page:</b> {qa_result['page']}</span>
                        <span class="meta-tag"><b>Chunk:</b> <code>{qa_result['chunk_id']}</code></span>
                        <span class="meta-tag"><b>{score_label}:</b> {qa_result['similarity_score'] * 100:.2f}%</span>
                        <span class="meta-tag"><b>Sentence Score:</b> {qa_result['sentence_score']:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Requirement 13: Evidence Section
                with st.expander("📜 Retrieved Evidence (Ground Truth Passage)", expanded=True):
                    st.caption("The exact original passage from which the answer was extracted.")
                    st.text_area(
                        label="Original Evidence Passage",
                        value=qa_result["evidence"],
                        height=110,
                        key=f"evidence_{qa_result['chunk_id']}",
                        disabled=True,
                    )

            else:
                # Requirement 15: No-Answer UI
                st.markdown("### 💡 Answer")
                st.markdown(
                    f"""
                    <div class="no-answer-card">
                        <div class="no-answer-title">❌ No Answer Found</div>
                        <div class="no-answer-msg">No answer could be found in the uploaded document.</div>
                        <div style="font-size: 0.88rem; color: #7f1d1d; margin-top: 0.4rem;">
                            <b>Reason:</b> {qa_result['reason']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # =================================================================
            # Requirement 14 & Phase 7: Top Retrieved Passages Section
            # =================================================================
            with st.expander(f"🔍 Top Retrieved Passages ({selected_method})", expanded=False):
                st.caption(
                    f"Inspect all candidate passages retrieved by {selected_method} that were analyzed by the QA engine."
                )

                if not retrieved_results:
                    st.info(f"ℹ️ No passages met the minimum similarity threshold ({min_similarity:.2f}).")
                else:
                    for rank, res in enumerate(retrieved_results, start=1):
                        score = res["similarity_score"]
                        pct_str = f"{score * 100:.2f}%"

                        if score >= 0.45:
                            sim_class = "similarity-high"
                        elif score >= 0.15:
                            sim_class = "similarity-med"
                        else:
                            sim_class = "similarity-low"

                        # Section 9: For Hybrid show Rank, Hybrid Score, TF-IDF Score, Semantic Score, Page, Chunk ID, Source
                        if selected_method == "Hybrid":
                            meta_scores_html = f"""
                                <span class="rank-pill">Rank #{rank}</span>
                                <span class="similarity-pill {sim_class}">Hybrid: {res.get('hybrid_score', score):.4f} ({pct_str})</span>
                                <span class="meta-tag"><b>TF-IDF:</b> {res.get('tfidf_score', 0.0):.4f}</span>
                                <span class="meta-tag"><b>Semantic:</b> {res.get('semantic_score', 0.0):.4f}</span>
                                <span class="meta-tag"><b>Page:</b> {res['page']}</span>
                                <span class="meta-tag"><b>Chunk:</b> <code>{res['chunk_id']}</code></span>
                                <span class="meta-tag"><b>Source:</b> {res.get('source', '')}</span>
                            """
                        else:
                            meta_scores_html = f"""
                                <span class="rank-pill">Rank #{rank}</span>
                                <span class="similarity-pill {sim_class}">Similarity: {score:.4f} ({pct_str})</span>
                                <span class="meta-tag"><b>Page:</b> {res['page']}</span>
                                <span class="meta-tag"><b>Chunk:</b> <code>{res['chunk_id']}</code></span>
                                <span class="meta-tag"><b>Source:</b> {res.get('source', '')}</span>
                                <span class="meta-tag"><b>Chars:</b> {res.get('char_count', len(res['original_text']))}</span>
                            """

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-header">
                                    {meta_scores_html}
                                </div>
                                <div class="passage-box">{res['original_text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # =========================================================================
    # PHASE 6: RETRIEVAL EVALUATION & BENCHMARKING SECTION
    # =========================================================================
    st.markdown("## ⚖️ Retrieval Evaluation & Benchmarking")
    st.caption(
        "Objectively benchmark and compare Lexical TF-IDF vs. Dense Semantic Embeddings using "
        "standard Information Retrieval metrics (Hit@K, Precision@K, Recall@K, and Mean Reciprocal Rank) "
        "on a verified ground-truth dataset."
    )

    with st.expander("⚙️ Evaluation Dataset & Parameters", expanded=True):
        col_ds_choice, col_eval_k = st.columns([3, 2])

        with col_ds_choice:
            dataset_option = st.radio(
                "Evaluation Dataset Source",
                options=["Default Project Dataset (evaluation/qa_dataset.json)", "Upload Custom Dataset (.json)"],
                index=0,
                key="eval_dataset_option",
            )

            loaded_questions = None
            eval_dataset_error = None

            if dataset_option == "Default Project Dataset (evaluation/qa_dataset.json)":
                default_ds_path = Path("evaluation/qa_dataset.json")
                if default_ds_path.exists():
                    try:
                        loaded_questions = load_evaluation_dataset(default_ds_path)
                    except Exception as e:
                        eval_dataset_error = str(e)
                else:
                    eval_dataset_error = "Default dataset file evaluation/qa_dataset.json not found."
            else:
                custom_ds_file = st.file_uploader(
                    "Upload JSON Evaluation Dataset",
                    type=["json"],
                    key="custom_eval_dataset_uploader",
                    help="Upload a JSON array containing objects with 'id', 'question', and 'relevant_chunk_ids'.",
                )
                if custom_ds_file is not None:
                    try:
                        loaded_questions = load_evaluation_dataset(custom_ds_file)
                    except Exception as e:
                        eval_dataset_error = str(e)
                else:
                    st.info("ℹ️ Upload a JSON dataset file to begin evaluation.")

        with col_eval_k:
            eval_k_values = st.multiselect(
                "Evaluation Cutoffs (K values)",
                options=[1, 2, 3, 5, 10],
                default=[1, 3, 5],
                key="eval_k_selection",
                help="Select Top-K ranks at which Hit@K, Precision@K, and Recall@K will be measured.",
            )
            eval_min_sim = st.number_input(
                "Evaluation Min Similarity Cutoff",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="eval_min_sim_input",
                help="Similarity threshold below which retrieved candidates are dropped during evaluation. Default 0.0 benchmarks raw ranking.",
            )

        # Validation status
        if eval_dataset_error:
            st.error(f"❌ Dataset Error: {eval_dataset_error}")
        elif loaded_questions:
            val_info = validate_dataset_against_chunks(loaded_questions, chunks)
            cat_list = sorted(list({q.get("category", "general") for q in loaded_questions}))

            if val_info["is_fully_aligned"]:
                st.success(
                    f"✓ Evaluation dataset ready: **{len(loaded_questions)} questions** across "
                    f"**{len(cat_list)} categories** ({', '.join(cat_list)}). "
                    f"**All {val_info['matching_chunks_count']} ground-truth chunks match the active document.**"
                )
            else:
                st.warning(
                    f"⚠️ Evaluation dataset loaded ({len(loaded_questions)} questions, {len(cat_list)} categories). "
                    f"**{val_info['aligned_questions_count']}/{len(loaded_questions)} questions** align with current document chunks. "
                    f"({val_info['missing_chunks_count']} ground-truth chunks from other documents: {', '.join(val_info['missing_chunk_ids'][:4])}...)"
                )

    # Explicit Execution Button to avoid rerunning on every Streamlit interaction
    col_eval_btn, col_eval_status = st.columns([1, 3])
    with col_eval_btn:
        run_eval_clicked = st.button("🚀 Run Retrieval Evaluation", type="primary", key="btn_run_retrieval_eval")

    if run_eval_clicked:
        if not loaded_questions:
            st.error("⚠️ No valid evaluation dataset loaded. Please select or upload a dataset first.")
        elif not eval_k_values:
            st.error("⚠️ Please select at least one K value for evaluation.")
        else:
            with st.spinner("Benchmarking TF-IDF, Semantic, and Hybrid Retrievers across evaluation dataset..."):
                try:
                    comparison_result = compare_retrieval_methods(
                        questions=loaded_questions,
                        retrievers={
                            "TF-IDF": tfidf_retriever,
                            "Semantic": semantic_retriever,
                            "Hybrid": hybrid_retriever,
                        },
                        k_values=eval_k_values,
                        min_similarity=eval_min_sim,
                    )
                    st.session_state["phase6_evaluation_result"] = comparison_result
                    st.success("✓ Evaluation completed successfully!")
                except Exception as exc:
                    logger.error(f"Evaluation error: {exc}", exc_info=True)
                    st.error(f"❌ Evaluation Error: {exc}")

    # Display evaluation results if available in session_state
    if "phase6_evaluation_result" in st.session_state:
        eval_res = st.session_state["phase6_evaluation_result"]

        st.markdown("### 📊 Benchmark Results")

        t_overall, t_cat, t_questions, t_export = st.tabs([
            "📈 Overall Metrics Comparison",
            "🏷️ Category Analysis",
            "🔍 Per-Question Breakdown",
            "📥 CSV Export",
        ])

        with t_overall:
            st.caption("Standardized Information Retrieval metrics comparing Lexical TF-IDF, Dense Semantic, and Hybrid retrieval across the dataset.")

            df_metrics = pd.DataFrame(eval_res.overall_metrics_table)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # High-level metric highlights
            hybrid_summary = eval_res.summaries.get("Hybrid")
            tfidf_summary = eval_res.summaries.get("TF-IDF")
            semantic_summary = eval_res.summaries.get("Semantic")
            highlight_summary = hybrid_summary or semantic_summary

            if highlight_summary:
                eval_max_k = max(eval_res.k_values)
                hl_name = "Hybrid" if hybrid_summary else "Semantic"
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                with c_m1:
                    st.metric(
                        f"Hit@{eval_max_k} ({hl_name})",
                        f"{highlight_summary.hit_rates.get(eval_max_k, 0.0):.4f}",
                    )
                with c_m2:
                    st.metric(
                        f"MRR ({hl_name})",
                        f"{highlight_summary.mrr:.4f}",
                    )
                with c_m3:
                    st.metric(
                        f"Precision@{eval_max_k} ({hl_name})",
                        f"{highlight_summary.precision_at_k.get(eval_max_k, 0.0):.4f}",
                        delta_color="off",
                    )
                with c_m4:
                    st.metric(
                        f"Recall@{eval_max_k} ({hl_name})",
                        f"{highlight_summary.recall_at_k.get(eval_max_k, 0.0):.4f}",
                        delta_color="off",
                    )

        with t_cat:
            st.caption("Performance comparison broken down by linguistic query category.")
            df_cat = pd.DataFrame(eval_res.category_metrics_table)
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

        with t_questions:
            st.caption("Detailed query-level retrieval ranking, hits, and reciprocal ranks.")
            df_pq = pd.DataFrame(eval_res.per_question_table)
            cat_filter = st.selectbox(
                "Filter by Category",
                options=["All Categories"] + sorted(list({q["category"] for q in eval_res.per_question_table})),
                key="eval_per_q_cat_filter",
            )
            if cat_filter != "All Categories":
                df_pq = df_pq[df_pq["category"] == cat_filter]

            st.dataframe(df_pq, use_container_width=True, hide_index=True)

        with t_export:
            st.caption("Download the empirical evaluation measurements in standard CSV format.")

            # Prepare CSV data
            all_q_results = []
            for s in eval_res.summaries.values():
                all_q_results.extend(s.query_results)

            csv_results = export_results_to_csv(all_q_results)
            csv_summary = export_summary_to_csv(eval_res)

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="📥 Download Detailed results.csv",
                    data=csv_results,
                    file_name="results.csv",
                    mime="text/csv",
                    help="Contains per-question metrics (Hit, Precision, Recall, Reciprocal Rank) for every K.",
                    use_container_width=True,
                    key="btn_dl_results_csv",
                )
            with exp_col2:
                st.download_button(
                    label="📥 Download Summary summary.csv",
                    data=csv_summary,
                    file_name="summary.csv",
                    mime="text/csv",
                    help="Contains aggregate summary metrics (Hit@1, Hit@3, Hit@5, Precision@5, Recall@5, MRR).",
                    use_container_width=True,
                    key="btn_dl_summary_csv",
                )

            # Optional save to disk
            if st.button("💾 Save CSV files to evaluation/ directory on disk", key="btn_save_csv_disk"):
                Path("evaluation").mkdir(parents=True, exist_ok=True)
                export_results_to_csv(all_q_results, "evaluation/results.csv")
                export_summary_to_csv(eval_res, "evaluation/summary.csv")
                st.success("✓ Saved to evaluation/results.csv and evaluation/summary.csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")


    # =========================================================================
    # PHASE 2: DOCUMENT CHUNKING SECTION (Maintained for full inspection)
    # =========================================================================
    st.markdown("## 🧩 Document Chunking")
    st.success(
        f"✓ Document processed &bull; ✓ Text chunking completed &bull; **{len(chunks)} chunks created** "
        f"(Chunk size: {chunk_size} chars | Overlap: {chunk_overlap} chars)"
    )

    st.markdown("#### 📈 Chunk Statistics")
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Total Chunks</div>
                <div class="metric-val">{chunk_stats["total_chunks"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Chunk Size</div>
                <div class="metric-val">{chunk_stats["avg_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748b;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Min Chunk Size</div>
                <div class="metric-val">{chunk_stats["min_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748b;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Chunk Size</div>
                <div class="metric-val">{chunk_stats["max_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748b;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pages with Chunks</div>
                <div class="metric-val">{chunk_stats["pages_with_chunks"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Cleaned Size</div>
                <div class="metric-val">{chunk_stats["avg_cleaned_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748b;">chars</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chunk Preview (First 5 chunks)
    preview_count = min(5, len(chunks))
    if preview_count > 0:
        st.markdown(f"#### 🔍 Chunk Preview (First {preview_count} Chunks)")
        preview_cols = st.columns(preview_count)
        for p_i in range(preview_count):
            chk = chunks[p_i]
            with preview_cols[p_i]:
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 0.65rem; height: 100%;">
                        <div style="font-family: 'Fira Code', monospace; font-size: 0.78rem; font-weight: 600; color: #1d4ed8;">{p_i + 1}. {chk['chunk_id']}</div>
                        <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.2rem;">Page {chk['page']} &bull; {chk['char_count']} chars</div>
                        <div style="font-size: 0.75rem; color: #334155; margin-top: 0.35rem; line-height: 1.25; max-height: 55px; overflow: hidden; text-overflow: ellipsis;">
                            {chk['original_text'][:70]}...
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # View Chunks Expandable Section with Pagination
    with st.expander("📑 View Chunks", expanded=False):
        st.caption(
            "Inspect individual chunk contents and metadata. Both original text (for user-facing display) "
            "and cleaned text (for Phase 3 retrieval/vectorization) are preserved."
        )

        if not chunks:
            st.info("ℹ️ No chunks were generated from this document.")
        else:
            col_filter, col_page = st.columns([1, 1])

            with col_filter:
                page_filter_options = ["All Pages"] + [f"Page {p}" for p in sorted(chunk_stats["chunks_per_page"].keys())]
                selected_filter = st.selectbox("Filter Chunks by Page", options=page_filter_options, key="chunk_page_filter")

            filtered_chunks = chunks
            if selected_filter != "All Pages":
                sel_page_num = int(selected_filter.replace("Page ", ""))
                filtered_chunks = [c for c in chunks if c["page"] == sel_page_num]

            chunks_per_view = 5
            total_filtered = len(filtered_chunks)
            total_pages_view = max(1, (total_filtered + chunks_per_view - 1) // chunks_per_view)

            with col_page:
                if total_pages_view > 1:
                    view_page = st.number_input(
                        f"Chunk Page (1 of {total_pages_view})",
                        min_value=1,
                        max_value=total_pages_view,
                        value=1,
                        step=1,
                        key="chunk_view_page",
                    )
                else:
                    view_page = 1

            start_idx = (view_page - 1) * chunks_per_view
            end_idx = min(start_idx + chunks_per_view, total_filtered)

            st.caption(f"Showing chunks **{start_idx + 1}** to **{end_idx}** of **{total_filtered}** matching chunks.")

            for chunk_data in filtered_chunks[start_idx:end_idx]:
                c_id = chunk_data["chunk_id"]
                c_src = chunk_data["source"]
                c_page = chunk_data["page"]
                c_orig = chunk_data["original_text"]
                c_clean = chunk_data["cleaned_text"]
                c_chars = chunk_data["char_count"]
                c_words = chunk_data["word_count"]

                with st.container():
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span class="chunk-id-tag">Chunk: {c_id}</span>
                                <span class="chunk-meta-pill"><b>Source:</b> {c_src}</span>
                                <span class="chunk-meta-pill"><b>Page:</b> {c_page}</span>
                                <span class="chunk-meta-pill"><b>Characters:</b> {c_chars}</span>
                                <span class="chunk-meta-pill"><b>Words:</b> {c_words}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    t_orig, t_clean = st.tabs([
                        f"📄 Original Chunk Text ({c_chars} chars)",
                        f"🔍 Cleaned Chunk Text ({len(c_clean)} chars)",
                    ])

                    with t_orig:
                        st.text_area(
                            label=f"Original Text (Passage to display to user)",
                            value=c_orig,
                            height=120,
                            key=f"chunk_orig_{c_id}",
                            disabled=True,
                        )

                    with t_clean:
                        st.text_area(
                            label=f"Cleaned Text (Normalized for retrieval / vectorization)",
                            value=c_clean,
                            height=120,
                            key=f"chunk_clean_{c_id}",
                            disabled=True,
                        )

    # =========================================================================
    # PHASE 1: PAGE-WISE RAW INSPECTION (Regression support)
    # =========================================================================
    with st.expander("🔍 View Phase 1 Extracted Pages (Page-Level Inspection)", expanded=False):
        st.caption("Inspect the page-by-page extracted text, cleaned text, tokens, and stop-word filtered tokens.")

        if num_pages > 1:
            page_options = [f"Page {p['page']}" for p in processed_pages]
            selected_page_str = st.selectbox("Select Page to Inspect", options=page_options, key="raw_page_select")
            page_num_selected = int(selected_page_str.replace("Page ", ""))
        else:
            page_num_selected = 1

        selected_page_data = next(
            (p for p in processed_pages if p["page"] == page_num_selected),
            processed_pages[0],
        )

        p_orig = selected_page_data["original_text"]
        p_cleaned = selected_page_data["cleaned_text"]
        p_tokens = selected_page_data["tokens"]
        p_filtered = selected_page_data["filtered_tokens"]

        if not p_cleaned.strip():
            st.info(f"ℹ️ Page {page_num_selected} does not contain any extractable text.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Page Characters", len(p_orig))
            c2.metric("Cleaned Chars", len(p_cleaned))
            c3.metric("Tokens", len(p_tokens))
            c4.metric("Filtered Tokens", len(p_filtered))

            t1, t2, t3, t4 = st.tabs([
                "📄 Original Text",
                "🧹 Cleaned Text",
                "🔤 Tokens",
                "🎯 Filtered Tokens (Stopwords Removed)",
            ])

            with t1:
                st.text_area(
                    label="Original Text (as extracted from PDF)",
                    value=p_orig,
                    height=240,
                    key=f"p1_orig_{page_num_selected}",
                    disabled=True,
                )

            with t2:
                st.text_area(
                    label="Cleaned Text (normalized whitespace & lines)",
                    value=p_cleaned,
                    height=240,
                    key=f"p1_cleaned_{page_num_selected}",
                    disabled=True,
                )

            with t3:
                st.caption(f"Total tokens on Page {page_num_selected}: **{len(p_tokens)}**")
                chips_html = "".join([f'<span class="token-chip">{token}</span>' for token in p_tokens])
                st.markdown(f'<div class="token-chip-container">{chips_html}</div>', unsafe_allow_html=True)

            with t4:
                st.caption(f"Total filtered tokens on Page {page_num_selected}: **{len(p_filtered)}** (English stop words excluded)")
                chips_filtered_html = "".join([f'<span class="token-chip filtered">{token}</span>' for token in p_filtered])
                st.markdown(f'<div class="token-chip-container">{chips_filtered_html}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
