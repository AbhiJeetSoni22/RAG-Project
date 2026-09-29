"""Streamlit application for Retrieval Based Question Answering (Phases 1, 2, 3 & 4).

Phase 1: Document Ingestion & Text Preprocessing
Phase 2: Text Chunking & Retrieval Data Preparation
Phase 3: Retrieval Engine (TF-IDF + Cosine Similarity)
Phase 4: Question Answering (Extractive, Retrieval-Grounded QA)
"""

from __future__ import annotations

import io
import logging
import streamlit as st

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

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    st.markdown('<span class="badge-phase">Phase 4 &bull; Extractive Question Answering</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Retrieval Based Question Answering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a PDF, retrieve relevant passages via TF-IDF, and extract grounded factual answers with exact page citations.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, int, int, float, int]:
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
        st.markdown("#### 🔍 Phase 3: Retrieval")
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
            * **Safety:** Never answers if no evidence exists in the document
            * **No LLM / Hallucinations:** 100% deterministic NLP extraction
            """
        )

    return chunk_size, chunk_overlap, top_k, min_similarity, max_sentences


def main() -> None:
    """Main application loop."""
    chunk_size, chunk_overlap, top_k, min_similarity, max_sentences = render_sidebar()
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

    # Phase 3: Fit TF-IDF Retriever on Chunk Collection
    try:
        with st.spinner("Step 3: Indexing document chunks into TF-IDF vector space..."):
            retriever = TFIDFRetriever(chunks)
    except RetrievalError as r_err:
        logger.error(f"Retriever initialization error: {r_err}", exc_info=True)
        st.error(f"❌ Retrieval Engine Error: {r_err}")
        return
    except Exception as exc:
        logger.error(f"Unexpected retriever error: {exc}", exc_info=True)
        st.error(f"❌ Failed to build TF-IDF retriever: {exc}")
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
                <div class="status-item"><span class="status-check">✓</span> <b>Extractive QA Engine ready</b></div>
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

    col_query, col_btn = st.columns([5, 1])

    with col_query:
        query_input = st.text_input(
            "Enter your question:",
            placeholder="e.g. What is Natural Language Processing? Or: What are the attendance requirements?",
            key="user_qa_question_input",
            label_visibility="collapsed",
        )

    with col_btn:
        search_clicked = st.button("💬 Ask Question", type="primary", use_container_width=True)

    # Retrieval & QA Pipeline Execution
    if query_input:
        cleaned_q = query_input.strip()

        # Requirement 16: Handle empty / whitespace / punctuation-only question
        if not cleaned_q or not any(c.isalnum() for c in cleaned_q):
            st.warning("⚠️ Please enter a question containing valid words or alphanumeric keywords.")
        else:
            with st.spinner("Retrieving relevant passages and extracting grounded answer..."):
                # Phase 3: Retrieve top-k chunks
                retrieved_results = retriever.search(
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
                st.markdown(
                    f"""
                    <div class="source-container">
                        <span style="font-weight: 700; font-size: 0.82rem; color: #475569; text-transform: uppercase;">Source Attribution:</span>
                        <span class="meta-tag"><b>Document:</b> {qa_result['source']}</span>
                        <span class="meta-tag"><b>Page:</b> {qa_result['page']}</span>
                        <span class="meta-tag"><b>Chunk:</b> <code>{qa_result['chunk_id']}</code></span>
                        <span class="meta-tag"><b>Retrieval Similarity:</b> {qa_result['similarity_score'] * 100:.2f}%</span>
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
            # Requirement 14: Top Retrieved Passages Section
            # =================================================================
            with st.expander("🔍 Top Retrieved Passages (Phase 3 Retrieval Candidates)", expanded=False):
                st.caption(
                    "Inspect all candidate passages retrieved by the Phase 3 TF-IDF engine that were analyzed by the QA engine."
                )

                if not retrieved_results:
                    st.info("ℹ️ No passages met the minimum similarity threshold.")
                else:
                    for rank, res in enumerate(retrieved_results, start=1):
                        score = res["similarity_score"]
                        pct_str = f"{score * 100:.2f}%"

                        if score >= 0.40:
                            sim_class = "similarity-high"
                        elif score >= 0.15:
                            sim_class = "similarity-med"
                        else:
                            sim_class = "similarity-low"

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-header">
                                    <span class="rank-pill">Rank #{rank}</span>
                                    <span class="similarity-pill {sim_class}">Similarity: {pct_str}</span>
                                    <span class="meta-tag"><b>Chunk:</b> <code>{res['chunk_id']}</code></span>
                                    <span class="meta-tag"><b>Page:</b> {res['page']}</span>
                                    <span class="meta-tag"><b>Chars:</b> {res['char_count']}</span>
                                </div>
                                <div class="passage-box">{res['original_text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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
