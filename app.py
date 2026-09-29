"""Streamlit application for Retrieval Based Question Answering (Phases 1 & 2).

Phase 1: Document Ingestion & Text Preprocessing
Phase 2: Text Chunking & Retrieval Data Preparation
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

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Retrieval Based Question Answering",
    page_icon="📄",
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
        color: #1e293b;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .badge-phase {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
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

    .token-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        max-height: 220px;
        overflow-y: auto;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    .token-chip {
        display: inline-block;
        background: #e2e8f0;
        color: #1e293b;
        font-size: 0.8rem;
        font-family: 'Fira Code', monospace;
        padding: 0.2rem 0.5rem;
        border-radius: 0.375rem;
    }

    .token-chip.filtered {
        background: #dbeafe;
        color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    """Render page title, badge, and subtitle."""
    st.markdown('<span class="badge-phase">Phase 2 &bull; Text Chunking & Retrieval Data Preparation</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Retrieval Based Question Answering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a PDF, extract page-level text, and partition it into retrieval-ready semantic chunks.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, int]:
    """Render sidebar controls for configurable chunking parameters."""
    with st.sidebar:
        st.markdown("### ⚙️ Chunking Configuration")
        st.caption("Configure chunk size and overlap parameters for the Phase 2 chunking engine.")

        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Target maximum character length per chunk. Paragraphs larger than this will be split by sentence boundaries.",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=min(250, chunk_size - 10),
            value=min(DEFAULT_CHUNK_OVERLAP, chunk_size // 5),
            step=10,
            help="Number of characters carried over from the end of a chunk into the start of the next chunk to preserve context at boundaries.",
        )

        st.markdown("---")
        st.markdown("#### ℹ️ About Parameters")
        st.markdown(
            """
            * **Chunk Size:** Controls the granularity of retrieved passages.
            * **Chunk Overlap:** Prevents information loss near chunk boundaries.
            * **Hierarchy:** Paragraph $\\rightarrow$ Sentence $\\rightarrow$ Word boundaries.
            """
        )

    return chunk_size, chunk_overlap


def main() -> None:
    """Main application loop."""
    chunk_size, chunk_overlap = render_sidebar()
    render_header()

    # Upload Section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file to extract, clean, tokenize, and chunk its content.",
    )

    if uploaded_file is None:
        st.info("👋 Please upload a PDF document above to start ingestion and chunking.")
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
                <div class="status-item"><span class="status-check">✓</span> <b>Text chunking completed ({len(chunks)} chunks)</b></div>
                <div class="status-item"><span class="status-check">✓</span> Chunk metadata verified</div>
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
                    <div class="metric-label">Total Chunks</div>
                    <div class="metric-val" style="color: #2563eb;">{chunk_stats["total_chunks"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section: Document Chunking Statistics (Requirement 11 & 12)
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

    # Requirement 13: Chunk Preview (First 5 chunks)
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

    # Requirement 12: View Chunks Expandable Section with Pagination
    with st.expander("📑 View Chunks", expanded=True):
        st.caption(
            "Inspect individual chunk contents and metadata. Both original text (for user-facing display) "
            "and cleaned text (for Phase 3 retrieval/vectorization) are preserved."
        )

        if not chunks:
            st.info("ℹ️ No chunks were generated from this document.")
        else:
            # Controls: filter by page or chunk pagination
            col_filter, col_page = st.columns([1, 1])

            with col_filter:
                page_filter_options = ["All Pages"] + [f"Page {p}" for p in sorted(chunk_stats["chunks_per_page"].keys())]
                selected_filter = st.selectbox("Filter Chunks by Page", options=page_filter_options, key="chunk_page_filter")

            filtered_chunks = chunks
            if selected_filter != "All Pages":
                sel_page_num = int(selected_filter.replace("Page ", ""))
                filtered_chunks = [c for c in chunks if c["page"] == sel_page_num]

            # Pagination controls
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

    # Phase 1: Expandable Page-wise Raw Inspection Section (Maintained for full regression support)
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
