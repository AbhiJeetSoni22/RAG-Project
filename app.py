"""Streamlit application for Phase 1: Document Ingestion & Text Preprocessing."""

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

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Retrieval Based Question Answering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern styling and readability
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
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.75rem;
        font-weight: 400;
    }

    .badge-phase {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
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
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
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
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
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
    st.markdown('<span class="badge-phase">Phase 1 &bull; Document Ingestion & Text Preprocessing</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Retrieval Based Question Answering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a PDF and preprocess its content for question answering.</div>', unsafe_allow_html=True)


def main() -> None:
    """Main application loop."""
    render_header()

    # Upload Section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file to extract and preprocess page-by-page text.",
    )

    if uploaded_file is None:
        st.info("👋 Please upload a PDF document above to start ingestion and preprocessing.")
        return

    # Check for empty file upload (0 bytes)
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("⚠️ The uploaded file is empty (0 bytes). Please upload a valid PDF document.")
        return

    # Processing pipeline
    try:
        with st.spinner("Processing document (extracting, cleaning, tokenizing)..."):
            # Step 1: Extract text page-by-page
            raw_pages = extract_text_from_pdf(io.BytesIO(file_bytes))
            num_pages = len(raw_pages)

            if num_pages == 0:
                st.error("⚠️ The PDF contains 0 pages.")
                return

            # Step 2: Clean, Tokenize, and Remove Stopwords
            processed_pages = process_page_data(raw_pages, remove_punctuation_in_filter=False)
            stats = calculate_document_stats(processed_pages)

    except PDFProcessingError as p_err:
        logger.error(f"PDF extraction error: {p_err}", exc_info=True)
        st.error(f"❌ PDF Processing Error: {p_err}")
        return
    except TextProcessingError as t_err:
        logger.error(f"NLP processing error: {t_err}", exc_info=True)
        st.error(f"❌ Text Preprocessing Error: {t_err}")
        return
    except Exception as exc:
        logger.error(f"Unexpected processing error: {exc}", exc_info=True)
        st.error(f"❌ An unexpected error occurred while processing the PDF: {exc}")
        return

    # Check if any text was extracted
    if stats["pages_with_text"] == 0:
        st.warning(
            "⚠️ No extractable text was found in this document. "
            "It may contain scanned images, drawings, or non-extractable text without an OCR layer."
        )

    st.markdown("---")

    # Layout: Status Checklist & Document Information
    col_status, col_info = st.columns([1, 2])

    with col_status:
        st.markdown("### 📋 Processing Status")
        st.markdown(
            """
            <div class="status-card">
                <div class="status-item"><span class="status-check">✓</span> PDF uploaded</div>
                <div class="status-item"><span class="status-check">✓</span> Text extracted</div>
                <div class="status-item"><span class="status-check">✓</span> Text cleaned</div>
                <div class="status-item"><span class="status-check">✓</span> Tokenization completed</div>
                <div class="status-item"><span class="status-check">✓</span> Stop-word removal completed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown("### 📊 Document Information")
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">File Name</div>
                    <div class="metric-val" style="font-size: 1.05rem; word-break: break-all;">{uploaded_file.name}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Pages</div>
                    <div class="metric-val">{stats["total_pages"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pages with Text</div>
                    <div class="metric-val">{stats["pages_with_text"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Extracted Chars</div>
                    <div class="metric-val">{stats["total_characters"]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Tokens</div>
                    <div class="metric-val">{stats["total_tokens"]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Filtered Tokens</div>
                    <div class="metric-val">{stats["total_filtered_tokens"]:,}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Expandable Page-wise Inspection Section
    with st.expander("🔍 View Extracted Text", expanded=True):
        st.caption("Inspect the structured output for each page, including original text, cleaned text, tokens, and stop-word filtered tokens.")

        if num_pages > 1:
            page_options = [f"Page {p['page']}" for p in processed_pages]
            selected_page_str = st.selectbox("Select Page to Inspect", options=page_options)
            page_num_selected = int(selected_page_str.replace("Page ", ""))
        else:
            page_num_selected = 1

        selected_page_data = next(
            (p for p in processed_pages if p["page"] == page_num_selected),
            processed_pages[0]
        )

        p_orig = selected_page_data["original_text"]
        p_cleaned = selected_page_data["cleaned_text"]
        p_tokens = selected_page_data["tokens"]
        p_filtered = selected_page_data["filtered_tokens"]

        if not p_cleaned.strip():
            st.info(f"ℹ️ Page {page_num_selected} does not contain any extractable text.")
        else:
            # Summary badge for current page
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
                    height=260,
                    key=f"orig_text_{page_num_selected}",
                    disabled=True,
                )

            with t2:
                st.text_area(
                    label="Cleaned Text (normalized whitespace & lines)",
                    value=p_cleaned,
                    height=260,
                    key=f"cleaned_text_{page_num_selected}",
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
