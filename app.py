"""Streamlit Application for Retrieval-Based Question Answering.

Production-grade Document Intelligence and Grounded Question Answering System:
- Document Ingestion & Text Preprocessing (PyMuPDF, text normalization, tokenization)
- Semantic Text Chunking & Retrievable Passage Preparation
- Dual Sparse & Dense Retrieval Engines (TF-IDF + all-MiniLM-L6-v2)
- Reciprocal Score Fusion Hybrid Retrieval with configurable alpha
- Strictly Grounded Extractive Question Answering
- Information Retrieval Evaluation & Empirical Benchmark Suite
"""

from __future__ import annotations

import io
import hashlib
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
    page_title="Document Intelligence | Grounded Question Answering",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# APPLICATION DESIGN SYSTEM (Light-Only Professional Theme)
# =============================================================================
LIGHT_THEME_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

    /* Global Reset & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 1. Canvas & Viewport Background */
    .stApp, [data-testid="stAppViewContainer"], .main, .main .block-container {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    .main .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.95) !important;
        color: #0F172A !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    [data-testid="stToolbar"] {
        color: #0F172A !important;
    }

    /* 2. Sidebar Navigation & Controls */
    section[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #475569 !important;
        font-size: 0.88rem !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #E2E8F0 !important;
        margin: 1.25rem 0 !important;
    }

    /* 3. Product Header & Branding */
    .brand-container {
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #EFF6FF !important;
        color: #2563EB !important;
        border: 1px solid #BFDBFE !important;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.65rem;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A !important;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-size: 1.02rem;
        color: #475569 !important;
        font-weight: 400;
        line-height: 1.55;
        max-width: 900px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 1.75rem !important;
        margin-bottom: 0.35rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.35rem !important;
    }

    p, .stMarkdown:not(button *) p, li {
        color: #334155 !important;
        line-height: 1.55;
    }

    [data-testid="stCaptionContainer"], caption, .stCaption {
        color: #64748B !important;
        font-size: 0.88rem !important;
    }

    hr {
        border-color: #E2E8F0 !important;
        margin: 1.75rem 0 !important;
    }

    /* 4. Active Document Banner */
    .active-doc-banner {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #2563EB !important;
        border-radius: 0.75rem;
        padding: 0.85rem 1.25rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .active-doc-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0F172A !important;
    }

    .active-doc-dot {
        width: 9px;
        height: 9px;
        background-color: #16A34A;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.2);
    }

    /* 5. Empty State Onboarding Card */
    .empty-hero-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 1rem;
        padding: 2.25rem 2rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .empty-hero-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }

    .empty-hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-bottom: 0.4rem;
    }

    .empty-hero-sub {
        font-size: 0.95rem;
        color: #475569 !important;
        max-width: 620px;
        margin: 0 auto 1.75rem auto;
        line-height: 1.5;
    }

    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        text-align: left;
        margin-top: 1.5rem;
    }

    .workflow-step {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem;
        padding: 1.15rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .workflow-step:hover {
        border-color: #BFDBFE !important;
        transform: translateY(-2px);
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: #2563EB;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.75rem;
        border-radius: 50%;
        margin-bottom: 0.6rem;
    }

    .step-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0F172A !important;
        margin-bottom: 0.25rem;
    }

    .step-desc {
        font-size: 0.82rem;
        color: #64748B !important;
        line-height: 1.45;
    }

    /* 6. File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem !important;
        transition: border-color 0.15s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2563EB !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
        border-radius: 0.5rem !important;
    }

    [data-testid="stFileUploader"] * {
        color: #0F172A !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 0.375rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem !important;
    }

    /* 7. Input Controls, Selects, and Textareas */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 0.85rem !important;
        font-size: 0.95rem !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #94A3B8 !important;
    }

    textarea:disabled, input:disabled {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        opacity: 1 !important;
        border: 1px solid #E2E8F0 !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stNumberInput label,
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stRadio label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
        border-radius: 0.5rem !important;
    }

    div[data-baseweb="select"] * {
        color: #0F172A !important;
    }

    div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
    }

    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
    }

    span[data-baseweb="tag"] {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 0.375rem !important;
    }

    span[data-baseweb="tag"] * {
        color: #1D4ED8 !important;
    }

    /* 8. Sliders & Radio Options */
    [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #2563EB !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stSlider"] * {
        color: #0F172A !important;
    }

    /* Radio Group Options - Clean Native Streamlit Radio Controls */
    [data-testid="stRadio"] label {
        color: #0F172A !important;
        cursor: pointer !important;
    }

    [data-testid="stRadio"] label p {
        color: #0F172A !important;
        font-size: 0.9rem !important;
    }

    /* 9. Buttons (Primary & Secondary) */
    button[kind="primary"],
    [data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"],
    div.st-key-btn_ask_question button,
    div.st-key-btn_run_retrieval_eval button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.15s ease !important;
    }

    button[kind="primary"] *,
    button[kind="primary"] p,
    button[kind="primary"] span,
    button[kind="primary"] div,
    [data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span,
    [data-testid="stBaseButton-primary"] div,
    .stButton > button[kind="primary"] *,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div,
    .stButton > button[kind="primary"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p,
    div.st-key-btn_ask_question button *,
    div.st-key-btn_ask_question button p,
    div.st-key-btn_run_retrieval_eval button *,
    div.st-key-btn_run_retrieval_eval button p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    button[kind="primary"]:hover,
    button[kind="primary"]:active,
    button[kind="primary"]:focus,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:active,
    [data-testid="stBaseButton-primary"]:focus,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:active,
    .stButton > button[kind="primary"]:focus,
    div.st-key-btn_ask_question button:hover,
    div.st-key-btn_run_retrieval_eval button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }

    button[kind="primary"]:hover *,
    button[kind="primary"]:active *,
    button[kind="primary"]:focus *,
    [data-testid="stBaseButton-primary"]:hover *,
    [data-testid="stBaseButton-primary"]:active *,
    [data-testid="stBaseButton-primary"]:focus *,
    .stButton > button[kind="primary"]:hover *,
    .stButton > button[kind="primary"]:active *,
    .stButton > button[kind="primary"]:focus *,
    div.st-key-btn_ask_question button:hover *,
    div.st-key-btn_run_retrieval_eval button:hover * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"],
    .stButton > button:not([kind="primary"]),
    [data-testid="stDownloadButton"] > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease !important;
    }

    button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover,
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
        transform: translateY(-1px) !important;
    }

    /* 10. Document Overview & KPI Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }

    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem;
        padding: 0.95rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.2;
    }

    /* 11. Pipeline Readiness Chips */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
    }

    .pipeline-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #0F172A !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    .pipeline-check {
        color: #16A34A !important;
        font-weight: bold;
    }

    /* 12. Answer & Evidence Presentation */
    .answer-card {
        background: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        border-left: 6px solid #16A34A !important;
        border-radius: 0.85rem;
        padding: 1.4rem 1.6rem;
        margin-top: 0.85rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(22, 163, 74, 0.08);
    }

    .answer-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #15803D !important;
        margin-bottom: 0.65rem;
    }

    .answer-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #14532D !important;
        line-height: 1.65;
    }

    .no-answer-card {
        background: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-left: 6px solid #DC2626 !important;
        border-radius: 0.85rem;
        padding: 1.35rem 1.6rem;
        margin-top: 0.85rem;
        margin-bottom: 1rem;
    }

    .no-answer-badge {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #B91C1C !important;
        margin-bottom: 0.4rem;
    }

    .no-answer-msg {
        font-size: 1.05rem;
        font-weight: 600;
        color: #991B1B !important;
        margin-bottom: 0.4rem;
    }

    .no-answer-reason {
        font-size: 0.88rem;
        color: #7F1D1D !important;
        background: rgba(220, 38, 38, 0.06);
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
    }

    .source-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 0.6rem 0.85rem;
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem;
        color: #334155 !important;
    }

    .meta-tag {
        background: #F8FAFC !important;
        color: #334155 !important;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.25rem 0.6rem;
        border-radius: 0.375rem;
        border: 1px solid #E2E8F0 !important;
    }

    .evidence-quote-box {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #2563EB !important;
        border-radius: 0.65rem;
        padding: 1.1rem 1.35rem;
        font-size: 0.98rem;
        line-height: 1.65;
        color: #1E293B !important;
        font-style: italic;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        white-space: pre-wrap;
    }

    /* 13. Top Retrieved Passages Cards */
    .result-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem;
        padding: 1.15rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .result-card:hover {
        border-color: #94A3B8 !important;
        box-shadow: 0 4px 8px -1px rgba(0, 0, 0, 0.07);
    }

    .result-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.75rem;
    }

    .rank-pill {
        background: #0F172A !important;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
    }

    .similarity-pill {
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
    }

    .similarity-high {
        background: #DCFCE7 !important;
        color: #15803D !important;
        border: 1px solid #BBF7D0 !important;
    }

    .similarity-med {
        background: #E0F2FE !important;
        color: #0369A1 !important;
        border: 1px solid #BAE6FD !important;
    }

    .similarity-low {
        background: #F1F5F9 !important;
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
    }

    .passage-box {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem;
        padding: 0.95rem 1.15rem;
        font-size: 0.93rem;
        line-height: 1.55;
        color: #1E293B !important;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }

    /* 14. Expanders */
    [data-testid="stExpander"],
    details[data-testid="stExpander"],
    .streamlit-expander,
    div[data-baseweb="accordion"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        overflow: hidden !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stExpander"] details,
    details[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-radius: 0.75rem !important;
        border: none !important;
    }

    [data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        border-radius: 0.75rem !important;
        padding: 0.8rem 1.15rem !important;
        border: none !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }

    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div,
    details[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary p,
    details[data-testid="stExpander"] summary span,
    details[data-testid="stExpander"] summary div,
    .streamlit-expanderHeader * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:hover,
    .streamlit-expanderHeader:hover {
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        color: #2563EB !important;
    }

    [data-testid="stExpander"] summary:hover *,
    [data-testid="stExpander"] summary:hover p,
    [data-testid="stExpander"] summary:hover span,
    details[data-testid="stExpander"] summary:hover *,
    details[data-testid="stExpander"] summary:hover p,
    details[data-testid="stExpander"] summary:hover span {
        color: #2563EB !important;
        -webkit-text-fill-color: #2563EB !important;
    }

    [data-testid="stExpander"] summary svg,
    details[data-testid="stExpander"] summary svg {
        fill: #475569 !important;
        color: #475569 !important;
    }

    [data-testid="stExpander"] summary:hover svg,
    details[data-testid="stExpander"] summary:hover svg {
        fill: #2563EB !important;
        color: #2563EB !important;
    }

    [data-testid="stExpanderDetails"],
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        padding: 1.25rem !important;
        border-top: 1px solid #F1F5F9 !important;
    }

    /* 15. Tabs */
    [data-testid="stTabs"] {
        background-color: transparent !important;
    }

    [data-testid="stTabs"] div[role="tablist"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-bottom: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem 0.5rem 0 0 !important;
        padding: 0 0.5rem !important;
        gap: 0.5rem !important;
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stTabs"] button[role="tab"] * {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
    }

    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #2563EB !important;
        background-color: #F8FAFC !important;
    }

    [data-testid="stTabs"] button[role="tab"]:hover * {
        color: #2563EB !important;
        -webkit-text-fill-color: #2563EB !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #2563EB !important;
        font-weight: 700 !important;
        background-color: transparent !important;
        border-bottom: 2px solid #2563EB !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
        color: #2563EB !important;
        -webkit-text-fill-color: #2563EB !important;
        font-weight: 700 !important;
    }

    [data-testid="stTabs"] div[role="tabpanel"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        padding-top: 1rem !important;
    }

    /* 16. Chunks & Chips */
    .chunk-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.75rem;
    }

    .chunk-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
    }

    .chunk-id-tag {
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-family: 'Fira Code', monospace;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.2rem 0.55rem;
        border-radius: 0.375rem;
        border: 1px solid #BFDBFE !important;
    }

    .chunk-meta-pill {
        background: #F1F5F9 !important;
        color: #475569 !important;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.2rem 0.5rem;
        border-radius: 0.375rem;
    }

    .preview-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem;
        padding: 0.65rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .preview-card-title {
        font-family: 'Fira Code', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #2563EB !important;
    }

    .preview-card-meta {
        font-size: 0.72rem;
        color: #64748B !important;
        margin-top: 0.2rem;
    }

    .preview-card-text {
        font-size: 0.75rem;
        color: #334155 !important;
        margin-top: 0.35rem;
        line-height: 1.25;
        max-height: 55px;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .token-chip-container {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem;
        padding: 0.6rem;
        max-height: 220px;
        overflow-y: auto;
    }

    .token-chip {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        display: inline-block;
        padding: 0.15rem 0.45rem;
        border-radius: 0.25rem;
        font-size: 0.78rem;
        font-family: 'Fira Code', monospace;
        margin: 0.15rem;
    }

    .token-chip.filtered {
        background: #EFF6FF !important;
        color: #2563EB !important;
        border-color: #BFDBFE !important;
    }

    /* 17. Button Alignment Helper */
    .btn-align-wrapper {
        height: 28px;
    }

    @media (max-width: 768px) {
        .btn-align-wrapper {
            height: 0px;
            display: none;
        }
    }
"""

# Inject custom light theme CSS immediately upon page initialization
st.markdown(f"<style>{LIGHT_THEME_CSS}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render page title, badge, and concise subtitle."""
    st.markdown(
        """
        <div class="brand-container">
            <span class="brand-badge">⚡ Enterprise Document Intelligence &bull; Extractive QA Engine</span>
            <div class="main-title">Document Intelligence</div>
            <div class="sub-title">
                Upload any PDF document to ask natural language questions with strict document-grounded extractive answering
                and objectively benchmark retrieval performance across lexical (TF-IDF), dense semantic (MiniLM), and hybrid retrieval engines.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """Render an inviting, professional onboarding interface when no document is uploaded."""
    st.markdown(
        """
        <div class="empty-hero-card">
            <div class="empty-hero-icon">📄</div>
            <div class="empty-hero-title">Ready for Document Question Answering</div>
            <div class="empty-hero-sub">
                Upload a PDF document above to extract text, build dual sparse-dense retrieval indices,
                and ask natural language questions with 100% grounded citations.
            </div>
            <div class="workflow-grid">
                <div class="workflow-step">
                    <span class="step-number">1</span>
                    <div class="step-title">Upload Document</div>
                    <div class="step-desc">Extract text via PyMuPDF, normalize whitespace, tokenize words, and generate overlapping semantic chunks.</div>
                </div>
                <div class="workflow-step">
                    <span class="step-number">2</span>
                    <div class="step-title">Choose Retrieval Engine</div>
                    <div class="step-desc">Select TF-IDF (keyword lexical), SentenceTransformers (dense contextual vector), or Hybrid reciprocal score fusion.</div>
                </div>
                <div class="workflow-step">
                    <span class="step-number">3</span>
                    <div class="step-title">Grounded Answer & Citation</div>
                    <div class="step-desc">Ask questions to extract precise answer spans directly from source passages with zero generative hallucination.</div>
                </div>
                <div class="workflow-step">
                    <span class="step-number">4</span>
                    <div class="step-title">Empirical Benchmarks</div>
                    <div class="step-desc">Objectively measure Hit@K, MRR, Precision@K, and Recall@K against a verified ground-truth QA dataset.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, int, str, float, int, float, int]:
    """Render sidebar controls for configurable chunking, retrieval, and QA parameters."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Configuration")

        st.markdown("#### 📄 Document Chunking")
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
        st.markdown("#### 🔎 Retrieval Engine")
        retrieval_method = st.radio(
            "Default Retrieval Method",
            options=["Hybrid", "Semantic Embeddings", "TF-IDF"],
            index=0,
            help="Select default retrieval strategy: Hybrid (fused lexical + dense semantic), Semantic (SentenceTransformers all-MiniLM-L6-v2), or Lexical TF-IDF.",
            key="cfg_retrieval_method",
        )

        hybrid_alpha = 0.50
        if retrieval_method == "Hybrid":
            hybrid_alpha = st.slider(
                "Hybrid Weight (α: Lexical vs 1-α: Semantic)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Weight α for lexical TF-IDF score vs (1 - α) for Semantic dense embedding score. α=1.0 is purely lexical; α=0.0 is purely semantic.",
                key="cfg_hybrid_alpha",
            )
            st.caption(f"**Current Fusion:** {hybrid_alpha * 100:.0f}% TF-IDF Lexical + {(1 - hybrid_alpha) * 100:.0f}% Semantic Embeddings")

        top_k = st.slider(
            "Top-K Retrieved Passages",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Maximum number of relevant candidate passages retrieved for answer extraction.",
            key="cfg_top_k",
        )

        min_similarity = st.slider(
            "Minimum Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            help="Passages with cosine similarity below this cutoff will be excluded as non-relevant.",
            key="cfg_min_similarity",
        )

        st.markdown("---")
        st.markdown("#### 🎯 Answer Extraction")
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
        st.markdown("#### 🛡️ Grounding Principles")
        st.markdown(
            """
            * **Extractive NLP:** Answers are extracted verbatim from retrieved passages.
            * **Zero Hallucination:** Deterministic sentence scoring prevents generative fabrication.
            * **Evidence First:** The system refuses to answer if no supporting evidence exists.
            * **Auditable Citations:** Every response links to its exact page number and chunk ID.
            """
        )

    return chunk_size, chunk_overlap, retrieval_method, hybrid_alpha, top_k, min_similarity, max_sentences


def render_extracted_pages_inspection(processed_pages: list[dict[str, Any]]) -> None:
    """Render the page-level inspection expander for extracted document pages.

    Safely handles empty pages, single-page documents, and multi-page documents
    without relying on uninitialized outer-scope variables.
    """
    with st.expander("🔍 View Extracted Pages (Page-Level Inspection)", expanded=False):
        st.caption("Inspect the page-by-page extracted text, cleaned text, tokens, and stop-word filtered tokens.")

        if not processed_pages:
            st.info("ℹ️ No extracted pages are available for inspection.")
            return

        num_pages = len(processed_pages)
        if num_pages > 1:
            page_options = [f"Page {p['page']}" for p in processed_pages]
            selected_page_str = st.selectbox("Select Page to Inspect", options=page_options, key="raw_page_select")
            page_num_selected = int(selected_page_str.replace("Page ", ""))
        else:
            page_num_selected = processed_pages[0]["page"]

        selected_page_data = next(
            (p for p in processed_pages if p["page"] == page_num_selected),
            processed_pages[0],
        )

        p_orig = selected_page_data.get("original_text", "")
        p_cleaned = selected_page_data.get("cleaned_text", "")
        p_tokens = selected_page_data.get("tokens", [])
        p_filtered = selected_page_data.get("filtered_tokens", [])

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


def main() -> None:
    """Main application loop."""
    chunk_size, chunk_overlap, retrieval_method, hybrid_alpha, top_k, min_similarity, max_sentences = render_sidebar()
    render_header()

    # Document Upload Section
    st.markdown("### 📤 Document Ingestion")
    st.caption("Upload any standard PDF document (academic paper, report, manual, policy) to build retrieval indices and begin querying.")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF file to extract, clean, tokenize, chunk, and perform Question Answering.",
    )

    if uploaded_file is None:
        # Clear document session state when no document is active
        for key in ["active_doc_hash", "cached_pipeline_key", "cached_pipeline_data", "phase6_evaluation_result", "current_qa_state"]:
            st.session_state.pop(key, None)
        render_empty_state()
        return

    # Check for empty file upload (0 bytes)
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("⚠️ The uploaded file is empty (0 bytes). Please upload a valid PDF document.")
        return

    # Document Identity Tracking via stable SHA-256 content hash
    doc_hash = hashlib.sha256(file_bytes).hexdigest()

    # When active document changes, invalidate document-specific cache, QA state, and evaluation results
    if st.session_state.get("active_doc_hash") != doc_hash:
        st.session_state["active_doc_hash"] = doc_hash
        st.session_state.pop("phase6_evaluation_result", None)
        st.session_state.pop("current_qa_state", None)
        st.session_state.pop("cached_pipeline_key", None)
        st.session_state.pop("cached_pipeline_data", None)

    pipeline_key = (doc_hash, chunk_size, chunk_overlap)
    cached_pipe = st.session_state.get("cached_pipeline_data")

    if st.session_state.get("cached_pipeline_key") == pipeline_key and cached_pipe is not None:
        processed_pages = cached_pipe["processed_pages"]
        doc_stats = cached_pipe["doc_stats"]
        chunks = cached_pipe["chunks"]
        chunk_stats = cached_pipe["chunk_stats"]
        tfidf_retriever = cached_pipe["tfidf_retriever"]
        semantic_retriever = cached_pipe["semantic_retriever"]
    else:
        # Step 1: Ingestion & Text Processing
        try:
            with st.spinner("Parsing PDF and preprocessing text page-by-page..."):
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

        # Check if any extractable text was found
        if doc_stats["pages_with_text"] == 0:
            st.warning(
                "⚠️ No extractable text was found in this document. "
                "It may contain scanned images, drawings, or non-extractable text without an OCR layer."
            )
            return

        # Step 2: Document Chunking
        try:
            with st.spinner("Partitioning text into retrievable passages with context overlap..."):
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

        # Step 3: Index Document Chunks for TF-IDF, Semantic, and Hybrid Retrieval
        try:
            with st.spinner("Building lexical TF-IDF index and computing dense neural embeddings..."):
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

                # Cache pipeline data for instantaneous reruns
                st.session_state["cached_pipeline_key"] = pipeline_key
                st.session_state["cached_pipeline_data"] = {
                    "processed_pages": processed_pages,
                    "doc_stats": doc_stats,
                    "chunks": chunks,
                    "chunk_stats": chunk_stats,
                    "tfidf_retriever": tfidf_retriever,
                    "semantic_retriever": semantic_retriever,
                }
        except (RetrievalError, SemanticRetrievalError, HybridRetrievalError) as r_err:
            logger.error(f"Retriever initialization error: {r_err}", exc_info=True)
            st.error(f"❌ Retrieval Engine Error: {r_err}")
            return
        except Exception as exc:
            logger.error(f"Unexpected retriever error: {exc}", exc_info=True)
            st.error(f"❌ Failed to build retrieval engines: {exc}")
            return

    # Authoritative page count derived safely from processed document pages
    num_pages = len(processed_pages) if processed_pages else 0

    # Initialize Hybrid Retriever reusing existing instances with active hybrid_alpha
    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        alpha=hybrid_alpha,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )

    # Initialize Extractive QA Engine
    qa_engine = ExtractiveQAEngine()

    # Active Document Banner
    st.markdown(
        f"""
        <div class="active-doc-banner">
            <div class="active-doc-title">
                <span class="active-doc-dot"></span>
                <span><b>Active Document:</b> {uploaded_file.name}</span>
            </div>
            <div>
                <span class="meta-tag"><b>Format:</b> PDF ({len(file_bytes) / 1024:.1f} KB)</span>
                <span class="meta-tag"><b>Pages:</b> {doc_stats['total_pages']}</span>
                <span class="meta-tag"><b>Status:</b> Indexed & Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Overview & Pipeline Readiness
    col_kpi, col_pipe = st.columns([3, 2])
    with col_kpi:
        st.markdown("### 📊 Document Overview")
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Total Pages</div>
                    <div class="metric-val">{doc_stats["total_pages"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pages With Text</div>
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
                    <div class="metric-val" style="color: #2563EB;">{chunk_stats["total_chunks"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Chunk Length</div>
                    <div class="metric-val">{chunk_stats["avg_chunk_size"]} <span style="font-size: 0.8rem; font-weight: 500; color: #64748B;">chars</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pipe:
        st.markdown("### ⚡ Pipeline Readiness")
        st.markdown(
            f"""
            <div class="pipeline-container">
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> PDF Ingestion & Normalization</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> Tokenization & Stop-words Removed</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> Chunk Partitioning ({len(chunks)} Chunks)</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> TF-IDF Lexical Vectorizer Indexed</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> Dense Embeddings Indexed (all-MiniLM-L6-v2)</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> Hybrid Fusion Engine Active (α={hybrid_alpha:.2f})</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> Extractive Grounded QA Engine Ready</div>
                <div class="pipeline-chip"><span class="pipeline-check">✓</span> IR Evaluation Framework Configured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # QUESTION ANSWERING SECTION
    # =========================================================================
    st.markdown("## 💬 Document Question Answering")
    st.caption("Ask questions about the uploaded document. The system retrieves relevant passages and extracts factual answers strictly grounded in document evidence.")

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
        st.markdown('<div class="btn-align-wrapper"></div>', unsafe_allow_html=True)
        search_clicked = st.button(
            "💬 Ask Question",
            type="primary",
            use_container_width=True,
            key="btn_ask_question",
        )

    if selected_method == "Hybrid":
        st.caption(f"⚡ **Hybrid Mode Active:** Balancing lexical TF-IDF (α={hybrid_alpha:.2f}) and semantic embeddings (1-α={1.0 - hybrid_alpha:.2f}). Adjust α in the sidebar.")
    elif selected_method == "Semantic Embeddings":
        st.caption("🧠 **Dense Semantic Mode Active:** Using SentenceTransformers (`all-MiniLM-L6-v2`) 384-dimensional contextual embeddings.")
    else:
        st.caption("🔤 **Lexical TF-IDF Mode Active:** Using sparse n-gram vector matching with sublinear term frequency scaling.")

    # Retrieval & QA Pipeline Execution
    if search_clicked:
        cleaned_q = query_input.strip() if query_input else ""

        # Handle empty / whitespace / punctuation-only question
        if not cleaned_q or not any(c.isalnum() for c in cleaned_q):
            st.warning("⚠️ Please enter a question containing valid words or alphanumeric keywords.")
            st.session_state.pop("current_qa_state", None)
        else:
            with st.spinner(f"Retrieving candidate passages via {selected_method} and extracting grounded answer..."):
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

                # Extract answer from retrieved passages
                qa_result = qa_engine.answer(
                    question=cleaned_q,
                    retrieved_chunks=retrieved_results,
                    max_sentences=max_sentences,
                )

                st.session_state["current_qa_state"] = {
                    "question": cleaned_q,
                    "qa_result": qa_result,
                    "selected_method": selected_method,
                    "retrieved_results": retrieved_results,
                }

    # Render Current QA Result from Session State
    if "current_qa_state" in st.session_state:
        qa_state = st.session_state["current_qa_state"]
        qa_result = qa_state["qa_result"]
        qa_method = qa_state["selected_method"]
        retrieved_results = qa_state["retrieved_results"]

        # Answer Presentation
        if qa_result.get("found") and qa_result.get("answer"):
            salience = float(qa_result.get("sentence_score", 0.0))
            if salience >= 0.25:
                badge_html = '<span style="font-weight: bold; font-size: 0.95rem;">✓</span> GROUNDED FACTUAL ANSWER &bull; HIGH SALIENCE'
                badge_style = ""
            else:
                badge_html = '<span style="font-weight: bold; font-size: 0.95rem;">ℹ️</span> CANDIDATE EXTRACTED PASSAGE &bull; MODERATE SALIENCE'
                badge_style = 'style="background: #FEF3C7 !important; color: #92400E !important; border: 1px solid #FDE68A !important;"'

            st.markdown("### 💡 Grounded Answer")
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-badge" {badge_style}>
                        {badge_html}
                    </div>
                    <div class="answer-text">{qa_result['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Source Attribution Section
            score_label = "Hybrid Score" if qa_method == "Hybrid" else "Similarity Score"
            st.markdown(
                f"""
                <div class="source-container">
                    <span style="font-weight: 700; font-size: 0.8rem; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">Citations:</span>
                    <span class="meta-tag"><b>Retriever:</b> {qa_method}</span>
                    <span class="meta-tag"><b>Source:</b> {qa_result['source']}</span>
                    <span class="meta-tag"><b>Page:</b> {qa_result['page']}</span>
                    <span class="meta-tag"><b>Chunk:</b> <code>{qa_result['chunk_id']}</code></span>
                    <span class="meta-tag"><b>{score_label}:</b> {qa_result['similarity_score'] * 100:.2f}% ({qa_result['similarity_score']:.4f})</span>
                    <span class="meta-tag"><b>Sentence Salience:</b> {qa_result['sentence_score']:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Evidence Section
            with st.expander("📜 Retrieved Evidence (Ground Truth Passage)", expanded=True):
                st.caption("The exact original passage from which the answer was extracted:")
                st.markdown(
                    f"""
                    <div class="evidence-quote-box">
                        "{qa_result['evidence']}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # Explicit Abstention UI (Zero-Hallucination Safe State)
            st.markdown("### 💡 Answer Status")
            st.markdown(
                f"""
                <div class="no-answer-card">
                    <div class="no-answer-badge">⚠️ NO GROUNDED ANSWER FOUND &bull; ABSTAINED</div>
                    <div class="no-answer-msg">I couldn't find a sufficiently relevant answer in this document.</div>
                    <div class="no-answer-reason">
                        <b>Strict Grounding Reason:</b> {qa_result.get('reason', 'Evidence insufficient')}
                    </div>
                    <div style="font-size: 0.85rem; color: #7F1D1D; margin-top: 0.5rem;">
                        💡 <i>Tip: The extractive QA engine enforces zero hallucinations. Check whether the requested information is present in the document.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top Retrieved Passages Section
            with st.expander(f"🔍 Top Retrieved Passages ({selected_method})", expanded=False):
                st.caption(
                    f"Inspect all candidate passages retrieved by {selected_method} that were evaluated by the QA engine."
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

            # Diagnostic Inspection (collapsible, keeps UI clean for normal users)
            if "diagnostics" in qa_result and qa_result["diagnostics"]:
                with st.expander("🛠️ Diagnostics & Grounding Signals (Developer)", expanded=False):
                    st.caption("Internal telemetry, candidate features, and abstention reasons used for answer verification:")
                    st.json(qa_result["diagnostics"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # =========================================================================
    # RETRIEVAL EVALUATION & BENCHMARKING SECTION
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
                    st.success("✓ Retrieval evaluation completed successfully!")
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

            # Metric Definition Tooltips
            st.markdown(
                """
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">
                    <span class="meta-tag" title="Whether at least one ground-truth passage was ranked within the top K.">🎯 <b>Hit@K:</b> Top-K Accuracy</span>
                    <span class="meta-tag" title="Mean Reciprocal Rank evaluates how early the first relevant passage appears.">🥇 <b>MRR:</b> Mean Reciprocal Rank (1/rank)</span>
                    <span class="meta-tag" title="Fraction of top-K retrieved passages that are ground-truth relevant.">⚖️ <b>Precision@K:</b> Exactness</span>
                    <span class="meta-tag" title="Fraction of all relevant passages retrieved in the top-K.">📡 <b>Recall@K:</b> Completeness</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # High-level metric highlights across methods
            hybrid_summary = eval_res.summaries.get("Hybrid")
            tfidf_summary = eval_res.summaries.get("TF-IDF")
            semantic_summary = eval_res.summaries.get("Semantic")

            eval_max_k = max(eval_res.k_values)
            m_cols = st.columns(3)

            if hybrid_summary:
                with m_cols[0]:
                    st.metric(
                        label=f"Hybrid Hit@{eval_max_k} / MRR",
                        value=f"{hybrid_summary.hit_rates.get(eval_max_k, 0.0):.4f}",
                        delta=f"MRR: {hybrid_summary.mrr:.4f}",
                        delta_color="normal",
                    )
            if semantic_summary:
                with m_cols[1]:
                    st.metric(
                        label=f"Semantic Hit@{eval_max_k} / MRR",
                        value=f"{semantic_summary.hit_rates.get(eval_max_k, 0.0):.4f}",
                        delta=f"MRR: {semantic_summary.mrr:.4f}",
                        delta_color="normal",
                    )
            if tfidf_summary:
                with m_cols[2]:
                    st.metric(
                        label=f"TF-IDF Hit@{eval_max_k} / MRR",
                        value=f"{tfidf_summary.hit_rates.get(eval_max_k, 0.0):.4f}",
                        delta=f"MRR: {tfidf_summary.mrr:.4f}",
                        delta_color="normal",
                    )

            df_metrics = pd.DataFrame(eval_res.overall_metrics_table)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

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
    # DOCUMENT CHUNKS & PREPROCESSING INSPECTION (Maintained for full inspection)
    # =========================================================================
    st.markdown("## 🧩 Document Chunks & Inspection")
    st.caption(
        f"Verified text chunking: {len(chunks)} chunks created from {doc_stats['pages_with_text']} pages "
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
                <div class="metric-val">{chunk_stats["avg_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748B;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Min Chunk Size</div>
                <div class="metric-val">{chunk_stats["min_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748B;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Chunk Size</div>
                <div class="metric-val">{chunk_stats["max_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748B;">chars</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pages with Chunks</div>
                <div class="metric-val">{chunk_stats["pages_with_chunks"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Cleaned Size</div>
                <div class="metric-val">{chunk_stats["avg_cleaned_chunk_size"]} <span style="font-size: 0.8rem; font-weight: normal; color: #64748B;">chars</span></div>
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
                    <div class="preview-card">
                        <div class="preview-card-title">{p_i + 1}. {chk['chunk_id']}</div>
                        <div class="preview-card-meta">Page {chk['page']} &bull; {chk['char_count']} chars</div>
                        <div class="preview-card-text">
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
            "and cleaned text (for retrieval and vectorization) are preserved."
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

    # Page-wise Raw Inspection
    render_extracted_pages_inspection(processed_pages)


if __name__ == "__main__":
    main()
