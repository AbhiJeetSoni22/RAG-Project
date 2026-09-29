# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and answer questions based on the document content.

---

## Current Status

### **Phase 2 — Text Chunking & Retrieval Data Preparation (ACTIVE)**

Phase 2 builds upon the document ingestion foundation of Phase 1. It converts the page-level processed text into meaningful, searchable semantic chunks while strictly preserving essential metadata (page numbers, source document filename, unique chunk identifiers, original text, and cleaned text) to prepare a retrieval-ready collection for Phase 3.

```text
PDF Document
     ↓
PDF Processor (PyMuPDF)
     ↓
Page-level Documents (Pages 1..N)
     ↓
Text Processor (Cleaning & Tokenization)
     ↓
Cleaned Page Text
     ↓
Chunker (Paragraph-aware with Overlap)
     ↓
Retrieval-Ready Chunk Collection
```

> **Roadmap Note:** Downstream phases will implement the retrieval engine (TF-IDF / vector embeddings / similarity search) and question answering.

---

## Features

### Phase 1: Ingestion & Text Preprocessing
* **PDF Upload & Validation:** Securely accepts `.pdf` documents via Streamlit, with validation for empty files, corrupted streams, and unreadable formats.
* **Page-wise Text Extraction:** Utilizes PyMuPDF (`fitz`) to extract text page-by-page while strictly preserving 1-indexed page numbers.
* **Non-destructive Text Cleaning:** Normalizes horizontal spaces, trims blank lines, cleans control/non-printable characters, and standardizes paragraph breaks without modifying original text.
* **NLTK Tokenization:** Segments cleaned text into words and punctuation tokens using NLTK `word_tokenize`.
* **Stop-word Removal:** Removes English stop words (via `nltk.corpus.stopwords`) while preserving original token casing.

### Phase 2: Text Chunking & Retrieval Data Preparation
* **Paragraph-Aware Hierarchy Chunking:** Chunks text using a natural boundary hierarchy:
  1. *Paragraph boundary* (primary semantic unit)
  2. *Sentence boundary* (via NLTK `sent_tokenize` if a paragraph exceeds `chunk_size`)
  3. *Word boundary* (if an individual sentence exceeds `chunk_size`)
  4. *Character boundary* (only as a fallback for giant strings without whitespace)
* **Context-Preserving Chunk Overlap:** Carries forward a configurable overlap window between consecutive chunks at clean word boundaries, ensuring critical contextual information spanning boundaries is never lost.
* **Dual Text Representation:**
  * `original_text`: Exact human-readable passage with casing and punctuation, used to display retrieved evidence to the user.
  * `cleaned_text`: Normalized, lowercase, punctuation-filtered representation (retaining interrogative tokens like *what, how, why*), prepared for downstream vectorization.
* **Strict Metadata Preservation:** Every chunk retains its `chunk_id` (`page_{n}_chunk_{m}`), `source` (filename), `page` (1-indexed), character count, and word count.
* **Chunk Validation:** Automated verification ensuring unique IDs, valid page numbers, non-empty content, and data integrity.
* **Chunk Statistics:** Real-time computation of total pages, pages with text, total chunks, average chunk size, minimum chunk size, maximum chunk size, and chunks-per-page distribution.
* **Interactive Streamlit UI:**
  * Sidebar parameter controls for `chunk_size` and `chunk_overlap`.
  * Live status checklist tracking all ingestion and chunking milestones.
  * Summary metric cards for document overview and chunk statistics.
  * Chunk preview cards displaying the first 5 generated chunks.
  * Paginated, expandable "View Chunks" section with page filtering and side-by-side tabs for Original vs Cleaned chunk text.
  * Expandable Phase 1 page-level inspection view for regression validation.

---

## Why Chunking is Required

When building retrieval-based question answering systems:
1. **Context Granularity:** Whole pages or entire documents contain multiple disparate topics. Searching at the document level dilutes relevance signals.
2. **Context Window Limitations:** Downstream models and similarity scorers operate most effectively on concise, focused passages.
3. **Pinpoint Evidence Attribution:** Generating compact chunks allows the QA system to tell the user exactly which passage and page contains the answer.

---

## Why Chunk Overlap is Used

If an important fact, definition, or relation appears across the boundary between two adjacent chunks:
* Without overlap, the relation is fractured: part of the premise is in Chunk 1 and the conclusion is in Chunk 2.
* With overlap, the tail of Chunk 1 is carried over into the head of Chunk 2, ensuring that queries matching either concept retrieve a complete, self-contained context.

---

## Why Page & Source Metadata is Preserved

Every generated chunk contains:
```python
{
    "chunk_id": "page_3_chunk_2",
    "source": "NLP_Syllabus.pdf",
    "page": 3,
    "original_text": "...",
    "cleaned_text": "...",
    "char_count": 421,
    "word_count": 65
}
```
Preserving the source document name and 1-indexed page number ensures downstream components can accurately cite sources (e.g., *"Relevant information found on Page 3 of NLP_Syllabus.pdf"*).

---

## How Chunks Will Be Used in Future Phases

* **Phase 3 (Retrieval Engine — Planned):**
  * `cleaned_text` will be vectorized using sparse techniques (e.g. TF-IDF / Bag of Words) or dense sentence embeddings.
  * User questions will be embedded into the same vector space.
  * Cosine similarity or vector search will rank the most relevant chunks.
* **Phase 4 (Question Answering — Planned):**
  * The top-ranked chunks' `original_text` will be presented to the user as direct evidence or provided as context for answer formulation.

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                  # Streamlit web application (Phases 1 & 2)
├── requirements.txt        # Core dependencies (streamlit, pymupdf, nltk)
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── __init__.py         # Package exports
│   ├── pdf_processor.py    # PyMuPDF page-by-page text extraction (Phase 1)
│   ├── text_processor.py   # Text cleaning, tokenization & stopwords (Phase 1)
│   └── chunker.py          # Semantic chunking, metadata & statistics (Phase 2)
│
├── data/
│   └── uploads/            # Temporary upload storage and sample documents
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py      # Automated test suite for Phase 1
    └── test_phase2.py      # Automated test suite for Phase 2
```

---

## Installation

### 1. Clone or navigate to the repository
```bash
cd retrieval-qa
```

### 2. Create and activate a Python virtual environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Once started, open `http://localhost:8501` in your browser, upload any PDF, configure chunk parameters in the sidebar, and inspect both document statistics and individual chunk cards.

---

## Running Automated Tests

To run the complete automated test suite across both Phase 1 and Phase 2:

```bash
python -m unittest discover tests
```

To run Phase 2 tests individually:

```bash
python -m unittest tests/test_phase2.py
```

To run Phase 1 tests individually:

```bash
python -m unittest tests/test_phase1.py
```
