# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and answer questions based on the document content.

---

## Current Phase

### **Phase 1 — Document Ingestion & Text Preprocessing**

Phase 1 establishes the text-ingestion foundation. It accepts a PDF uploaded via Streamlit, extracts its contents page-by-page, cleans the raw text, tokenizes it into words/punctuation, and performs stop-word removal using NLTK, yielding structured data for downstream retrieval.

> **Note on Roadmap:** Later phases will implement document chunking, vector indexing/retrieval (TF-IDF / embeddings), similarity search, and question answering.

---

## Features (Phase 1)

* **PDF Upload & Validation:** Securely accepts `.pdf` documents via Streamlit, with validation for empty files, corrupted streams, and unreadable formats.
* **Page-wise Text Extraction:** Utilizes PyMuPDF (`fitz`) to extract text page-by-page while strictly preserving 1-indexed page numbers.
* **Non-destructive Text Cleaning:** Normalizes unnecessary horizontal spaces, trims blank lines, cleans up control/non-printable characters, and standardizes paragraph breaks without destroying the original text.
* **NLTK Tokenization:** Segments cleaned text into words and punctuation tokens using NLTK `word_tokenize`.
* **Stop-word Removal:** Removes English stop words (via `nltk.corpus.stopwords`) while preserving token casing and offering structured access to both raw and filtered tokens.
* **Interactive Streamlit UI:**
  * Live status checklist (upload, extraction, cleaning, tokenization, stop-word removal).
  * Document summary metrics (total pages, pages with text, characters, tokens, filtered tokens).
  * Expandable page-by-page viewer with tabs for original text, cleaned text, tokens, and filtered tokens.

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Minimal Phase 1 dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── __init__.py         # Package initializer
│   ├── pdf_processor.py    # PyMuPDF-based page-by-page text extraction
│   └── text_processor.py   # Text cleaning, tokenization & stop-word removal
│
├── data/
│   └── uploads/            # Temporary upload storage (.gitkeep)
│
└── tests/
    ├── __init__.py
    └── test_phase1.py      # Automated tests for Phase 1 verification
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

## Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

Once started, open the local URL (usually `http://localhost:8501`) in your browser, upload any PDF, and inspect the preprocessed output.

---

## Running the Automated Tests

To run the Phase 1 test suite covering normal PDFs, multi-page PDFs, blank pages, corrupted/invalid files, and stop-word filtering:

```bash
python -m unittest tests/test_phase1.py
```
