# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and answer questions based on the document content.

---

## Current Status

### **Phase 4 — Question Answering (ACTIVE)**

Phase 4 implements the Question Answering layer on top of the Phase 3 retrieval engine. It analyzes the top retrieved document passages, scores candidate sentences according to lexical and semantic alignment with the user's question, extracts the most relevant answer span, and presents the answer with complete source attribution (document filename, page number, chunk identifier, retrieval similarity, and raw evidence passage).

```text
PDF Document
     ↓
PDF Processor (PyMuPDF)
     ↓
Preprocessing (Cleaning, Tokenization & Stopwords)
     ↓
Chunking (Paragraph-aware with Overlap)
     ↓
TF-IDF Retrieval (Cosine Similarity Ranking)
     ↓
Top-K Relevant Passages
     ↓
Extractive QA Engine (Sentence Scoring & Span Selection)
     ↓
Answer + Source Attribution + Ground Truth Evidence
```

> **Grounding Principle:** *Retrieve first $\rightarrow$ Answer strictly from retrieved evidence.* No external LLMs or generative models are used; answers are 100% deterministic and restricted to uploaded document content.

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
  4. *Character boundary* (only as a fallback for giant continuous strings)
* **Context-Preserving Chunk Overlap:** Carries forward a configurable overlap window between consecutive chunks at clean word boundaries, ensuring critical contextual information spanning boundaries is never lost.
* **Dual Text Representation:**
  * `original_text`: Exact human-readable passage with casing and punctuation, used to display retrieved evidence to the user.
  * `cleaned_text`: Normalized, lowercase, punctuation-filtered representation (retaining interrogative tokens like *what, how, why*), prepared for downstream vectorization.
* **Strict Metadata Preservation:** Every chunk retains its `chunk_id` (`page_{n}_chunk_{m}`), `source` (filename), `page` (1-indexed), character count, and word count.
* **Chunk Validation & Statistics:** Automated verification of uniqueness and data integrity, along with comprehensive chunk metrics.

### Phase 3: Retrieval Engine (TF-IDF + Cosine Similarity)
* **TF-IDF Vectorization:** Utilizes `scikit-learn`'s `TfidfVectorizer` (with unigram + bigram support and sublinear TF scaling) fitted once over the chunk collection corpus.
* **Symmetric Question Preprocessing:** Normalizes user queries through the exact same preprocessing pipeline used on chunks, ensuring lexical vocabulary alignment.
* **Cosine Similarity Scoring:** Measures the angular orientation between the query TF-IDF vector and all chunk vectors, producing a continuous similarity metric in $[0.0, 1.0]$.
* **Relevance Ranking & Thresholding:** Ranks retrieved chunks in descending order of similarity score and excludes passages below a configurable cutoff (default: `0.10`).

### Phase 4: Question Answering (Extractive, Retrieval-Grounded QA)
* **Extractive Answer Formulation:** Analyzes candidate sentences from the top retrieved passages and extracts the exact factual answer sentence(s) without generating synthetic text.
* **Multi-Factor Sentence Scoring:**
  * *Lexical Keyword Overlap Ratio:* Fraction of question content words covered by the sentence.
  * *Sentence-level TF-IDF Cosine Similarity:* Measures vector alignment between question and candidate sentence.
  * *Informative / Answer-Type Cues:* Rewards definition markers (*"is a"*, *"refers to"*, *"stands for"*) for definitional queries and numeric/percentage tokens (*75%*, *digits*) for quantitative queries.
  * *Retrieval Confidence Prior:* Integrates the parent chunk's retrieval similarity score.
* **Concise Multi-Sentence Answers:** Dynamically joins adjacent highly relevant sentences up to a configurable maximum (default: 2 sentences) when a question requires context, while strictly avoiding bloated paragraphs.
* **Strict Document Grounding:** If Phase 3 retrieval returns no chunks above the similarity threshold, the QA engine refuses to answer and reports that no sufficiently relevant information was found.
* **Complete Source Attribution:** Every answer displays the source PDF filename, page number, chunk identifier, retrieval similarity score, and the exact `original_text` evidence passage.
* **Interactive Streamlit QA UI:**
  * Prominent **Answer Card** with highlighted formatting.
  * Transparent **Source Attribution** pills.
  * Collapsible **Retrieved Evidence** showing the unedited passage.
  * Collapsible **Top Retrieved Passages** showing Phase 3 ranking candidates.

---

## Technical Details: Phase 4 QA Engine

### 1. How Answers Are Extracted
1. **Retrieve Candidate Chunks:** The top-ranked chunks from Phase 3 are passed to the QA engine.
2. **Segment into Sentences:** Each candidate chunk's `original_text` is partitioned into sentences using NLTK `sent_tokenize`.
3. **Score Each Sentence:** Each candidate sentence $s$ is evaluated against question $q$ with parent chunk retrieval score $C_{\text{sim}}$:
   $$\text{Score}(s) = 0.45 \times \text{OverlapRatio}(q, s) + 0.30 \times \text{TfidfSim}(q, s) + 0.15 \times C_{\text{sim}} + \text{Bonus}$$
4. **Select Answer Span:** The highest-scoring sentence is selected. If neighboring sentences in the same chunk also show strong relevance ($\ge 60\%$ of top score), they are combined up to `max_sentences`.
5. **Attach Metadata:** The answer is paired with the chunk's `source`, `page`, `chunk_id`, and `original_text` evidence.

### 2. Why Extractive QA (No LLM)
* **Zero Hallucinations:** Every word in the answer is a direct quote from the uploaded document.
* **Complete Auditability:** Every answer can be verified against the exact page and sentence in the source PDF.
* **Fast & Offline:** Runs locally in milliseconds without API keys, GPU acceleration, or cloud dependencies.

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                  # Streamlit web application (Phases 1, 2, 3 & 4)
├── requirements.txt        # Dependencies (streamlit, pymupdf, nltk, scikit-learn)
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── __init__.py         # Package exports
│   ├── pdf_processor.py    # PyMuPDF page-by-page text extraction (Phase 1)
│   ├── text_processor.py   # Text cleaning, tokenization & stopwords (Phase 1)
│   ├── chunker.py          # Semantic chunking, metadata & statistics (Phase 2)
│   ├── retriever.py        # TF-IDF & Cosine Similarity retrieval (Phase 3)
│   └── qa_engine.py        # Extractive Question Answering engine (Phase 4)
│
├── data/
│   └── uploads/            # Temporary upload storage and sample documents
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py      # Automated test suite for Phase 1
    ├── test_phase2.py      # Automated test suite for Phase 2
    ├── test_phase3.py      # Automated test suite for Phase 3
    └── test_phase4.py      # Automated test suite for Phase 4
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

Once started, open `http://localhost:8501` in your browser:
1. Upload any PDF document.
2. In the **"Question Answering"** section, enter a natural language question.
3. Inspect the extracted **Answer**, the **Source Attribution** (Page & Document), the **Retrieved Evidence**, and the candidate retrieval ranking.

---

## Running Automated Tests

Run the complete automated test suite across all 4 phases (37 tests):

```bash
python -m unittest discover tests
```

To run individual phase test suites:

```bash
python -m unittest tests/test_phase4.py
python -m unittest tests/test_phase3.py
python -m unittest tests/test_phase2.py
python -m unittest tests/test_phase1.py
```
