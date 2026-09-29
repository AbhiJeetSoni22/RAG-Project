# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and answer questions based on the document content.

---

## Current Status

### **Phase 3 — Retrieval Engine (ACTIVE)**

Phase 3 implements the first working retrieval engine. It vectorizes the preprocessed document chunks from Phase 2 using **TF-IDF (Term Frequency-Inverse Document Frequency)** and compares incoming user questions against the document chunks using **Cosine Similarity**, returning the Top-K most relevant passages ranked by lexical relevance.

```text
PDF Document
     ↓
PDF Processor (PyMuPDF)
     ↓
Page-level Documents (Pages 1..N)
     ↓
Text Processor (Cleaning & Tokenization)
     ↓
Chunker (Paragraph-aware with Overlap)
     ↓
TF-IDF Retriever (Scikit-Learn Vectorizer & Cosine Similarity)
     ↓
Ranked Relevant Chunks (Top-K)
```

> **Roadmap Note:** Downstream Phase 4 will implement question answering and answer generation using the retrieved passages as context.

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
* **Relevance Ranking:** Ranks retrieved chunks in descending order of similarity score.
* **Configurable Top-K Retrieval:** Limits returned results to the top $K$ passages (configurable from 1 to 10, default: 5).
* **Configurable Similarity Threshold:** Filters out passages falling below a minimum similarity cutoff (default: 0.10) to prevent irrelevant matches.
* **Grounded No-Match Handling:** Gracefully detects unrelated queries or out-of-vocabulary questions, informing the user that no sufficiently relevant passages exist instead of hallucinating relevance.
* **Strict Original Text Evidence Display:** Retrieved passages are strictly rendered using `original_text` so human-readable formatting, casing, and punctuation are preserved.
* **Interactive Streamlit UI:**
  * "Ask a Question" input box with search trigger.
  * Sidebar parameter sliders for Top-K and Minimum Similarity Threshold.
  * Informational breakdown explaining the TF-IDF + Cosine Similarity retrieval mechanism.
  * Styled result cards showing Rank, Similarity %, Chunk ID, Page, and the raw passage.

---

## Technical Details: Phase 3 Retrieval Engine

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF reflects how important a word is to a specific document chunk within the uploaded document collection:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
* **Term Frequency ($\text{TF}$):** Measures how frequently term $t$ appears in chunk $d$. Sublinear scaling ($1 + \log(\text{TF})$) is applied to prevent repetitive words from dominating.
* **Inverse Document Frequency ($\text{IDF}$):** Down-weights terms that appear ubiquitously across all chunks, elevating distinctive, content-rich keywords.

### 2. Why TF-IDF is Used
* **Lightweight & Interpretable:** Does not require heavy neural networks or external API calls, making it fast and fully reproducible for local lab environments.
* **Effective Lexical Matching:** Strongly rewards exact phrase and term matches between the user's question and relevant document passages.
* **Proper Baseline:** Serves as the standard information retrieval baseline before comparing against dense semantic embeddings in later phases.

### 3. Cosine Similarity
Cosine similarity evaluates the similarity between the question vector $\mathbf{q}$ and a chunk vector $\mathbf{d}$:
$$\text{similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$
Because both vectors are $L_2$-normalized by `TfidfVectorizer`, the cosine similarity is computed efficiently as the dot product:
$$\text{similarity}(\mathbf{q}, \mathbf{d}) \in [0.0, 1.0]$$
Higher similarity scores indicate stronger lexical alignment with the user's question.

### 4. Top-K Retrieval & Thresholding
* **Top-K:** Ensures the user is not overwhelmed with an excessive number of passages; returns only the most relevant $K$ results.
* **Minimum Similarity Threshold:** Any chunk with a similarity score below the threshold (default: 0.10) is discarded.

### 5. No-Match Handling
If an unrelated question (e.g. *"What is the capital of France?"*) is asked about a technical document, all chunk similarity scores will be zero or below the threshold. The engine reports:
> *"No sufficiently relevant information was found in this document."*
This prevents ungrounded answers in the upcoming QA phase.

---

## How Chunks Will Be Used in Future Phases

* **Phase 4 (Question Answering — Planned):**
  * The top-ranked chunks' `original_text` retrieved in Phase 3 will be provided as ground-truth context to an answer formulation component or language model to generate concise, accurate answers with exact page citations.

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                  # Streamlit web application (Phases 1, 2 & 3)
├── requirements.txt        # Core dependencies (streamlit, pymupdf, nltk, scikit-learn)
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── __init__.py         # Package exports
│   ├── pdf_processor.py    # PyMuPDF page-by-page text extraction (Phase 1)
│   ├── text_processor.py   # Text cleaning, tokenization & stopwords (Phase 1)
│   ├── chunker.py          # Semantic chunking, metadata & statistics (Phase 2)
│   └── retriever.py        # TF-IDF & Cosine Similarity retrieval engine (Phase 3)
│
├── data/
│   └── uploads/            # Temporary upload storage and sample documents
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py      # Automated test suite for Phase 1
    ├── test_phase2.py      # Automated test suite for Phase 2
    └── test_phase3.py      # Automated test suite for Phase 3
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
2. Inspect the preprocessed pages and chunk collection.
3. Enter questions in the **"Ask a Question"** section to retrieve ranked passages with similarity scores.

---

## Running Automated Tests

Run the complete automated test suite across all 3 phases:

```bash
python -m unittest discover tests
```

To run Phase 3 retrieval tests individually:

```bash
python -m unittest tests/test_phase3.py
```

To run Phase 2 chunking tests:

```bash
python -m unittest tests/test_phase2.py
```

To run Phase 1 preprocessing tests:

```bash
python -m unittest tests/test_phase1.py
```
