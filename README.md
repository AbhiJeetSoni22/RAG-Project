# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and extract factual answers strictly based on document content.

---

## Current Status

### **Phase 5 — Semantic Retrieval Enhancement (ACTIVE)**

Phase 5 enhances the retrieval engine by introducing **dense semantic embeddings** via **Sentence Transformers (`all-MiniLM-L6-v2`)** alongside the existing **TF-IDF** retrieval engine from Phase 3. Users can dynamically switch between lexical TF-IDF retrieval and dense semantic retrieval. The selected retrieval engine ranks candidate passages, which are then passed to the deterministic extractive QA engine from Phase 4 to extract grounded factual answers with complete source citations.

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Preprocessing (Cleaning, Tokenization, Stopwords)
    ↓
Chunking (Paragraph-aware with Overlap)
    ↓
Retrieval Selection
 ┌───────────────────────┐
 │                       │
TF-IDF               Semantic
Retrieval            Retrieval
(Lexical Overlap)    (Sentence Transformers)
 │                       │
 └───────────┬───────────┘
             ↓
       Top-K Chunks
             ↓
     Existing QA Engine
(Deterministic Sentence Scoring)
             ↓
      Extractive Answer
             ↓
Source + Page + Ground Truth Evidence
```

> **Grounding Principle:** *Retrieve first $\rightarrow$ Answer strictly from retrieved evidence.* No external LLMs or generative models are used; answers are 100% deterministic and restricted to uploaded document content.

---

## Phase 5 — Semantic Retrieval Enhancement

### 1. Why TF-IDF Has Lexical Limitations
In Phase 3, retrieval relies on Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity. While effective for keyword matching, TF-IDF operates purely on **lexical overlap** (exact surface forms or n-grams).

Lexical retrieval suffers from two fundamental linguistic limitations:
* **Synonymy / Paraphrasing:** A question using different vocabulary (e.g., *"What percentage of attendance must students maintain?"*) will fail to match a passage phrased as *"Students are required to maintain a minimum attendance of 75 percent"* if key terms like *percentage* vs *percent* or *mandatory* vs *required* do not overlap in the TF-IDF vocabulary.
* **Semantic Context:** Words with identical forms but different meanings (polysemy) cannot be distinguished based purely on term counts.

### 2. Dense Sentence Embeddings & Sentence Transformers
Dense sentence embeddings project textual passages and questions into a shared continuous vector space $\mathbb{R}^d$ (where $d = 384$). Unlike sparse high-dimensional TF-IDF vectors where each dimension represents a specific n-gram word token, dimensions in dense embeddings represent latent semantic and syntactic features learned from massive contrastive text pairs.

In this project, we employ **Sentence Transformers** using the lightweight and efficient model:
* **Model:** `all-MiniLM-L6-v2`
* **Embedding Dimension:** 384
* **Parameters:** ~22 Million (compact, fast CPU inference)
* **Optimization:** Fine-tuned on over 1 billion sentence pairs for semantic similarity and symmetric information retrieval tasks.

### 3. Cosine Similarity & Unit Normalization
To measure semantic similarity between a query vector $\mathbf{q} \in \mathbb{R}^d$ and a chunk vector $\mathbf{c} \in \mathbb{R}^d$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|_2 \|\mathbf{c}\|_2}$$

During initialization, chunk embeddings are unit-normalized ($L_2$ norm $= 1$):
$$\|\mathbf{c}\|_2 = 1, \quad \|\mathbf{q}\|_2 = 1 \implies \text{Cosine Similarity}(\mathbf{q}, \mathbf{c}) = \mathbf{q} \cdot \mathbf{c}$$

This allows similarity scoring across all document passages to be computed via a fast matrix-vector dot product in memory.

### 4. Efficient Caching Strategy
Embedding generation is computationally more intensive than TF-IDF vectorization. To ensure responsiveness in the Streamlit application:
1. **Model Caching (`@st.cache_resource`):** The `SentenceTransformer("all-MiniLM-L6-v2")` model is loaded once into memory upon startup and reused across all sessions and reruns.
2. **Chunk Embeddings Caching (`@st.cache_data`):** Document chunk embeddings are computed once when a PDF is uploaded (or chunking settings change) and cached. Subsequent user questions encode **only the question vector**, running in milliseconds.

### 5. Seamless Integration with Phase 4 Extractive QA
Semantic retrieval changes **which passages are retrieved**, not how answers are generated. The candidate passages identified by the semantic retriever preserve all Phase 2 metadata (`chunk_id`, `source`, `page`, `original_text`, `cleaned_text`, `similarity_score`). These candidate passages are passed directly into the existing `ExtractiveQAEngine`, which scores sentences deterministically and extracts the exact factual answer span without hallucinations or synthetic text generation.

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
  4. *Character boundary* (fallback for giant continuous strings)
* **Context-Preserving Chunk Overlap:** Carries forward a configurable overlap window between consecutive chunks at clean word boundaries.
* **Dual Text Representation:**
  * `original_text`: Exact human-readable passage with casing and punctuation, used to display retrieved evidence.
  * `cleaned_text`: Normalized, lowercase, punctuation-filtered representation prepared for downstream vectorization.
* **Strict Metadata Preservation:** Every chunk retains its `chunk_id` (`page_{n}_chunk_{m}`), `source` (filename), `page` (1-indexed), character count, and word count.
* **Chunk Validation & Statistics:** Automated verification of uniqueness and data integrity, along with comprehensive chunk metrics.

### Phase 3: Lexical Retrieval Engine (TF-IDF + Cosine Similarity)
* **TF-IDF Vectorization:** Utilizes `scikit-learn`'s `TfidfVectorizer` (with unigram + bigram support and sublinear TF scaling) fitted once over the chunk collection corpus.
* **Symmetric Question Preprocessing:** Normalizes user queries through the exact same preprocessing pipeline used on chunks.
* **Cosine Similarity Scoring:** Measures the angular orientation between the query TF-IDF vector and all chunk vectors in $[0.0, 1.0]$.
* **Relevance Ranking & Thresholding:** Ranks retrieved chunks in descending order of similarity score and excludes passages below a configurable cutoff (default: `0.10`).

### Phase 4: Question Answering (Extractive, Retrieval-Grounded QA)
* **Extractive Answer Formulation:** Analyzes candidate sentences from the top retrieved passages and extracts the exact factual answer sentence(s) without generating synthetic text.
* **Multi-Factor Sentence Scoring:**
  * *Lexical Keyword Overlap Ratio:* Fraction of question content words covered by the sentence.
  * *Sentence-level TF-IDF Cosine Similarity:* Measures vector alignment between question and candidate sentence.
  * *Informative / Answer-Type Cues:* Rewards definition markers (*"is a"*, *"refers to"*, *"stands for"*) for definitional queries and numeric/percentage tokens (*75%*, *digits*) for quantitative queries.
  * *Retrieval Confidence Prior:* Integrates the parent chunk's retrieval similarity score.
* **Concise Multi-Sentence Answers:** Dynamically joins adjacent highly relevant sentences up to a configurable maximum (default: 2 sentences) when a question requires context.
* **Strict Document Grounding:** If retrieval returns no chunks above the similarity threshold, the QA engine refuses to answer and reports that no sufficiently relevant information was found.
* **Complete Source Attribution:** Every answer displays the source PDF filename, page number, chunk identifier, retrieval similarity score, and the exact `original_text` evidence passage.

### Phase 5: Semantic Retrieval Enhancement
* **Dense Embedding Retriever (`SemanticRetriever`):** Generates 384-dimensional normalized dense embeddings for document chunks using Sentence Transformers (`all-MiniLM-L6-v2`).
* **Semantic Paraphrase Matching:** Successfully pairs user queries with relevant passages even when wording, synonyms, or sentence structures differ.
* **Dual Retrieval Mode Selector:** Toggle seamlessly between `Semantic Embeddings` (default) and `TF-IDF` via the UI without reloading documents.
* **High-Performance In-Memory Search:** Vectorized dot-product cosine similarity computation without requiring external vector databases or heavyweight infrastructure.
* **Metadata & Attribution Transparency:** Transparent display of semantic similarity scores, rank positions, page citations, and chunk identifiers.

---

## Technical Comparison: TF-IDF vs. Semantic Embeddings

| Feature | TF-IDF Retrieval (Phase 3) | Semantic Embeddings (Phase 5) |
| :--- | :--- | :--- |
| **Representation** | High-dimensional, sparse lexical vectors | Low-dimensional (384-d), dense semantic vectors |
| **Matching Basis** | Exact word & n-gram overlap | Learned contextual semantic similarity |
| **Synonym Handling** | Limited (misses distinct terms with similar meaning) | Excellent (maps semantically related phrases closely) |
| **Paraphrase Resilience** | Vulnerable to surface rephrasing | High resilience to varying syntactic patterns |
| **Computation Cost** | Extremely low CPU overhead | Moderate CPU overhead during chunk indexing (cached) |
| **Model Footprint** | Zero external neural weights | Lightweight model weights (~80MB, cached locally) |
| **Use Case** | Exact keyword lookup, specific codes / identifiers | Conceptual questions, natural variations, semantic QA |

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                     # Streamlit web application (Phases 1, 2, 3, 4 & 5)
├── requirements.txt           # Dependencies (streamlit, pymupdf, nltk, scikit-learn, sentence-transformers)
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── src/
│   ├── __init__.py            # Package exports
│   ├── pdf_processor.py       # PyMuPDF page-by-page text extraction (Phase 1)
│   ├── text_processor.py      # Text cleaning, tokenization & stopwords (Phase 1)
│   ├── chunker.py             # Semantic chunking, metadata & statistics (Phase 2)
│   ├── retriever.py           # TF-IDF & Cosine Similarity retrieval (Phase 3)
│   ├── qa_engine.py           # Extractive Question Answering engine (Phase 4)
│   └── semantic_retriever.py  # Dense Sentence Transformers retrieval (Phase 5)
│
├── data/
│   └── uploads/               # Temporary upload storage and sample documents
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py         # Automated test suite for Phase 1 (8 tests)
    ├── test_phase2.py         # Automated test suite for Phase 2 (9 tests)
    ├── test_phase3.py         # Automated test suite for Phase 3 (10 tests)
    ├── test_phase4.py         # Automated test suite for Phase 4 (10 tests)
    └── test_semantic_retriever.py # Automated test suite for Phase 5 (10 tests)
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
2. In the **"Question Answering"** section, select your preferred **Retrieval Method**:
   * `Semantic Embeddings` (Sentence Transformers `all-MiniLM-L6-v2`)
   * `TF-IDF` (Lexical unigram/bigram cosine similarity)
3. Enter a natural language question (e.g. *"What are the attendance requirements?"*).
4. Inspect the extracted **Answer**, the **Source Attribution** (Retrieval Method, Document, Page, Chunk ID, Similarity Score, Sentence Score), the **Retrieved Evidence**, and the candidate retrieval ranking.

---

## Running Automated Tests

Run the complete automated test suite across all 5 phases (**47 tests**):

```bash
python -m unittest discover tests
```

To run individual phase test suites:

```bash
python -m unittest tests/test_semantic_retriever.py
python -m unittest tests/test_phase4.py
python -m unittest tests/test_phase3.py
python -m unittest tests/test_phase2.py
python -m unittest tests/test_phase1.py
```
