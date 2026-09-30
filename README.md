# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and extract factual answers strictly based on document content.

---

## Current Status

### **Phase 6 — Retrieval Evaluation (ACTIVE)**

Phase 6 introduces an objective, standardized benchmarking and evaluation framework to systematically measure and compare the existing retrieval engines:
1. **Lexical TF-IDF Retrieval** (Phase 3)
2. **Dense Semantic Embedding Retrieval** (Phase 5)

The evaluation layer computes standard Information Retrieval (IR) metrics—**Hit@K**, **Precision@K**, **Recall@K**, and **Mean Reciprocal Rank (MRR)**—against a verified ground-truth dataset grounded in actual document passages.

```text
Evaluation Dataset
       ↓
Evaluation Questions
       ↓
 ┌───────────────┐
 │               │
TF-IDF        Semantic
Retriever     Retriever
 │               │
 └───────┬───────┘
         ↓
      Top-K
         ↓
   Evaluation Layer
         ↓
 ┌───────────────────────┐
 │ Hit@K                 │
 │ Precision@K           │
 │ Recall@K              │
 │ MRR                   │
 └───────────────────────┘
```

> **Evaluation Principle:** Benchmarking is strictly empirical and objective. The system reports measured metrics without subjective rankings or arbitrary scores.

---

## Phase 6 — Retrieval Evaluation

### 1. Why Retrieval Evaluation is Required
In earlier phases, retrieval quality was assessed qualitatively via manual queries. However, system development requires quantitative, reproducible benchmarks to answer:
* How often does a retriever find the relevant chunk in its top-1, top-3, or top-5 candidates?
* How early in the ranked list does the first relevant passage appear?
* How do lexical (TF-IDF) and dense semantic methods differ across distinct query patterns (exact keyword vs. paraphrased vs. conceptual questions)?

Retrieval evaluation provides empirical evidence to understand algorithm behavior without relying on subjective impressions.

### 2. Ground-Truth Evaluation Dataset
The benchmark uses `evaluation/qa_dataset.json`, where every query is manually curated and mapped to verified ground-truth chunk IDs produced by the Phase 2 chunking pipeline:

```json
[
  {
    "id": "q001",
    "question": "What is Natural Language Processing?",
    "category": "definition",
    "relevant_chunk_ids": [
      "page_1_chunk_1"
    ],
    "relevant_pages": [1],
    "source": "sample_document.pdf"
  }
]
```

#### Supported Linguistic Categories:
* `exact_keyword`: Questions containing identical keywords and surface phrasing as the source text.
* `paraphrased`: Questions expressing the concept using synonyms or alternative syntactic structures without exact surface overlap.
* `conceptual`: Questions probing underlying mechanisms, architectural rationale, or methodology.
* `definition`: Questions asking for the formal scope or definition of a domain term.
* `factual`: Questions seeking specific stated facts or documented attributes.
* `numeric`: Questions regarding numbers, counts, or quantities.

### 3. Evaluation Metrics

#### Hit@K
A query is considered a hit (value `1`) if at least one ground-truth relevant chunk appears within the top $K$ retrieved passages; otherwise `0`.

$$\text{Hit@K} = \begin{cases} 1 & \text{if } \text{Top-K}(\mathbf{q}) \cap \text{Relevant}(\mathbf{q}) \neq \emptyset \\ 0 & \text{otherwise} \end{cases}$$

$$\text{Hit Rate@K} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{Hit@K}(q)$$

#### Precision@K
Measures the proportion of retrieved chunks within the top $K$ candidates that are relevant:

$$\text{Precision@K} = \frac{|\text{Top-K}(\mathbf{q}) \cap \text{Relevant}(\mathbf{q})|}{K}$$

#### Recall@K
Measures the proportion of all ground-truth relevant chunks that were successfully retrieved in the top $K$:

$$\text{Recall@K} = \frac{|\text{Top-K}(\mathbf{q}) \cap \text{Relevant}(\mathbf{q})|}{|\text{Relevant}(\mathbf{q})|}$$

#### Mean Reciprocal Rank (MRR)
Evaluates how high in the ranking the first relevant chunk appears. If the first relevant chunk appears at 1-based rank $r$:
$$\text{RR}(q) = \frac{1}{r} \quad (\text{or } 0 \text{ if no relevant chunk is retrieved})$$

$$\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{RR}(q)$$

---

## Empirical Benchmark Results

> **Academic Rule:** The system reports measured experimental values directly. Algorithm characteristics are distinguished from measured empirical results below.

### Measured Results on `sample_document.pdf` (15 Questions, K = [1, 3, 5])

```text
==================================================
Metric           TF-IDF       Semantic    
--------------------------------------------------
Hit@1            0.8000       0.9333      
Hit@3            1.0000       1.0000      
Hit@5            1.0000       1.0000      
Precision@5      0.2133       0.2133      
Recall@5         1.0000       1.0000      
MRR              0.8889       0.9667      
==================================================
```

### Measured Category Breakdown:

| Category | Queries | TF-IDF Hit@5 | Semantic Hit@5 | TF-IDF MRR | Semantic MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **exact_keyword** | 2 | 1.00 | 1.00 | 1.0000 | 1.0000 |
| **definition** | 2 | 1.00 | 1.00 | 1.0000 | 1.0000 |
| **factual** | 3 | 1.00 | 1.00 | 1.0000 | 1.0000 |
| **conceptual** | 4 | 1.00 | 1.00 | 0.8750 | 1.0000 |
| **numeric** | 1 | 1.00 | 1.00 | 0.5000 | 1.0000 |
| **paraphrased** | 3 | 1.00 | 1.00 | 0.7778 | 0.8333 |

### Observations from Measured Data:
1. On `exact_keyword`, `definition`, and `factual` queries where exact terms are preserved, both TF-IDF and Semantic retrieval achieve an MRR of 1.0000.
2. On `paraphrased` queries where wording differs from the source text, Semantic retrieval measured higher reciprocal ranks (MRR: 0.8333 vs. 0.7778).
3. On `conceptual` queries, Semantic retrieval consistently placed the relevant chunk at rank 1 (MRR: 1.0000 vs. 0.8750).
4. Both methods achieved 100% Hit Rate at $K=3$ and $K=5$ on this document collection.

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
* **Multi-Factor Sentence Scoring:** Lexical overlap ratio, sentence-level cosine similarity, answer-type cues, and retrieval confidence prior.
* **Concise Multi-Sentence Answers:** Dynamically joins adjacent highly relevant sentences up to a configurable maximum (default: 2 sentences).
* **Strict Document Grounding:** If retrieval returns no chunks above the similarity threshold, the QA engine refuses to answer and reports that no sufficiently relevant information was found.
* **Complete Source Attribution:** Every answer displays the source PDF filename, page number, chunk identifier, retrieval similarity score, and the exact `original_text` evidence passage.

### Phase 5: Semantic Retrieval Enhancement
* **Dense Embedding Retriever (`SemanticRetriever`):** Generates 384-dimensional normalized dense embeddings for document chunks using Sentence Transformers (`all-MiniLM-L6-v2`).
* **Semantic Paraphrase Matching:** Pairs user queries with relevant passages even when wording, synonyms, or sentence structures differ.
* **Dual Retrieval Mode Selector:** Toggle seamlessly between `Semantic Embeddings` (default) and `TF-IDF` via the UI without reloading documents.
* **High-Performance In-Memory Search:** Vectorized dot-product cosine similarity computation.

### Phase 6: Retrieval Evaluation & Benchmarking
* **Standard IR Metrics:** Implementation of Hit@K, Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
* **Objective Comparative Benchmark:** Side-by-side evaluation of TF-IDF and Semantic retrieval on identical evaluation sets and identical Top-K cutoffs.
* **Category-Level Linguistic Analysis:** Aggregates metrics grouped by query category (`exact_keyword`, `paraphrased`, `conceptual`, `definition`, `factual`, `numeric`).
* **Dataset Grounding & Validation:** Automated alignment checks verifying that evaluation chunk IDs correspond to valid document chunks.
* **CSV Result Export:** Exports detailed query-level measurements (`evaluation/results.csv`) and comparative summary tables (`evaluation/summary.csv`).
* **Interactive Streamlit UI:** Dedicated evaluation section with explicit execution trigger, metric highlights, drill-down breakdown, and download buttons.

---

## Technical Comparison: TF-IDF vs. Semantic Embeddings

| Feature | TF-IDF Retrieval (Phase 3) | Semantic Embeddings (Phase 5) |
| :--- | :--- | :--- |
| **Representation** | High-dimensional, sparse lexical vectors | Low-dimensional (384-d), dense semantic vectors |
| **Matching Basis** | Exact word & n-gram overlap | Learned contextual semantic similarity |
| **Synonym Handling** | Limited (misses distinct terms with similar meaning) | Maps semantically related phrases closely |
| **Paraphrase Resilience** | Vulnerable to surface rephrasing | High resilience to varying syntactic patterns |
| **Computation Cost** | Extremely low CPU overhead | Moderate CPU overhead during chunk indexing (cached) |
| **Model Footprint** | Zero external neural weights | Lightweight model weights (~80MB, cached locally) |
| **Measured MRR (sample doc)** | `0.8889` | `0.9667` |

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                     # Streamlit web application (Phases 1, 2, 3, 4, 5 & 6)
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
│   ├── semantic_retriever.py  # Dense Sentence Transformers retrieval (Phase 5)
│   └── evaluator.py           # IR Evaluation & Benchmarking metrics (Phase 6)
│
├── data/
│   └── uploads/               # Sample document and uploads
│       └── sample_document.pdf
│
├── evaluation/
│   ├── qa_dataset.json        # Ground-truth evaluation dataset (15 queries)
│   ├── results.csv            # Exported query-level evaluation results
│   ├── summary.csv            # Exported method comparison summary
│   ├── run_benchmark.py       # Standalone CLI evaluation runner
│   └── README.md              # Dataset schema and chunk ID extraction documentation
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py         # Automated test suite for Phase 1 (8 tests)
    ├── test_phase2.py         # Automated test suite for Phase 2 (9 tests)
    ├── test_phase3.py         # Automated test suite for Phase 3 (10 tests)
    ├── test_phase4.py         # Automated test suite for Phase 4 (10 tests)
    ├── test_semantic_retriever.py # Automated test suite for Phase 5 (10 tests)
    └── test_evaluator.py      # Automated test suite for Phase 6 (14 tests)
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
1. Upload any PDF document (or use `data/uploads/sample_document.pdf`).
2. In the **"Question Answering"** section, query the document via **Semantic Embeddings** or **TF-IDF**.
3. In the **"Retrieval Evaluation & Benchmarking"** section, select the evaluation dataset and click **"Run Retrieval Evaluation"**.
4. Inspect the comparative metrics table, category breakdown, per-question drilldown, and download CSV reports.

---

## Running Automated Tests

Run the complete automated test suite across all 6 phases (**61 tests**):

```bash
python -m unittest discover tests
```

To run individual phase test suites:

```bash
python -m unittest tests/test_evaluator.py
python -m unittest tests/test_semantic_retriever.py
python -m unittest tests/test_phase4.py
python -m unittest tests/test_phase3.py
python -m unittest tests/test_phase2.py
python -m unittest tests/test_phase1.py
```

---

## Running the Evaluation Benchmark from CLI

To benchmark retrieval engines from the terminal and export CSV metrics:

```bash
python evaluation/run_benchmark.py
```
