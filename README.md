# Retrieval Based Question Answering

A modular Natural Language Processing (NLP) system designed to allow users to upload PDF documents, retrieve relevant passages, and extract factual answers strictly based on document content.

---

## Current Status

### **Phase 7 — Hybrid Retrieval (ACTIVE)**

Phase 7 introduces a modular **Hybrid Retrieval Engine** that unifies sparse lexical relevance from TF-IDF with dense contextual similarity from Sentence Transformer embeddings into a single ranked relevance signal:

```text
TF-IDF lexical similarity
+
Semantic embedding similarity
↓
Score Normalization (Min-Max)
↓
Weighted Score Fusion (α)
↓
Hybrid relevance score
↓
Final Top-K ranking
↓
Existing Extractive QA Engine
```

The application continues to support all three retrieval paradigms:
- **TF-IDF** (Lexical matching via n-gram cosine similarity)
- **Semantic Embeddings** (Dense neural embeddings via `all-MiniLM-L6-v2`)
- **Hybrid** (Combined lexical + semantic score fusion)

---

## Phase 7 Architecture

```text
                    Question
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
           TF-IDF           Semantic
          Retrieval          Retrieval
              ↓                 ↓
          Lexical Score    Semantic Score
              └────────┬────────┘
                       ↓
              Candidate Pool Union
                       ↓
             Score Normalization
                       ↓
                 Score Fusion (α)
                       ↓
                Hybrid Ranking
                       ↓
                     Top-K
                       ↓
             Extractive QA Engine
                       ↓
              Exact Answer + Proof
```

---

## Why Hybrid Retrieval?

1. **Lexical Matching (TF-IDF)** excels when queries contain exact proper nouns, specialized acronyms, identifiers, or specific keywords. However, it fails when the user paraphrases or uses synonyms.
2. **Dense Semantic Embeddings (Sentence Transformers)** excel at capturing semantic intent, contextual meaning, and paraphrased formulations, but can occasionally over-generalize or assign lower rank to precise lexical matches.
3. **Hybrid Retrieval** combines both signals:
   - Preserves high rank for exact keyword matches via the TF-IDF channel.
   - Recovers semantic meaning for paraphrased and conceptual queries via the embedding channel.
   - Provides explainability through exposed component scores (`tfidf_score`, `semantic_score`, `normalized_tfidf_score`, `normalized_semantic_score`, `hybrid_score`).

---

## Mathematical Formulation & Design Details

### 1. Score Normalization (Min-Max)
Raw TF-IDF and semantic similarity scores have distinct distributions and dynamic ranges. Before fusion, raw scores across the candidate pool are normalized into $[0.0, 1.0]$ using deterministic min-max scaling:

$$\text{NormalizedScore}(c) = \frac{\text{Score}(c) - \min(\text{Scores})}{\max(\text{Scores}) - \min(\text{Scores})}$$

**Division-by-Zero Handling:**
When all candidates in the candidate pool have identical scores ($\max(\text{Scores}) == \min(\text{Scores})$):
- If $\max(\text{Scores}) > 0$, the normalized score is set to `1.0` (all candidates have tied positive similarity).
- If $\max(\text{Scores}) == 0$, the normalized score is set to `0.0`.

### 2. Candidate Union Strategy
Rather than restricting evaluation to the intersection of candidate lists ($A \cap B$), the hybrid retriever pools the **union** of relevant candidates from both underlying retrieval engines ($A \cup B$):

$$\text{CandidatePool} = \text{Top-K}_{\text{TF-IDF}} \cup \text{Top-K}_{\text{Semantic}}$$

This ensures that a passage discovered exclusively by one engine (e.g. an exact lexical match missed by semantic search, or a paraphrased passage with zero lexical overlap) remains fully eligible for the final ranking.

### 3. Missing Score Handling
When a candidate appears in one retriever's candidate set but is absent from the other:
- If present in TF-IDF only: $\text{RawSemanticScore} = 0.0$.
- If present in Semantic only: $\text{RawTFIDFScore} = 0.0$.

Missing scores are assigned `0.0` deterministically rather than an arbitrary high value, correctly reflecting the absence of evidence from that retrieval channel.

### 4. Weighted Score Fusion
The final hybrid relevance score is computed as a convex combination parametrized by $\alpha \in [0.0, 1.0]$:

$$\text{HybridScore} = \alpha \times \text{NormalizedTFIDF} + (1 - \alpha) \times \text{NormalizedSemantic}$$

- **$\alpha = 0.50$ (Default):** Equal contribution from lexical and semantic channels.
- **$\alpha = 1.00$:** Purely lexical TF-IDF ranking behavior.
- **$\alpha = 0.00$:** Purely semantic embedding ranking behavior.
- Configurable dynamically both via the `HybridRetriever(alpha=...)` constructor and the `search(alpha=...)` query override. Invalid weights outside $[0.0, 1.0]$ are strictly rejected.

### 5. Metadata Preservation & Explainability
Every retrieved passage preserves all Phase 2 metadata (`chunk_id`, `source`, `page`, `original_text`, `cleaned_text`) and exposes complete explainability scores:
- `tfidf_score`: Raw cosine similarity from TF-IDF vectorizer.
- `semantic_score`: Raw cosine similarity from Sentence Transformer embeddings.
- `normalized_tfidf_score`: Min-max scaled TF-IDF score in $[0.0, 1.0]$.
- `normalized_semantic_score`: Min-max scaled semantic score in $[0.0, 1.0]$.
- `hybrid_score`: Final combined relevance score in $[0.0, 1.0]$.
- `similarity_score`: Alias of `hybrid_score` providing seamless compatibility with `ExtractiveQAEngine`.

---

## Empirical Benchmark Results (Phase 7)

All metrics are empirically calculated from `evaluation/qa_dataset.json` (15 verified questions) on `sample_document.pdf` across cutoff ranks $K \in [1, 3, 5]$ using `evaluation/run_benchmark.py`.

### 1. Comparative Performance Across All Methods

| Metric | TF-IDF | Semantic | Hybrid ($\alpha=0.25$) | Hybrid ($\alpha=0.50$) | Hybrid ($\alpha=0.75$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hit@1** | 0.8000 | 0.9333 | 0.9333 | **0.9333** | 0.8667 |
| **Hit@3** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Hit@5** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Precision@5** | 0.2133 | 0.2133 | 0.2133 | **0.2133** | 0.2133 |
| **Recall@5** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **MRR** | 0.8889 | 0.9667 | 0.9667 | **0.9667** | 0.9333 |

### 2. Hybrid Weight Experiment ($\alpha = 0.25$ vs. $0.50$ vs. $0.75$)

| Weight ($\alpha$) | Configuration Focus | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Recall@5 | MRR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.25** | Stronger Semantic Influence | 0.9333 | 1.0000 | 1.0000 | 0.2133 | 1.0000 | 0.9667 |
| **0.50** | Balanced Equal Contribution (Default) | 0.9333 | 1.0000 | 1.0000 | 0.2133 | 1.0000 | 0.9667 |
| **0.75** | Stronger Lexical Influence | 0.8667 | 1.0000 | 1.0000 | 0.2133 | 1.0000 | 0.9333 |

### 3. Category Breakdown (Hybrid $\alpha=0.50$ vs. Baselines)

| Category | Queries | Hybrid Hit@5 | Hybrid MRR | TF-IDF MRR | Semantic MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **conceptual** | 4 | 1.0000 | **1.0000** | 0.8750 | 1.0000 |
| **definition** | 2 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **exact_keyword** | 2 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **factual** | 3 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **numeric** | 1 | 1.0000 | **1.0000** | 0.5000 | 1.0000 |
| **paraphrased** | 3 | 1.0000 | **0.8333** | 0.7778 | 0.8333 |

### Key Benchmark Findings:
1. **Balanced Fusion ($\alpha=0.50$):** Achieves **0.9333 Hit@1** and **0.9667 MRR**, significantly improving over pure TF-IDF (0.8000 Hit@1, 0.8889 MRR) and matching Semantic retrieval.
2. **Lexical Weight Sensitivity:** Shifting $\alpha$ from 0.50 to 0.75 (heavier lexical weight) yields a 0.8667 Hit@1 and 0.9333 MRR, demonstrating that semantic embeddings provide crucial disambiguation on paraphrased and conceptual questions while lexical signals remain effective for exact keywords.
3. **100% Coverage at K=3 and K=5:** Across all configurations, Hit@3 and Hit@5 reach 1.0000, confirming that the candidate union ensures ground-truth passages are consistently present in the top retrieved set.

---

## Technical Comparison of All Retrieval Methods

| Feature | TF-IDF (Phase 3) | Semantic Embeddings (Phase 5) | Hybrid Retrieval (Phase 7) |
| :--- | :--- | :--- | :--- |
| **Representation** | High-dimensional sparse lexical vectors | Dense 384-d semantic embedding vectors | Combined sparse + dense normalized representations |
| **Matching Basis** | Surface word & n-gram co-occurrence | Learned contextual similarity | Joint lexical overlap + contextual semantic match |
| **Synonym Handling** | Limited (requires exact token match) | High (semantic space clustering) | Robust (semantic channel recovers synonyms) |
| **Out-of-Vocab / Acronyms** | Strong for rare exact keywords | Can dilute rare exact tokens | Strong (lexical channel preserves exact matches) |
| **Score Normalization** | Raw cosine similarity $[0.0, 1.0]$ | Unit-normalized dot product $[-1.0, 1.0]$ | Min-max scaled normalized fusion $[0.0, 1.0]$ |
| **Candidate Sourcing** | Top-K lexical chunks | Top-K dense embedding chunks | Candidate union pool ($\text{TF-IDF} \cup \text{Semantic}$) |
| **Tunable Balance** | Fixed lexical | Fixed semantic | Fully tunable $\alpha \in [0.0, 1.0]$ |
| **Measured MRR** | `0.8889` | `0.9667` | `0.9667` ($\alpha=0.50$) |
| **Measured Hit@1** | `0.8000` | `0.9333` | `0.9333` ($\alpha=0.50$) |

---

## Project Structure

```text
retrieval-qa/
│
├── app.py                     # Streamlit web application (Phases 1-7)
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── src/
│   ├── __init__.py            # Package exports (retrievers, QA engine, evaluator)
│   ├── pdf_processor.py       # PyMuPDF page-by-page text extraction (Phase 1)
│   ├── text_processor.py      # Text cleaning, tokenization & stopwords (Phase 1)
│   ├── chunker.py             # Semantic chunking, metadata & statistics (Phase 2)
│   ├── retriever.py           # TF-IDF & Cosine Similarity retrieval (Phase 3)
│   ├── qa_engine.py           # Extractive Question Answering engine (Phase 4)
│   ├── semantic_retriever.py  # Dense Sentence Transformers retrieval (Phase 5)
│   ├── hybrid_retriever.py    # Hybrid Lexical + Semantic retrieval (Phase 7)
│   └── evaluator.py           # IR Evaluation & Benchmarking metrics (Phase 6)
│
├── data/
│   └── uploads/               # Sample document and uploads
│       └── sample_document.pdf
│
├── evaluation/
│   ├── qa_dataset.json        # Ground-truth evaluation dataset (15 queries)
│   ├── results.csv            # Exported query-level evaluation results (with alpha)
│   ├── summary.csv            # Exported method comparison summary (with alpha)
│   ├── run_benchmark.py       # Standalone CLI evaluation runner (Phases 3, 5, 7)
│   └── README.md              # Dataset schema and documentation
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py             # Phase 1 test suite (8 tests)
    ├── test_phase2.py             # Phase 2 test suite (9 tests)
    ├── test_phase3.py             # Phase 3 test suite (10 tests)
    ├── test_phase4.py             # Phase 4 test suite (10 tests)
    ├── test_semantic_retriever.py # Phase 5 test suite (10 tests)
    ├── test_evaluator.py          # Phase 6 test suite (14 tests)
    └── test_hybrid_retriever.py   # Phase 7 test suite (18 tests)
```

---

## Installation & Setup

### 1. Activate the virtual environment
On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies (if needed)
```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

### User Interface Features:
1. **Retrieval Method Selector:** Choose between **Hybrid** (default), **Semantic Embeddings**, or **TF-IDF**.
2. **Hybrid Lexical Weight ($\alpha$):** Slider in the sidebar to dynamically tune the balance between lexical and semantic signals (default: `0.50`).
3. **Question Answering:** Enter any question; extracts exact factual sentence(s) with full source attribution.
4. **Explainability Drill-down:** When Hybrid is selected, each retrieved passage card displays:
   - Rank
   - Hybrid Score
   - TF-IDF Score
   - Semantic Score
   - Page, Chunk ID, Source, and Original Text
5. **Interactive Benchmarking:** Run side-by-side evaluation directly within the UI and download `results.csv` and `summary.csv`.

---

## Running Automated Tests

Run the complete test suite across all 7 phases (**79 tests**):

```bash
python -m unittest discover tests
```

To run individual test suites:

```bash
python -m unittest tests/test_hybrid_retriever.py   # Phase 7 Hybrid (18 tests)
python -m unittest tests/test_evaluator.py          # Phase 6 Evaluation (14 tests)
python -m unittest tests/test_semantic_retriever.py # Phase 5 Semantic (10 tests)
python -m unittest tests/test_phase4.py             # Phase 4 QA Engine (10 tests)
python -m unittest tests/test_phase3.py             # Phase 3 TF-IDF (10 tests)
python -m unittest tests/test_phase2.py             # Phase 2 Chunking (9 tests)
python -m unittest tests/test_phase1.py             # Phase 1 Ingestion (8 tests)
```

---

## Running the CLI Evaluation Benchmark

To execute the benchmark runner comparing TF-IDF, Semantic, and Hybrid retrieval across multiple weights:

```bash
python evaluation/run_benchmark.py
```

Outputs comprehensive metric tables to the console and updates `evaluation/results.csv` and `evaluation/summary.csv`.
