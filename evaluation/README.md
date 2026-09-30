# Retrieval Evaluation Dataset Framework

This directory contains evaluation resources for benchmarking and comparing retrieval engines in Phase 6.

## Dataset Structure (`qa_dataset.json`)

The evaluation dataset consists of manually formulated queries grounded in the actual document chunks produced by the Phase 2 chunking pipeline.

### Schema:
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

### Fields:
* `id` (*string*, required): Unique question identifier (e.g. `q001`, `q002`).
* `question` (*string*, required): The natural language query to evaluate.
* `category` (*string*, recommended): Linguistic query type for sub-group analysis. Supported categories:
  * `exact_keyword`: Questions containing identical keywords and surface phrasing as the source passage.
  * `paraphrased`: Questions expressing the concept using synonyms or alternative sentence structures without exact lexical overlap.
  * `conceptual`: Questions testing higher-level concepts, reasoning, or methodological rationale.
  * `definition`: Questions asking for the meaning or scope of a domain term.
  * `factual`: Questions seeking specific declared facts or system attributes.
  * `numeric`: Questions regarding counts, figures, or quantities.
* `relevant_chunk_ids` (*list of strings*, required): Ground-truth chunk identifiers (`page_{N}_chunk_{M}`) that contain the relevant information.
* `relevant_pages` (*list of integers*, optional): Page numbers corresponding to the relevant chunks.
* `source` (*string*, optional): Name of the source document file.

---

## Documented Mechanism for Obtaining Valid Chunk IDs

To ensure evaluations are strictly grounded in reality and avoid synthetic or hallucinated IDs, ground-truth chunk IDs should be derived directly from the document processing pipeline:

### Programmatic Extraction:
```python
from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_document

# 1. Ingest PDF
pages = extract_text_from_pdf("data/uploads/sample_document.pdf")

# 2. Chunk with default pipeline settings (chunk_size=500, chunk_overlap=50)
chunks = chunk_document(pages, source="sample_document.pdf")

# 3. Inspect chunk IDs and passages
for chunk in chunks:
    print(f"ID: {chunk['chunk_id']} | Page: {chunk['page']}")
    print(f"Text snippet: {chunk['original_text'][:100]}...\n")
```

### Via the Streamlit UI:
1. Upload your PDF in the web interface.
2. Expand the **"📑 View Chunks"** section.
3. Every chunk displays its verified identifier (e.g., `page_1_chunk_1`) along with the page number and full original passage text.
4. Use these exact chunk IDs when annotating `relevant_chunk_ids` in `qa_dataset.json`.
