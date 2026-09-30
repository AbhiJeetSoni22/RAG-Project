"""Benchmark runner for Phase 6 retrieval evaluation on sample_document.pdf."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdf_processor import extract_text_from_pdf

from src.chunker import chunk_document
from src.retriever import TFIDFRetriever
from src.semantic_retriever import SemanticRetriever, get_sentence_transformer
from src.evaluator import (
    load_evaluation_dataset,
    validate_dataset_against_chunks,
    compare_retrieval_methods,
    export_results_to_csv,
    export_summary_to_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_benchmark():
    pdf_path = Path("data/uploads/sample_document.pdf")
    dataset_path = Path("evaluation/qa_dataset.json")

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Step 1: Extracting text & chunking sample_document.pdf...")
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_document(pages, source="sample_document.pdf")
    print(f"  Extracted {len(pages)} pages, created {len(chunks)} chunks.")

    print("\nStep 2: Loading & validating evaluation dataset...")
    questions = load_evaluation_dataset(dataset_path)
    val = validate_dataset_against_chunks(questions, chunks)
    print(f"  Loaded {len(questions)} evaluation questions.")
    print(f"  Aligned: {val['aligned_questions_count']}/{len(questions)}, Missing chunks: {val['missing_chunks_count']}")

    print("\nStep 3: Initializing TF-IDF & Semantic retrievers...")
    tfidf_retriever = TFIDFRetriever(chunks)
    st_model = get_sentence_transformer()
    semantic_retriever = SemanticRetriever(chunks=chunks, model=st_model)

    print("\nStep 4: Running comparative evaluation (K = [1, 3, 5])...")
    comparison = compare_retrieval_methods(
        questions=questions,
        retrievers={
            "TF-IDF": tfidf_retriever,
            "Semantic": semantic_retriever,
        },
        k_values=[1, 3, 5],
        min_similarity=0.0,
    )

    print("\n" + "=" * 50)
    print(f"{'Metric':<16} {'TF-IDF':<12} {'Semantic':<12}")
    print("-" * 50)
    for row in comparison.overall_metrics_table:
        print(f"{row['Metric']:<16} {row['TF-IDF']:<12.4f} {row['Semantic']:<12.4f}")
    print("=" * 50)

    print("\n=== CATEGORY ANALYSIS ===")
    for row in comparison.category_metrics_table:
        cat = row["Category"]
        cnt = row["Count"]
        t_hit = row["TF-IDF Hit@5"]
        s_hit = row["Semantic Hit@5"]
        t_mrr = row["TF-IDF MRR"]
        s_mrr = row["Semantic MRR"]
        print(f"{cat:<16} (n={cnt}): TF-IDF Hit@5={t_hit:.2f}, Semantic Hit@5={s_hit:.2f} | TF-IDF MRR={t_mrr:.4f}, Semantic MRR={s_mrr:.4f}")

    # Export to CSV
    results_path = Path("evaluation/results.csv")
    summary_path = Path("evaluation/summary.csv")

    all_q_results = []
    for s in comparison.summaries.values():
        all_q_results.extend(s.query_results)

    export_results_to_csv(all_q_results, results_path)
    export_summary_to_csv(comparison, summary_path)
    print(f"\nResults successfully exported to:")
    print(f"  - {results_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    run_benchmark()
