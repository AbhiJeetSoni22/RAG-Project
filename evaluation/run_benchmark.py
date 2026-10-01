"""Benchmark runner for Phase 7 retrieval evaluation on sample_document.pdf.

Evaluates and compares:
- TF-IDF Lexical Retrieval
- Semantic Dense Embedding Retrieval (all-MiniLM-L6-v2)
- Hybrid Retrieval with weights α=0.25, α=0.50, α=0.75
across standard Information Retrieval metrics (Hit@1, Hit@3, Hit@5, Precision@5, Recall@5, MRR)
and exports empirical measurements to evaluation/results.csv and evaluation/summary.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_document
from src.retriever import TFIDFRetriever
from src.semantic_retriever import SemanticRetriever, get_sentence_transformer
from src.hybrid_retriever import HybridRetriever
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

    print("\nStep 3: Initializing TF-IDF, Semantic, and Hybrid retrievers...")
    tfidf_retriever = TFIDFRetriever(chunks)
    st_model = get_sentence_transformer()
    semantic_retriever = SemanticRetriever(chunks=chunks, model=st_model)

    # Initialize Hybrid retrievers sharing the underlying index instances
    hybrid_025 = HybridRetriever(
        chunks=chunks,
        alpha=0.25,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )
    hybrid_050 = HybridRetriever(
        chunks=chunks,
        alpha=0.50,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )
    hybrid_075 = HybridRetriever(
        chunks=chunks,
        alpha=0.75,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )

    print("\nStep 4: Running comparative evaluation (K = [1, 3, 5])...")
    retrievers = {
        "TF-IDF": tfidf_retriever,
        "Semantic": semantic_retriever,
        "Hybrid (alpha=0.25)": hybrid_025,
        "Hybrid (alpha=0.50)": hybrid_050,
        "Hybrid (alpha=0.75)": hybrid_075,
    }

    comparison = compare_retrieval_methods(
        questions=questions,
        retrievers=retrievers,
        k_values=[1, 3, 5],
        min_similarity=0.0,
    )

    # 1. Full Method Comparison Table
    print("\n" + "=" * 95)
    print("PHASE 7 BENCHMARK: RETRIEVAL METHOD COMPARISON (K = [1, 3, 5])")
    print("=" * 95)
    col_width = 17
    headers = ["Metric"] + list(retrievers.keys())
    header_str = f"{headers[0]:<16}" + "".join(f"{h:>{col_width}}" for h in headers[1:])
    print(header_str)
    print("-" * len(header_str))

    for row in comparison.overall_metrics_table:
        line = f"{row['Metric']:<16}"
        for name in retrievers.keys():
            val_score = row.get(name, 0.0)
            line += f"{val_score:>{col_width}.4f}"
        print(line)
    print("=" * len(header_str))

    # 2. Main Retrieval Methods Comparison Table (TF-IDF vs Semantic vs Hybrid alpha=0.50)
    print("\n" + "=" * 65)
    print("CORE COMPARISON: TF-IDF vs. SEMANTIC vs. HYBRID (alpha=0.50)")
    print("=" * 65)
    print(f"{'Metric':<16} {'TF-IDF':>14} {'Semantic':>14} {'Hybrid (alpha=0.50)':>22}")
    print("-" * 65)
    for row in comparison.overall_metrics_table:
        t_val = row.get("TF-IDF", 0.0)
        s_val = row.get("Semantic", 0.0)
        h_val = row.get("Hybrid (alpha=0.50)", 0.0)
        print(f"{row['Metric']:<16} {t_val:>14.4f} {s_val:>14.4f} {h_val:>22.4f}")
    print("=" * 65)

    # 3. Hybrid Weight Experiment Table (Section 12)
    print("\n" + "=" * 70)
    print("HYBRID WEIGHT EXPERIMENT (alpha = 0.25 vs 0.50 vs 0.75)")
    print("=" * 70)
    print(f"{'Weight (alpha)':<16} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'Prec@5':>8} {'Rec@5':>8} {'MRR':>8}")
    print("-" * 70)

    weights_to_show = [
        ("0.25 (Semantic)", "Hybrid (alpha=0.25)"),
        ("0.50 (Equal)", "Hybrid (alpha=0.50)"),
        ("0.75 (Lexical)", "Hybrid (alpha=0.75)"),
    ]
    for label, method_key in weights_to_show:
        s = comparison.summaries[method_key]
        h1 = s.hit_rates.get(1, 0.0)
        h3 = s.hit_rates.get(3, 0.0)
        h5 = s.hit_rates.get(5, 0.0)
        p5 = s.precision_at_k.get(5, 0.0)
        r5 = s.recall_at_k.get(5, 0.0)
        mrr = s.mrr
        print(f"{label:<16} {h1:>8.4f} {h3:>8.4f} {h5:>8.4f} {p5:>8.4f} {r5:>8.4f} {mrr:>8.4f}")
    print("=" * 70)

    # 4. Category Breakdown Table (Section 13)
    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS: HYBRID (alpha=0.50) vs BASELINES")
    print("=" * 80)
    print(f"{'Category':<16} {'Count':>6} {'Hybrid Hit@5':>14} {'Hybrid MRR':>12} {'TF-IDF MRR':>12} {'Semantic MRR':>14}")
    print("-" * 80)
    for row in comparison.category_metrics_table:
        cat = row["Category"]
        cnt = row["Count"]
        h_hit = row.get("Hybrid (alpha=0.50) Hit@5", 0.0)
        h_mrr = row.get("Hybrid (alpha=0.50) MRR", 0.0)
        t_mrr = row.get("TF-IDF MRR", 0.0)
        s_mrr = row.get("Semantic MRR", 0.0)
        print(f"{cat:<16} {cnt:>6} {h_hit:>14.4f} {h_mrr:>12.4f} {t_mrr:>12.4f} {s_mrr:>14.4f}")
    print("=" * 80)

    # 5. Export to CSV (Section 15)
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
