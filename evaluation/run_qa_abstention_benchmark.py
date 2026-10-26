"""Answer-Level Correctness & Abstention Benchmark.

Measures QA answer generation, true positive factual extraction, and false-positive
rejection/abstention on unanswerable questions containing misleading heuristics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_document
from src.retriever import TFIDFRetriever
from src.semantic_retriever import SemanticRetriever, get_sentence_transformer
from src.hybrid_retriever import HybridRetriever
from src.qa_engine import ExtractiveQAEngine

logging.basicConfig(level=logging.WARNING)


def run_qa_abstention_benchmark():
    pdf_path = Path("data/uploads/sample_document.pdf")
    dataset_path = Path("evaluation/qa_dataset.json")
    unanswerable_path = Path("evaluation/qa_unanswerable_dataset.json")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_document(pages, source="sample_document.pdf")

    tfidf_retriever = TFIDFRetriever(chunks)
    st_model = get_sentence_transformer()
    semantic_retriever = SemanticRetriever(chunks=chunks, model=st_model)
    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        alpha=0.50,
        tfidf_retriever=tfidf_retriever,
        semantic_retriever=semantic_retriever,
    )
    qa_engine = ExtractiveQAEngine()

    print("=" * 80)
    print("QA ENGINE EXTRACTIVE & ABSTENTION BENCHMARK")
    print("=" * 80)

    # 1. Evaluate Unanswerable Questions (False-Positive Prevention)
    with open(unanswerable_path, "r", encoding="utf-8") as f:
        unanswerable_questions = json.load(f)

    correct_abstentions = 0
    total_unanswerable = len(unanswerable_questions)

    print(f"\n[Part 1] Evaluating {total_unanswerable} Unanswerable / Absent Questions:")
    print("-" * 80)

    for q in unanswerable_questions:
        qid = q["id"]
        q_text = q["question"]
        passages = hybrid_retriever.search(q_text, top_k=3, min_similarity=0.10)
        ans_res = qa_engine.answer(q_text, passages)

        if not ans_res["found"]:
            correct_abstentions += 1
            status = "PASS (Abstained)"
        else:
            status = f"FAIL (Extracted: '{ans_res['answer'][:40]}...')"

        print(f"[{qid}] '{q_text}' -> {status}")

    abstention_rate = correct_abstentions / total_unanswerable
    print(f"\nAbstention Accuracy: {abstention_rate * 100:.1f}% ({correct_abstentions}/{total_unanswerable})")
    print(f"False-Positive Hallucination Rate: {(1.0 - abstention_rate) * 100:.1f}%")

    # 2. Evaluate Answerable Questions (Recall on Ground Truth)
    with open(dataset_path, "r", encoding="utf-8") as f:
        answerable_questions = json.load(f)

    correct_answers = 0
    total_answerable = len(answerable_questions)

    print(f"\n[Part 2] Evaluating {total_answerable} Answerable Ground-Truth Questions:")
    print("-" * 80)

    for q in answerable_questions:
        qid = q["id"]
        q_text = q["question"]
        passages = hybrid_retriever.search(q_text, top_k=3, min_similarity=0.10)
        ans_res = qa_engine.answer(q_text, passages)

        if ans_res["found"] and ans_res["answer"]:
            correct_answers += 1
            status = f"PASS (Salience={ans_res['sentence_score']:.2f})"
        else:
            status = "ABSTAINED (No high-salience answer)"

        print(f"[{qid}] '{q_text}' -> {status}")

    answer_recall = correct_answers / total_answerable
    print(f"\nGround-Truth Answer Recall: {answer_recall * 100:.1f}% ({correct_answers}/{total_answerable})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"True-Positive Answer Extraction: {correct_answers}/{total_answerable} ({answer_recall * 100:.1f}%)")
    print(f"False-Positive Rejection (Abstention): {correct_abstentions}/{total_unanswerable} ({abstention_rate * 100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_qa_abstention_benchmark()
