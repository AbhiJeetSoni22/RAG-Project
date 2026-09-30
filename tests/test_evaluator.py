"""Automated Test Suite for Phase 6: Retrieval Evaluation & Benchmarking.

Covers:
- Test 1: Correct Hit@K when the relevant chunk is rank 1.
- Test 2: Correct Hit@K when the relevant chunk is rank 3.
- Test 3: Correct miss when no relevant chunk appears.
- Test 4: Correct Precision@K.
- Test 5: Correct Recall@K.
- Test 6: Correct reciprocal rank.
- Test 7: Correct MRR across multiple questions.
- Test 8: Correct comparison between TF-IDF and Semantic results.
- Test 9: Correct category aggregation.
- Test 10: Correct handling of multiple relevant chunks.
- Test 11: Correct handling of empty retrieval results.
- Test 12: Correct handling of an empty evaluation dataset.
- Additional tests: Dataset loading, chunk alignment validation, CSV export.
"""

import json
import unittest
from pathlib import Path

from src.evaluator import (
    compute_hit_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_retriever,
    compare_retrieval_methods,
    load_evaluation_dataset,
    validate_dataset_against_chunks,
    export_results_to_csv,
    export_summary_to_csv,
    EvaluationError,
)


class MockRetriever:
    """Deterministic mock retriever for unit testing without heavy model dependencies."""

    def __init__(self, mapping: dict[str, list[dict]] | None = None) -> None:
        self.mapping = mapping or {}

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict]:
        results = self.mapping.get(query, [])
        filtered = [
            r for r in results
            if float(r.get("similarity_score", 0.0)) >= min_similarity
        ]
        return filtered[:top_k]


class TestPhase6RetrievalEvaluation(unittest.TestCase):
    """Test suite covering all requirements of Phase 6 retrieval evaluation."""

    def setUp(self):
        """Set up standard test fixtures."""
        self.sample_chunks = [
            {"chunk_id": "page_1_chunk_1", "original_text": "Text 1", "page": 1},
            {"chunk_id": "page_1_chunk_2", "original_text": "Text 2", "page": 1},
            {"chunk_id": "page_2_chunk_1", "original_text": "Text 3", "page": 2},
            {"chunk_id": "page_2_chunk_2", "original_text": "Text 4", "page": 2},
            {"chunk_id": "page_3_chunk_1", "original_text": "Text 5", "page": 3},
        ]

    # -------------------------------------------------------------------------
    # Test 1: Correct Hit@K when the relevant chunk is rank 1
    # -------------------------------------------------------------------------
    def test_01_hit_at_k_rank_1(self):
        retrieved = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
        relevant = ["chunk_A"]

        # Hit at K=1, K=3, K=5 must all be 1
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=1), 1)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=3), 1)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=5), 1)

    # -------------------------------------------------------------------------
    # Test 2: Correct Hit@K when the relevant chunk is rank 3
    # -------------------------------------------------------------------------
    def test_02_hit_at_k_rank_3(self):
        retrieved = ["chunk_X", "chunk_Y", "chunk_Z", "chunk_W"]
        relevant = ["chunk_Z"]

        # Hit@1 and Hit@2 must be 0, Hit@3 and Hit@5 must be 1
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=1), 0)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=2), 0)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=3), 1)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=5), 1)

    # -------------------------------------------------------------------------
    # Test 3: Correct miss when no relevant chunk appears
    # -------------------------------------------------------------------------
    def test_03_miss_when_no_relevant_chunk(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        relevant = ["chunk_99"]

        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=1), 0)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=3), 0)
        self.assertEqual(compute_hit_at_k(retrieved, relevant, k=5), 0)
        self.assertEqual(compute_precision_at_k(retrieved, relevant, k=3), 0.0)
        self.assertEqual(compute_recall_at_k(retrieved, relevant, k=3), 0.0)
        self.assertEqual(compute_reciprocal_rank(retrieved, relevant), 0.0)

    # -------------------------------------------------------------------------
    # Test 4: Correct Precision@K
    # -------------------------------------------------------------------------
    def test_04_precision_at_k(self):
        # 5 retrieved, chunks at rank 1 and rank 4 are relevant
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c1", "c4", "c9"]

        # K=1: 1 hit / 1 = 1.0
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, k=1), 1.0)
        # K=2: 1 hit / 2 = 0.5
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, k=2), 0.5)
        # K=3: 1 hit / 3 = 0.3333333333333333
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, k=3), 1.0 / 3.0)
        # K=4: 2 hits / 4 = 0.5
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, k=4), 0.5)
        # K=5: 2 hits / 5 = 0.4
        self.assertAlmostEqual(compute_precision_at_k(retrieved, relevant, k=5), 0.4)

    # -------------------------------------------------------------------------
    # Test 5: Correct Recall@K
    # -------------------------------------------------------------------------
    def test_05_recall_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        # 4 total ground-truth relevant chunks
        relevant = ["c1", "c3", "c8", "c9"]

        # K=1: 1 found / 4 = 0.25
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=1), 0.25)
        # K=2: 1 found / 4 = 0.25
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=2), 0.25)
        # K=3: 2 found (c1, c3) / 4 = 0.50
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=3), 0.50)
        # K=5: 2 found / 4 = 0.50
        self.assertAlmostEqual(compute_recall_at_k(retrieved, relevant, k=5), 0.50)

    # -------------------------------------------------------------------------
    # Test 6: Correct reciprocal rank
    # -------------------------------------------------------------------------
    def test_06_reciprocal_rank(self):
        # Rank 1 -> 1.0
        self.assertAlmostEqual(compute_reciprocal_rank(["c1", "c2"], ["c1"]), 1.0)
        # Rank 2 -> 0.5
        self.assertAlmostEqual(compute_reciprocal_rank(["c1", "c2", "c3"], ["c2"]), 0.5)
        # Rank 3 -> 1/3
        self.assertAlmostEqual(compute_reciprocal_rank(["c1", "c2", "c3"], ["c3"]), 1.0 / 3.0)
        # Rank 4 -> 0.25
        self.assertAlmostEqual(compute_reciprocal_rank(["c1", "c2", "c3", "c4"], ["c4"]), 0.25)
        # Miss -> 0.0
        self.assertAlmostEqual(compute_reciprocal_rank(["c1", "c2"], ["c9"]), 0.0)

    # -------------------------------------------------------------------------
    # Test 7: Correct MRR across multiple questions
    # -------------------------------------------------------------------------
    def test_07_mrr_across_multiple_questions(self):
        # Q1: relevant at rank 1 (RR = 1.0)
        # Q2: relevant at rank 2 (RR = 0.5)
        # Q3: not retrieved (RR = 0.0)
        # Q4: relevant at rank 4 (RR = 0.25)
        # Expected MRR = (1.0 + 0.5 + 0.0 + 0.25) / 4 = 1.75 / 4 = 0.4375
        mock_retriever = MockRetriever({
            "What is A?": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            "What is B?": [{"chunk_id": "cX"}, {"chunk_id": "c2"}],
            "What is C?": [{"chunk_id": "c9"}, {"chunk_id": "c8"}],
            "What is D?": [{"chunk_id": "cA"}, {"chunk_id": "cB"}, {"chunk_id": "cC"}, {"chunk_id": "c4"}],
        })

        questions = [
            {"id": "q1", "question": "What is A?", "category": "def", "relevant_chunk_ids": ["c1"]},
            {"id": "q2", "question": "What is B?", "category": "def", "relevant_chunk_ids": ["c2"]},
            {"id": "q3", "question": "What is C?", "category": "def", "relevant_chunk_ids": ["c3"]},
            {"id": "q4", "question": "What is D?", "category": "def", "relevant_chunk_ids": ["c4"]},
        ]

        summary = evaluate_retriever(mock_retriever, questions, k_values=[1, 3, 5])
        self.assertEqual(summary.total_queries, 4)
        self.assertAlmostEqual(summary.mrr, 0.4375, places=4)

    # -------------------------------------------------------------------------
    # Test 8: Correct comparison between TF-IDF and Semantic results
    # -------------------------------------------------------------------------
    def test_08_comparison_tfidf_vs_semantic(self):
        # TF-IDF gets Q1 at rank 1, misses Q2 (synonym)
        tfidf_mock = MockRetriever({
            "exact query": [{"chunk_id": "chunk_1", "similarity_score": 0.8}],
            "paraphrased query": [{"chunk_id": "chunk_wrong", "similarity_score": 0.3}],
        })

        # Semantic gets Q1 at rank 1 and Q2 at rank 1
        semantic_mock = MockRetriever({
            "exact query": [{"chunk_id": "chunk_1", "similarity_score": 0.85}],
            "paraphrased query": [{"chunk_id": "chunk_2", "similarity_score": 0.75}],
        })

        questions = [
            {"id": "q1", "question": "exact query", "category": "exact_keyword", "relevant_chunk_ids": ["chunk_1"]},
            {"id": "q2", "question": "paraphrased query", "category": "paraphrased", "relevant_chunk_ids": ["chunk_2"]},
        ]

        comparison = compare_retrieval_methods(
            questions=questions,
            retrievers={"TF-IDF": tfidf_mock, "Semantic": semantic_mock},
            k_values=[1, 3, 5],
        )

        self.assertIn("TF-IDF", comparison.summaries)
        self.assertIn("Semantic", comparison.summaries)

        tfidf_sum = comparison.summaries["TF-IDF"]
        semantic_sum = comparison.summaries["Semantic"]

        # TF-IDF: 1 hit out of 2 queries -> Hit@1 = 0.5, MRR = 0.5
        self.assertAlmostEqual(tfidf_sum.hit_rates[1], 0.5)
        self.assertAlmostEqual(tfidf_sum.mrr, 0.5)

        # Semantic: 2 hits out of 2 queries -> Hit@1 = 1.0, MRR = 1.0
        self.assertAlmostEqual(semantic_sum.hit_rates[1], 1.0)
        self.assertAlmostEqual(semantic_sum.mrr, 1.0)

        # Verify overall metrics comparison table contains Hit@1, Hit@3, Hit@5, Precision@5, Recall@5, MRR
        metric_names = [row["Metric"] for row in comparison.overall_metrics_table]
        self.assertIn("Hit@1", metric_names)
        self.assertIn("Hit@3", metric_names)
        self.assertIn("Hit@5", metric_names)
        self.assertIn("Precision@5", metric_names)
        self.assertIn("Recall@5", metric_names)
        self.assertIn("MRR", metric_names)

    # -------------------------------------------------------------------------
    # Test 9: Correct category aggregation
    # -------------------------------------------------------------------------
    def test_09_category_aggregation(self):
        mock_retriever = MockRetriever({
            "q_kw_1": [{"chunk_id": "c1"}],
            "q_kw_2": [{"chunk_id": "c2"}],
            "q_para_1": [{"chunk_id": "wrong"}],
        })

        questions = [
            {"id": "q1", "question": "q_kw_1", "category": "exact_keyword", "relevant_chunk_ids": ["c1"]},
            {"id": "q2", "question": "q_kw_2", "category": "exact_keyword", "relevant_chunk_ids": ["c2"]},
            {"id": "q3", "question": "q_para_1", "category": "paraphrased", "relevant_chunk_ids": ["c3"]},
        ]

        summary = evaluate_retriever(mock_retriever, questions, k_values=[1, 3, 5])
        cat_metrics = summary.category_metrics

        self.assertIn("exact_keyword", cat_metrics)
        self.assertIn("paraphrased", cat_metrics)

        # exact_keyword: 2/2 hits -> Hit@1 = 1.0, MRR = 1.0
        self.assertEqual(cat_metrics["exact_keyword"].query_count, 2)
        self.assertAlmostEqual(cat_metrics["exact_keyword"].hit_rates[1], 1.0)
        self.assertAlmostEqual(cat_metrics["exact_keyword"].mrr, 1.0)

        # paraphrased: 0/1 hits -> Hit@1 = 0.0, MRR = 0.0
        self.assertEqual(cat_metrics["paraphrased"].query_count, 1)
        self.assertAlmostEqual(cat_metrics["paraphrased"].hit_rates[1], 0.0)
        self.assertAlmostEqual(cat_metrics["paraphrased"].mrr, 0.0)

    # -------------------------------------------------------------------------
    # Test 10: Correct handling of multiple relevant chunks
    # -------------------------------------------------------------------------
    def test_10_multiple_relevant_chunks(self):
        # Query has 3 ground-truth relevant chunks
        mock_retriever = MockRetriever({
            "multi-part query": [
                {"chunk_id": "doc_1"},
                {"chunk_id": "doc_2"},
                {"chunk_id": "other_3"},
                {"chunk_id": "doc_3"},
                {"chunk_id": "other_5"},
            ]
        })

        questions = [{
            "id": "qm",
            "question": "multi-part query",
            "category": "factual",
            "relevant_chunk_ids": ["doc_1", "doc_2", "doc_3"],
        }]

        summary = evaluate_retriever(mock_retriever, questions, k_values=[1, 2, 3, 5])

        # At K=1: retrieved [doc_1] -> 1/3 recall, 1/1 precision, hit=1
        self.assertAlmostEqual(summary.recall_at_k[1], 1.0 / 3.0)
        self.assertAlmostEqual(summary.precision_at_k[1], 1.0)
        self.assertEqual(summary.hit_rates[1], 1.0)

        # At K=2: retrieved [doc_1, doc_2] -> 2/3 recall, 2/2 precision
        self.assertAlmostEqual(summary.recall_at_k[2], 2.0 / 3.0)
        self.assertAlmostEqual(summary.precision_at_k[2], 1.0)

        # At K=3: retrieved [doc_1, doc_2, other_3] -> 2/3 recall, 2/3 precision
        self.assertAlmostEqual(summary.recall_at_k[3], 2.0 / 3.0)
        self.assertAlmostEqual(summary.precision_at_k[3], 2.0 / 3.0)

        # At K=5: retrieved all 3 docs -> 3/3 recall = 1.0, 3/5 precision = 0.6
        self.assertAlmostEqual(summary.recall_at_k[5], 1.0)
        self.assertAlmostEqual(summary.precision_at_k[5], 0.6)

    # -------------------------------------------------------------------------
    # Test 11: Correct handling of empty retrieval results
    # -------------------------------------------------------------------------
    def test_11_empty_retrieval_results(self):
        # Retriever returns empty list [] for all queries
        empty_retriever = MockRetriever({})

        questions = [
            {"id": "q1", "question": "Unknown term?", "category": "general", "relevant_chunk_ids": ["c1"]},
        ]

        summary = evaluate_retriever(empty_retriever, questions, k_values=[1, 3, 5])

        self.assertEqual(summary.total_queries, 1)
        self.assertEqual(summary.hit_rates[1], 0.0)
        self.assertEqual(summary.hit_rates[3], 0.0)
        self.assertEqual(summary.hit_rates[5], 0.0)
        self.assertEqual(summary.precision_at_k[5], 0.0)
        self.assertEqual(summary.recall_at_k[5], 0.0)
        self.assertEqual(summary.mrr, 0.0)

        # Verify query-level evaluation object
        q_res = summary.query_results[0]
        self.assertEqual(q_res.hit, 0)
        self.assertEqual(q_res.precision, 0.0)
        self.assertEqual(q_res.recall, 0.0)
        self.assertEqual(q_res.reciprocal_rank, 0.0)
        self.assertEqual(q_res.retrieved_chunk_ids, [])

    # -------------------------------------------------------------------------
    # Test 12: Correct handling of an empty evaluation dataset
    # -------------------------------------------------------------------------
    def test_12_empty_evaluation_dataset(self):
        mock_retriever = MockRetriever({"dummy": [{"chunk_id": "c1"}]})

        summary = evaluate_retriever(mock_retriever, questions=[], k_values=[1, 3, 5])

        self.assertEqual(summary.total_queries, 0)
        self.assertEqual(summary.mrr, 0.0)
        self.assertEqual(summary.hit_rates[1], 0.0)
        self.assertEqual(summary.hit_rates[3], 0.0)
        self.assertEqual(summary.hit_rates[5], 0.0)
        self.assertEqual(summary.precision_at_k[5], 0.0)
        self.assertEqual(summary.recall_at_k[5], 0.0)
        self.assertEqual(summary.category_metrics, {})
        self.assertEqual(summary.query_results, [])

    # -------------------------------------------------------------------------
    # Additional Test: Dataset Loading & Alignment Validation
    # -------------------------------------------------------------------------
    def test_13_dataset_loader_and_alignment(self):
        # Load the real evaluation dataset
        dataset_path = Path("evaluation/qa_dataset.json")
        self.assertTrue(dataset_path.is_file(), "qa_dataset.json should exist")

        questions = load_evaluation_dataset(dataset_path)
        self.assertGreaterEqual(len(questions), 10)

        for q in questions:
            self.assertIn("id", q)
            self.assertIn("question", q)
            self.assertIn("relevant_chunk_ids", q)
            self.assertIn("category", q)
            self.assertIsInstance(q["relevant_chunk_ids"], list)
            self.assertGreater(len(q["relevant_chunk_ids"]), 0)

        # Alignment check against sample chunks
        chunks = [
            {"chunk_id": "page_1_chunk_1"},
            {"chunk_id": "page_1_chunk_2"},
            {"chunk_id": "page_1_chunk_3"},
            {"chunk_id": "page_2_chunk_1"},
            {"chunk_id": "page_2_chunk_2"},
            {"chunk_id": "page_3_chunk_1"},
            {"chunk_id": "page_3_chunk_2"},
        ]
        val = validate_dataset_against_chunks(questions, chunks)
        self.assertTrue(val["is_fully_aligned"])
        self.assertEqual(val["missing_chunks_count"], 0)
        self.assertEqual(val["aligned_questions_count"], len(questions))

    # -------------------------------------------------------------------------
    # Additional Test: CSV Export for Results and Summary
    # -------------------------------------------------------------------------
    def test_14_csv_export(self):
        mock_retriever = MockRetriever({
            "What is X?": [{"chunk_id": "c1", "similarity_score": 0.9}],
        })
        questions = [
            {"id": "q1", "question": "What is X?", "category": "def", "relevant_chunk_ids": ["c1"]},
        ]
        summary = evaluate_retriever(mock_retriever, questions, k_values=[1, 3, 5])

        # Test results CSV export
        csv_results = export_results_to_csv(summary.query_results)
        self.assertIn("question_id,question,category,method,k,hit,precision,recall,reciprocal_rank", csv_results)
        self.assertIn("q1,What is X?,def,MockRetriever", csv_results)

        # Test summary CSV export
        csv_summary = export_summary_to_csv({"Mock": summary})
        self.assertIn("method,total_queries,hit@1,hit@3,hit@5,precision@5,recall@5,mrr", csv_summary)
        self.assertIn("Mock,1,1.0,1.0,1.0", csv_summary)


if __name__ == "__main__":
    unittest.main()
