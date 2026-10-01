"""Automated Test Suite for Phase 7: Hybrid Retrieval.

Covers:
- Test 1: Hybrid retriever initializes correctly.
- Test 2: TF-IDF and semantic candidates are combined correctly.
- Test 3: Candidate union contains candidates unique to either retriever.
- Test 4: Score normalization works correctly.
- Test 5: Normalization handles equal scores without division by zero.
- Test 6: Hybrid weighted score is calculated correctly.
- Test 7: α = 0 produces semantic-only ranking behavior.
- Test 8: α = 1 produces TF-IDF-only ranking behavior.
- Test 9: Default α = 0.5 works correctly.
- Test 10: Invalid alpha values are rejected.
- Test 11: Top-K is respected.
- Test 12: Metadata is preserved.
- Test 13: Missing scores are handled correctly.
- Test 14: Hybrid results are deterministic.
- Test 15: Functional interface retrieve_hybrid_chunks works correctly.
- Test 16: Integration with ExtractiveQAEngine produces grounded answer.
- Test 17: Query with empty / punctuation-only text handles gracefully.
- Test 18: Score thresholding filtering works correctly.
"""

import unittest
from typing import Any

from src.hybrid_retriever import (
    HybridRetriever,
    retrieve_hybrid_chunks,
    HybridRetrievalError,
)
from src.retriever import TFIDFRetriever
from src.semantic_retriever import SemanticRetriever, get_sentence_transformer, DEFAULT_MODEL_NAME
from src.qa_engine import ExtractiveQAEngine


class MockEngine:
    """Deterministic mock retriever for testing candidate fusion, weights, and edge cases."""

    def __init__(self, mapping: dict[str, list[dict[str, Any]]]) -> None:
        self.mapping = mapping

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict[str, Any]]:
        candidates = self.mapping.get(query, [])
        filtered = [c for c in candidates if float(c.get("similarity_score", 0.0)) >= min_similarity]
        return filtered[:top_k]


class TestPhase7HybridRetriever(unittest.TestCase):
    """Test suite covering all requirements of Phase 7 Hybrid Retrieval."""

    def setUp(self):
        """Set up standard fixtures and sample chunks."""
        self.sample_chunks = [
            {
                "chunk_id": "chunk_A",
                "source": "Handbook.pdf",
                "page": 1,
                "original_text": "Natural Language Processing enables computers to understand human language.",
                "cleaned_text": "natural language processing enables computers understand human language",
                "char_count": 75,
                "word_count": 9,
            },
            {
                "chunk_id": "chunk_B",
                "source": "Handbook.pdf",
                "page": 1,
                "original_text": "Attendance requirements stipulate a minimum attendance threshold of 75 percent.",
                "cleaned_text": "attendance requirements stipulate minimum attendance threshold 75 percent",
                "char_count": 80,
                "word_count": 9,
            },
            {
                "chunk_id": "chunk_C",
                "source": "Handbook.pdf",
                "page": 2,
                "original_text": "Tuition fees must be paid before semester commencement through the university portal.",
                "cleaned_text": "tuition fees paid semester commencement university portal",
                "char_count": 85,
                "word_count": 10,
            },
            {
                "chunk_id": "chunk_D",
                "source": "Handbook.pdf",
                "page": 2,
                "original_text": "Machine learning algorithms learn representations and patterns directly from raw data.",
                "cleaned_text": "machine learning algorithms learn representations patterns directly raw data",
                "char_count": 86,
                "word_count": 10,
            },
        ]

    # -------------------------------------------------------------------------
    # Test 1: Hybrid retriever initializes correctly
    # -------------------------------------------------------------------------
    def test_01_initialization(self):
        mock_tfidf = MockEngine({})
        mock_semantic = MockEngine({})

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.6,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )

        self.assertEqual(retriever.alpha, 0.6)
        self.assertEqual(len(retriever.chunks), 4)
        self.assertIs(retriever.tfidf_retriever, mock_tfidf)
        self.assertIs(retriever.semantic_retriever, mock_semantic)

    # -------------------------------------------------------------------------
    # Test 2: TF-IDF and semantic candidates are combined correctly
    # -------------------------------------------------------------------------
    def test_02_candidate_combination(self):
        mock_tfidf = MockEngine({
            "query": [
                {"chunk_id": "chunk_A", "similarity_score": 0.8, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.4, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "query": [
                {"chunk_id": "chunk_B", "similarity_score": 0.7, "page": 1, "original_text": "B"},
                {"chunk_id": "chunk_C", "similarity_score": 0.5, "page": 2, "original_text": "C"},
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("query", top_k=5, min_similarity=0.0)

        result_ids = {r["chunk_id"] for r in results}
        self.assertIn("chunk_A", result_ids)
        self.assertIn("chunk_B", result_ids)
        self.assertIn("chunk_C", result_ids)

    # -------------------------------------------------------------------------
    # Test 3: Candidate union contains candidates unique to either retriever
    # -------------------------------------------------------------------------
    def test_03_candidate_union_uniqueness(self):
        mock_tfidf = MockEngine({
            "query": [
                {"chunk_id": "chunk_A", "similarity_score": 0.9, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.6, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "query": [
                {"chunk_id": "chunk_C", "similarity_score": 0.85, "page": 2, "original_text": "C"},
                {"chunk_id": "chunk_D", "similarity_score": 0.50, "page": 2, "original_text": "D"},
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("query", top_k=10, min_similarity=0.0)

        # Union should contain all 4 candidates: A (TF-IDF only), B (TF-IDF only), C (Semantic only), D (Semantic only)
        result_ids = [r["chunk_id"] for r in results]
        self.assertEqual(len(result_ids), 4)
        self.assertEqual(set(result_ids), {"chunk_A", "chunk_B", "chunk_C", "chunk_D"})

    # -------------------------------------------------------------------------
    # Test 4: Score normalization works correctly
    # -------------------------------------------------------------------------
    def test_04_score_normalization(self):
        # Sequence input
        raw_list = [10.0, 20.0, 30.0]
        norm_list = HybridRetriever.normalize_scores(raw_list)
        self.assertAlmostEqual(norm_list[0], 0.0)
        self.assertAlmostEqual(norm_list[1], 0.5)
        self.assertAlmostEqual(norm_list[2], 1.0)

        # Dict input
        raw_dict = {"A": 0.2, "B": 0.5, "C": 0.8}
        norm_dict = HybridRetriever.normalize_scores(raw_dict)
        self.assertAlmostEqual(norm_dict["A"], 0.0)
        self.assertAlmostEqual(norm_dict["B"], 0.5)
        self.assertAlmostEqual(norm_dict["C"], 1.0)

    # -------------------------------------------------------------------------
    # Test 5: Normalization handles equal scores without division by zero
    # -------------------------------------------------------------------------
    def test_05_equal_scores_division_by_zero(self):
        # All equal non-zero scores -> 1.0
        norm_equal = HybridRetriever.normalize_scores([0.7, 0.7, 0.7])
        self.assertEqual(norm_equal, [1.0, 1.0, 1.0])

        norm_dict_equal = HybridRetriever.normalize_scores({"A": 0.5, "B": 0.5})
        self.assertEqual(norm_dict_equal, {"A": 1.0, "B": 1.0})

        # All equal zero scores -> 0.0
        norm_zeros = HybridRetriever.normalize_scores([0.0, 0.0])
        self.assertEqual(norm_zeros, [0.0, 0.0])

        norm_dict_zeros = HybridRetriever.normalize_scores({"A": 0.0, "B": 0.0})
        self.assertEqual(norm_dict_zeros, {"A": 0.0, "B": 0.0})

        # Empty
        self.assertEqual(HybridRetriever.normalize_scores([]), [])
        self.assertEqual(HybridRetriever.normalize_scores({}), {})

    # -------------------------------------------------------------------------
    # Test 6: Hybrid weighted score is calculated correctly
    # -------------------------------------------------------------------------
    def test_06_hybrid_score_calculation(self):
        # Candidate A: tfidf 0.8, semantic 0.2
        # Candidate B: tfidf 0.4, semantic 0.6
        mock_tfidf = MockEngine({
            "q": [
                {"chunk_id": "chunk_A", "similarity_score": 0.8, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.4, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "q": [
                {"chunk_id": "chunk_B", "similarity_score": 0.6, "page": 1, "original_text": "B"},
                {"chunk_id": "chunk_A", "similarity_score": 0.2, "page": 1, "original_text": "A"},
            ]
        })

        # With alpha = 0.5:
        # tfidf: A=0.8 (norm 1.0), B=0.4 (norm 0.0)
        # semantic: B=0.6 (norm 1.0), A=0.2 (norm 0.0)
        # Hybrid A = 0.5*1.0 + 0.5*0.0 = 0.50
        # Hybrid B = 0.5*0.0 + 0.5*1.0 = 0.50
        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("q", top_k=2, min_similarity=0.0)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertAlmostEqual(r["hybrid_score"], 0.5, places=3)
            self.assertAlmostEqual(r["similarity_score"], 0.5, places=3)

    # -------------------------------------------------------------------------
    # Test 7: α = 0 produces semantic-only ranking behavior
    # -------------------------------------------------------------------------
    def test_07_alpha_zero_semantic_only(self):
        # Chunk A is #1 in TF-IDF, but #2 in Semantic
        # Chunk B is #2 in TF-IDF, but #1 in Semantic
        mock_tfidf = MockEngine({
            "q": [
                {"chunk_id": "chunk_A", "similarity_score": 0.9, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.3, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "q": [
                {"chunk_id": "chunk_B", "similarity_score": 0.95, "page": 1, "original_text": "B"},
                {"chunk_id": "chunk_A", "similarity_score": 0.20, "page": 1, "original_text": "A"},
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.0,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("q", top_k=2, min_similarity=0.0)

        # When alpha = 0.0, ranking must strictly follow semantic relevance (chunk_B rank 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_B")
        self.assertEqual(results[1]["chunk_id"], "chunk_A")
        self.assertAlmostEqual(results[0]["hybrid_score"], 1.0)
        self.assertAlmostEqual(results[1]["hybrid_score"], 0.0)

    # -------------------------------------------------------------------------
    # Test 8: α = 1 produces TF-IDF-only ranking behavior
    # -------------------------------------------------------------------------
    def test_08_alpha_one_tfidf_only(self):
        mock_tfidf = MockEngine({
            "q": [
                {"chunk_id": "chunk_A", "similarity_score": 0.9, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.3, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "q": [
                {"chunk_id": "chunk_B", "similarity_score": 0.95, "page": 1, "original_text": "B"},
                {"chunk_id": "chunk_A", "similarity_score": 0.20, "page": 1, "original_text": "A"},
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=1.0,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("q", top_k=2, min_similarity=0.0)

        # When alpha = 1.0, ranking must strictly follow TF-IDF relevance (chunk_A rank 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_A")
        self.assertEqual(results[1]["chunk_id"], "chunk_B")
        self.assertAlmostEqual(results[0]["hybrid_score"], 1.0)
        self.assertAlmostEqual(results[1]["hybrid_score"], 0.0)

    # -------------------------------------------------------------------------
    # Test 9: Default α = 0.5 works correctly
    # -------------------------------------------------------------------------
    def test_09_default_alpha_value(self):
        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            tfidf_retriever=MockEngine({}),
            semantic_retriever=MockEngine({}),
        )
        self.assertEqual(retriever.alpha, 0.5)

    # -------------------------------------------------------------------------
    # Test 10: Invalid alpha values are rejected
    # -------------------------------------------------------------------------
    def test_10_invalid_alpha_rejection(self):
        mock_t = MockEngine({})
        mock_s = MockEngine({})

        with self.assertRaises(ValueError):
            HybridRetriever(chunks=self.sample_chunks, alpha=-0.1, tfidf_retriever=mock_t, semantic_retriever=mock_s)

        with self.assertRaises(ValueError):
            HybridRetriever(chunks=self.sample_chunks, alpha=1.01, tfidf_retriever=mock_t, semantic_retriever=mock_s)

        with self.assertRaises(ValueError):
            HybridRetriever(chunks=self.sample_chunks, alpha="invalid", tfidf_retriever=mock_t, semantic_retriever=mock_s)

        # Also in search() override
        valid_retriever = HybridRetriever(chunks=self.sample_chunks, alpha=0.5, tfidf_retriever=mock_t, semantic_retriever=mock_s)
        with self.assertRaises(ValueError):
            valid_retriever.search("q", alpha=1.5)

    # -------------------------------------------------------------------------
    # Test 11: Top-K is respected
    # -------------------------------------------------------------------------
    def test_11_top_k_bounding(self):
        mock_tfidf = MockEngine({
            "q": [
                {"chunk_id": f"c_{i}", "similarity_score": 0.8 - i * 0.1, "page": 1, "original_text": f"text {i}"}
                for i in range(10)
            ]
        })
        mock_semantic = MockEngine({
            "q": [
                {"chunk_id": f"c_{i}", "similarity_score": 0.7 - i * 0.05, "page": 1, "original_text": f"text {i}"}
                for i in range(10)
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks * 3,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )

        results_k2 = retriever.search("q", top_k=2, min_similarity=0.0)
        self.assertEqual(len(results_k2), 2)

        results_k5 = retriever.search("q", top_k=5, min_similarity=0.0)
        self.assertEqual(len(results_k5), 5)

        # Non-positive K returns empty
        self.assertEqual(retriever.search("q", top_k=0), [])
        self.assertEqual(retriever.search("q", top_k=-1), [])

    # -------------------------------------------------------------------------
    # Test 12: Metadata is preserved
    # -------------------------------------------------------------------------
    def test_12_metadata_preservation(self):
        mock_tfidf = MockEngine({
            "query": [{"chunk_id": "chunk_A", "similarity_score": 0.85, "page": 1, "source": "Handbook.pdf", "original_text": "NLP text", "cleaned_text": "nlp text"}]
        })
        mock_semantic = MockEngine({
            "query": [{"chunk_id": "chunk_A", "similarity_score": 0.75, "page": 1, "source": "Handbook.pdf", "original_text": "NLP text", "cleaned_text": "nlp text"}]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("query", top_k=1, min_similarity=0.0)
        self.assertEqual(len(results), 1)
        res = results[0]

        # Phase 2 metadata
        self.assertEqual(res["chunk_id"], "chunk_A")
        self.assertEqual(res["source"], "Handbook.pdf")
        self.assertEqual(res["page"], 1)
        self.assertEqual(res["original_text"], "NLP text")
        self.assertEqual(res["cleaned_text"], "nlp text")

        # Phase 7 explainability scores
        self.assertIn("tfidf_score", res)
        self.assertIn("semantic_score", res)
        self.assertIn("normalized_tfidf_score", res)
        self.assertIn("normalized_semantic_score", res)
        self.assertIn("hybrid_score", res)
        self.assertIn("similarity_score", res)
        self.assertEqual(res["tfidf_score"], 0.85)
        self.assertEqual(res["semantic_score"], 0.75)

    # -------------------------------------------------------------------------
    # Test 13: Missing scores are handled correctly
    # -------------------------------------------------------------------------
    def test_13_missing_scores_handled_correctly(self):
        # Chunk A is only in TF-IDF; Chunk B is only in Semantic
        mock_tfidf = MockEngine({
            "query": [{"chunk_id": "chunk_A", "similarity_score": 0.8, "page": 1, "original_text": "A"}]
        })
        mock_semantic = MockEngine({
            "query": [{"chunk_id": "chunk_B", "similarity_score": 0.8, "page": 1, "original_text": "B"}]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        results = retriever.search("query", top_k=5, min_similarity=0.0)

        self.assertEqual(len(results), 2)
        res_map = {r["chunk_id"]: r for r in results}

        # Chunk A was missing from semantic: raw semantic score is 0.0
        self.assertEqual(res_map["chunk_A"]["semantic_score"], 0.0)
        self.assertEqual(res_map["chunk_A"]["tfidf_score"], 0.8)

        # Chunk B was missing from tfidf: raw tfidf score is 0.0
        self.assertEqual(res_map["chunk_B"]["tfidf_score"], 0.0)
        self.assertEqual(res_map["chunk_B"]["semantic_score"], 0.8)

    # -------------------------------------------------------------------------
    # Test 14: Hybrid results are deterministic
    # -------------------------------------------------------------------------
    def test_14_results_are_deterministic(self):
        mock_tfidf = MockEngine({
            "deterministic_q": [
                {"chunk_id": "chunk_A", "similarity_score": 0.7, "page": 1, "original_text": "A"},
                {"chunk_id": "chunk_B", "similarity_score": 0.5, "page": 1, "original_text": "B"},
            ]
        })
        mock_semantic = MockEngine({
            "deterministic_q": [
                {"chunk_id": "chunk_B", "similarity_score": 0.8, "page": 1, "original_text": "B"},
                {"chunk_id": "chunk_C", "similarity_score": 0.6, "page": 2, "original_text": "C"},
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )

        res1 = retriever.search("deterministic_q", top_k=3, min_similarity=0.0)
        res2 = retriever.search("deterministic_q", top_k=3, min_similarity=0.0)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1["chunk_id"], r2["chunk_id"])
            self.assertEqual(r1["hybrid_score"], r2["hybrid_score"])
            self.assertEqual(r1["similarity_score"], r2["similarity_score"])

    # -------------------------------------------------------------------------
    # Test 15: Functional interface retrieve_hybrid_chunks works correctly
    # -------------------------------------------------------------------------
    def test_15_functional_interface(self):
        mock_tfidf = MockEngine({
            "func_q": [{"chunk_id": "chunk_A", "similarity_score": 0.9, "page": 1, "original_text": "A"}]
        })
        mock_semantic = MockEngine({
            "func_q": [{"chunk_id": "chunk_A", "similarity_score": 0.8, "page": 1, "original_text": "A"}]
        })

        results = retrieve_hybrid_chunks(
            chunks=self.sample_chunks,
            query="func_q",
            top_k=1,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_A")

    # -------------------------------------------------------------------------
    # Test 16: Hybrid results feed directly into ExtractiveQAEngine
    # -------------------------------------------------------------------------
    def test_16_qa_engine_integration(self):
        mock_tfidf = MockEngine({
            "What is the minimum attendance?": [
                {
                    "chunk_id": "chunk_B",
                    "similarity_score": 0.85,
                    "page": 1,
                    "source": "Handbook.pdf",
                    "original_text": "Attendance requirements stipulate a minimum attendance threshold of 75 percent.",
                    "cleaned_text": "attendance requirements stipulate minimum attendance threshold 75 percent",
                }
            ]
        })
        mock_semantic = MockEngine({
            "What is the minimum attendance?": [
                {
                    "chunk_id": "chunk_B",
                    "similarity_score": 0.90,
                    "page": 1,
                    "source": "Handbook.pdf",
                    "original_text": "Attendance requirements stipulate a minimum attendance threshold of 75 percent.",
                    "cleaned_text": "attendance requirements stipulate minimum attendance threshold 75 percent",
                }
            ]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            alpha=0.5,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        retrieved_passages = retriever.search("What is the minimum attendance?", top_k=3)

        qa_engine = ExtractiveQAEngine()
        qa_result = qa_engine.answer(
            question="What is the minimum attendance?",
            retrieved_chunks=retrieved_passages,
        )

        self.assertTrue(qa_result["found"])
        self.assertIn("75 percent", qa_result["answer"])
        self.assertEqual(qa_result["chunk_id"], "chunk_B")
        self.assertEqual(qa_result["page"], 1)

    # -------------------------------------------------------------------------
    # Test 17: Empty / punctuation-only handling
    # -------------------------------------------------------------------------
    def test_17_empty_query_handling(self):
        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            tfidf_retriever=MockEngine({}),
            semantic_retriever=MockEngine({}),
        )
        self.assertEqual(retriever.search(""), [])
        self.assertEqual(retriever.search("   "), [])
        self.assertEqual(retriever.search("???!!!"), [])

    # -------------------------------------------------------------------------
    # Test 18: Score threshold filtering
    # -------------------------------------------------------------------------
    def test_18_threshold_filtering(self):
        mock_tfidf = MockEngine({
            "q": [{"chunk_id": "chunk_A", "similarity_score": 0.05, "page": 1, "original_text": "A"}]
        })
        mock_semantic = MockEngine({
            "q": [{"chunk_id": "chunk_A", "similarity_score": 0.05, "page": 1, "original_text": "A"}]
        })

        retriever = HybridRetriever(
            chunks=self.sample_chunks,
            tfidf_retriever=mock_tfidf,
            semantic_retriever=mock_semantic,
        )
        # min_similarity = 0.50 -> chunk with low similarity dropped
        results = retriever.search("q", top_k=5, min_similarity=0.50)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
