"""Automated Test Suite for Phase 5: Semantic Retrieval Enhancement.

Covers:
- Test 1: Model initialization
- Test 2: Embedding generation for document chunks
- Test 3: Query embedding generation
- Test 4: Semantic relevance ranking
- Test 5: Top-K bounding
- Test 6: Minimum similarity threshold filtering
- Test 7: Chunk metadata preservation
- Test 8: Paraphrased query handling (different wording, same semantic meaning)
- Test 9: Safe handling of empty / invalid inputs
- Test 10: Backward compatibility & pipeline integration with ExtractiveQAEngine
"""

import unittest
import numpy as np

from src.semantic_retriever import (
    SemanticRetriever,
    retrieve_semantic_chunks,
    get_sentence_transformer,
    DEFAULT_MODEL_NAME,
)
from src.qa_engine import ExtractiveQAEngine


class TestPhase5SemanticRetriever(unittest.TestCase):
    """Test suite covering all requirements of Phase 5 Semantic Retrieval."""

    @classmethod
    def setUpClass(cls):
        """Pre-load SentenceTransformer model once for the test suite."""
        cls.model = get_sentence_transformer(DEFAULT_MODEL_NAME)

    def setUp(self):
        """Set up realistic sample document chunks."""
        self.sample_chunks = [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "Student_Handbook.pdf",
                "page": 1,
                "original_text": "Students are required to maintain a minimum attendance of 75 percent across all enrolled courses.",
                "cleaned_text": "students required maintain minimum attendance 75 percent across enrolled courses",
                "char_count": 98,
                "word_count": 14,
            },
            {
                "chunk_id": "page_1_chunk_2",
                "source": "Student_Handbook.pdf",
                "page": 1,
                "original_text": "Tuition fees must be paid before the start of each semester via the university financial portal.",
                "cleaned_text": "tuition fees paid start semester via university financial portal",
                "char_count": 96,
                "word_count": 14,
            },
            {
                "chunk_id": "page_2_chunk_1",
                "source": "Student_Handbook.pdf",
                "page": 2,
                "original_text": "The library remains accessible twenty-four hours a day for registered university scholars.",
                "cleaned_text": "library remains accessible twenty four hours day registered university scholars",
                "char_count": 91,
                "word_count": 12,
            },
            {
                "chunk_id": "page_2_chunk_2",
                "source": "Student_Handbook.pdf",
                "page": 2,
                "original_text": "Final course grades are officially published within two weeks following examination week.",
                "cleaned_text": "final course grades officially published within two weeks following examination week",
                "char_count": 89,
                "word_count": 11,
            },
        ]
        self.retriever = SemanticRetriever(self.sample_chunks, model=self.model)

    def test_01_model_initialization(self):
        """Test 1: Verify that the semantic retriever initializes properly."""
        retriever = SemanticRetriever(self.sample_chunks, model=self.model)
        self.assertIsNotNone(retriever.model)
        self.assertTrue(retriever.is_indexed)
        self.assertEqual(len(retriever.chunks), 4)
        self.assertEqual(len(retriever.corpus), 4)

    def test_02_embedding_generation(self):
        """Test 2: Verify embeddings are generated and cached for all document chunks."""
        self.assertIsNotNone(self.retriever.chunk_embeddings)
        self.assertIsInstance(self.retriever.chunk_embeddings, np.ndarray)
        # Should have shape (num_chunks, embedding_dim)
        self.assertEqual(self.retriever.chunk_embeddings.shape[0], len(self.sample_chunks))
        # all-MiniLM-L6-v2 embedding dimension is 384
        self.assertEqual(self.retriever.chunk_embeddings.shape[1], 384)

        # Verify unit normalization: norm of each vector should be approx 1.0
        norms = np.linalg.norm(self.retriever.chunk_embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_03_query_embedding(self):
        """Test 3: Verify a user query can be converted into an embedding."""
        query = "How much attendance is required?"
        query_vec = self.retriever.encode_query(query)
        self.assertIsNotNone(query_vec)
        self.assertIsInstance(query_vec, np.ndarray)
        self.assertEqual(query_vec.shape, (384,))
        # Unit normalized
        self.assertAlmostEqual(float(np.linalg.norm(query_vec)), 1.0, places=4)

    def test_04_retrieval_ranking(self):
        """Test 4: Verify semantically relevant chunks receive higher similarity scores."""
        query = "How much attendance is mandatory for students?"
        results = self.retriever.search(query, top_k=4, min_similarity=0.0)

        self.assertGreater(len(results), 0)
        top_match = results[0]
        # Chunk 1 describes attendance requirements
        self.assertEqual(top_match["chunk_id"], "page_1_chunk_1")
        self.assertGreater(top_match["similarity_score"], 0.40)

        # Check descending sort order
        scores = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_05_top_k(self):
        """Test 5: Verify top_k bounds the number of returned results."""
        results = self.retriever.search("university regulations", top_k=2, min_similarity=0.0)
        self.assertLessEqual(len(results), 2)

        results_1 = self.retriever.search("university regulations", top_k=1, min_similarity=0.0)
        self.assertEqual(len(results_1), 1)

    def test_06_threshold(self):
        """Test 6: Verify results below minimum similarity threshold are excluded."""
        # Unrelated query with high threshold should yield 0 results
        results = self.retriever.search(
            "quantum mechanical wave function superposition",
            top_k=5,
            min_similarity=0.60,
        )
        self.assertEqual(len(results), 0)

        # Relevant query with standard threshold should return matches
        results_valid = self.retriever.search(
            "paying semester tuition",
            top_k=5,
            min_similarity=0.30,
        )
        self.assertGreater(len(results_valid), 0)
        for r in results_valid:
            self.assertGreaterEqual(r["similarity_score"], 0.30)

    def test_07_metadata_preservation(self):
        """Test 7: Verify retrieved results preserve all original chunk metadata."""
        results = self.retriever.search("attendance requirement", top_k=1)
        self.assertEqual(len(results), 1)
        item = results[0]

        # Must preserve chunk_id, page, source, original_text, cleaned_text
        self.assertIn("chunk_id", item)
        self.assertEqual(item["chunk_id"], "page_1_chunk_1")
        self.assertIn("page", item)
        self.assertEqual(item["page"], 1)
        self.assertIn("source", item)
        self.assertEqual(item["source"], "Student_Handbook.pdf")
        self.assertIn("original_text", item)
        self.assertIn("cleaned_text", item)
        self.assertIn("similarity_score", item)
        self.assertIsInstance(item["similarity_score"], float)

    def test_08_paraphrased_query(self):
        """Test 8: Verify semantic matching when question and passage use different wording.

        Question: "What percentage of attendance must students maintain?"
        Passage: "Students are required to maintain a minimum attendance of 75 percent across all enrolled courses."
        Notice: 'percentage' vs 'percent', 'maintain' vs 'required to maintain', etc.
        """
        question = "What percentage of attendance must students maintain?"
        results = self.retriever.search(question, top_k=3, min_similarity=0.10)

        self.assertGreater(len(results), 0)
        best_match = results[0]
        self.assertEqual(best_match["chunk_id"], "page_1_chunk_1")
        self.assertIn("75 percent", best_match["original_text"])
        # Semantic similarity should be strong (> 0.50)
        self.assertGreater(best_match["similarity_score"], 0.50)

    def test_09_empty_and_edge_inputs(self):
        """Test 9: Verify safe behavior for empty / whitespace / punctuation query and empty chunks."""
        # Empty query
        self.assertEqual(self.retriever.search("", top_k=3), [])
        self.assertEqual(self.retriever.search("   ", top_k=3), [])
        self.assertEqual(self.retriever.search("???!!!", top_k=3), [])
        self.assertEqual(self.retriever.search("attendance", top_k=0), [])

        # Empty retriever
        empty_retriever = SemanticRetriever([], model=self.model)
        self.assertEqual(empty_retriever.search("attendance", top_k=3), [])
        self.assertIsNone(empty_retriever.chunk_embeddings)

        # Chunks with blank text
        blank_chunks = [{"chunk_id": "c1", "source": "f.pdf", "page": 1, "original_text": ""}]
        blank_retriever = SemanticRetriever(blank_chunks, model=self.model)
        self.assertEqual(blank_retriever.search("attendance"), [])

    def test_10_integration_with_qa_engine(self):
        """Test 10: Verify retrieved semantic chunks seamlessly integrate into ExtractiveQAEngine."""
        qa_engine = ExtractiveQAEngine()
        question = "What percentage of attendance must students maintain?"

        # 1. Retrieve top passages semantically
        retrieved_chunks = self.retriever.search(question, top_k=2, min_similarity=0.10)
        self.assertGreater(len(retrieved_chunks), 0)

        # 2. Feed into existing extractive QA engine
        qa_result = qa_engine.answer(question, retrieved_chunks)

        # 3. Verify grounded extractive answer
        self.assertTrue(qa_result["found"])
        self.assertIsNotNone(qa_result["answer"])
        self.assertIn("75 percent", qa_result["answer"])
        self.assertEqual(qa_result["page"], 1)
        self.assertEqual(qa_result["source"], "Student_Handbook.pdf")
        self.assertEqual(qa_result["chunk_id"], "page_1_chunk_1")
        self.assertGreater(qa_result["similarity_score"], 0.50)


if __name__ == "__main__":
    unittest.main()
