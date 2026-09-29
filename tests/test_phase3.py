"""Automated Test Suite for Phase 3: Retrieval Engine (TF-IDF + Cosine Similarity)."""

import unittest
from src.retriever import TFIDFRetriever, retrieve_relevant_chunks, RetrievalError


class TestPhase3RetrievalEngine(unittest.TestCase):
    """Test suite covering all requirements of Phase 3."""

    def setUp(self):
        """Set up a realistic collection of document chunks for testing."""
        self.sample_chunks = [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "NLP_Textbook.pdf",
                "page": 1,
                "original_text": "Natural Language Processing (NLP) is a branch of artificial intelligence focused on human-computer interaction.",
                "cleaned_text": "natural language processing nlp branch artificial intelligence focused human computer interaction",
                "char_count": 113,
                "word_count": 14,
            },
            {
                "chunk_id": "page_1_chunk_2",
                "source": "NLP_Textbook.pdf",
                "page": 1,
                "original_text": "Tokenization is the process of breaking a stream of text into words, phrases, or symbols known as tokens.",
                "cleaned_text": "tokenization process breaking stream text words phrases symbols known tokens",
                "char_count": 107,
                "word_count": 17,
            },
            {
                "chunk_id": "page_2_chunk_1",
                "source": "NLP_Textbook.pdf",
                "page": 2,
                "original_text": "Stop-word removal filters out high-frequency functional words like 'and', 'the', and 'is' from the corpus.",
                "cleaned_text": "stop word removal filters high frequency functional words like corpus",
                "char_count": 107,
                "word_count": 15,
            },
            {
                "chunk_id": "page_2_chunk_2",
                "source": "NLP_Textbook.pdf",
                "page": 2,
                "original_text": "TF-IDF stands for Term Frequency-Inverse Document Frequency and measures lexical importance across documents.",
                "cleaned_text": "tf idf stands term frequency inverse document frequency measures lexical importance across documents",
                "char_count": 110,
                "word_count": 14,
            },
            {
                "chunk_id": "page_3_chunk_1",
                "source": "NLP_Textbook.pdf",
                "page": 3,
                "original_text": "Cosine similarity computes the angular distance between two vectors in a high-dimensional space.",
                "cleaned_text": "cosine similarity computes angular distance two vectors high dimensional space",
                "char_count": 98,
                "word_count": 13,
            },
        ]
        self.retriever = TFIDFRetriever(self.sample_chunks)

    def test_1_relevant_query(self):
        """Test 1: A query containing terms from a known chunk should rank that chunk highly."""
        query = "What is tokenization and how are tokens formed?"
        results = self.retriever.search(query=query, top_k=3, min_similarity=0.05)

        self.assertGreater(len(results), 0)
        top_result = results[0]
        # Top result should specifically be chunk 2 about tokenization
        self.assertEqual(top_result["chunk_id"], "page_1_chunk_2")
        self.assertEqual(top_result["page"], 1)
        self.assertIn("Tokenization", top_result["original_text"])
        self.assertGreater(top_result["similarity_score"], 0.20)

    def test_2_ranking(self):
        """Test 2: Verify that more relevant chunks appear before less relevant chunks."""
        query = "TF-IDF term frequency measures"
        results = self.retriever.search(query=query, top_k=5, min_similarity=0.01)

        self.assertGreaterEqual(len(results), 1)
        # Top match must be chunk 4 (page_2_chunk_2)
        self.assertEqual(results[0]["chunk_id"], "page_2_chunk_2")

        # Verify strictly descending score order
        scores = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_3_top_k(self):
        """Test 3: If top_k=2 is specified, return at most 2 results even when more match."""
        # Create a corpus of 10 chunks all mentioning 'neural networks'
        ten_chunks = [
            {
                "chunk_id": f"page_{i}_chunk_1",
                "source": "DeepLearning.pdf",
                "page": i,
                "original_text": f"Chapter {i} explains neural networks and deep learning architectures.",
                "cleaned_text": f"chapter {i} explains neural networks deep learning architectures",
                "char_count": 65,
                "word_count": 9,
            }
            for i in range(1, 11)
        ]
        large_retriever = TFIDFRetriever(ten_chunks)
        results = large_retriever.search(query="neural networks deep learning", top_k=3, min_similarity=0.01)

        self.assertEqual(len(results), 3)

        # Also verify bounding when top_k > available chunks
        results_over_k = large_retriever.search(query="neural networks", top_k=20, min_similarity=0.01)
        self.assertLessEqual(len(results_over_k), 10)

    def test_4_metadata(self):
        """Test 4: Verify that page, source, chunk ID, original_text, and character counts remain intact."""
        query = "artificial intelligence NLP"
        results = self.retriever.search(query=query, top_k=1, min_similarity=0.01)

        self.assertEqual(len(results), 1)
        top = results[0]

        # Verify all metadata keys
        self.assertEqual(top["chunk_id"], "page_1_chunk_1")
        self.assertEqual(top["source"], "NLP_Textbook.pdf")
        self.assertEqual(top["page"], 1)
        self.assertIn("Natural Language Processing (NLP)", top["original_text"])
        self.assertEqual(top["char_count"], 113)
        self.assertEqual(top["word_count"], 14)
        self.assertIn("similarity_score", top)

    def test_5_similarity_score(self):
        """Test 5: Verify scores are numeric and within 0.0 <= score <= 1.0."""
        query = "cosine similarity vectors"
        results = self.retriever.search(query=query, top_k=5, min_similarity=0.0)

        for res in results:
            score = res["similarity_score"]
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_6_empty_query(self):
        """Test 6: Verify graceful handling of empty and invalid queries."""
        # Pure empty
        self.assertEqual(self.retriever.search(query=""), [])
        # Whitespace only
        self.assertEqual(self.retriever.search(query="   \t \n  "), [])
        # Punctuation only
        self.assertEqual(self.retriever.search(query="??? ... !!!"), [])
        # Top-K non-positive
        self.assertEqual(self.retriever.search(query="NLP", top_k=0), [])
        self.assertEqual(self.retriever.search(query="NLP", top_k=-1), [])

    def test_7_unrelated_query(self):
        """Test 7: Verify that low-similarity queries produce empty results when below threshold."""
        query = "What is the recipe for chocolate chip cookies in Paris?"
        results = self.retriever.search(query=query, top_k=5, min_similarity=0.10)

        # Unrelated query should yield 0 results
        self.assertEqual(results, [])

    def test_8_single_chunk(self):
        """Test 8: Verify retrieval works when only one chunk exists."""
        single_chunk = [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "Single.pdf",
                "page": 1,
                "original_text": "Unique lone paragraph discussing quantum computing algorithms.",
                "cleaned_text": "unique lone paragraph discussing quantum computing algorithms",
                "char_count": 63,
                "word_count": 7,
            }
        ]
        retriever = TFIDFRetriever(single_chunk)

        # Matching query
        res = retriever.search(query="quantum computing", top_k=5, min_similarity=0.10)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "page_1_chunk_1")

        # Non-matching query
        res_none = retriever.search(query="agriculture farming wheat", top_k=5, min_similarity=0.10)
        self.assertEqual(len(res_none), 0)

    def test_9_repeated_query(self):
        """Test 9: Verify repeated searches produce identical, deterministic results."""
        query = "How does stop-word removal work in corpus preprocessing?"
        res1 = self.retriever.search(query=query, top_k=3, min_similarity=0.10)
        res2 = self.retriever.search(query=query, top_k=3, min_similarity=0.10)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1["chunk_id"], r2["chunk_id"])
            self.assertEqual(r1["similarity_score"], r2["similarity_score"])

    def test_10_functional_api(self):
        """Verify the convenience function retrieve_relevant_chunks matches class behavior."""
        query = "angular distance cosine similarity"
        res_class = self.retriever.search(query=query, top_k=2, min_similarity=0.10)
        res_func = retrieve_relevant_chunks(self.sample_chunks, query=query, top_k=2, min_similarity=0.10)

        self.assertEqual(len(res_class), len(res_func))
        if res_class and res_func:
            self.assertEqual(res_class[0]["chunk_id"], res_func[0]["chunk_id"])
            self.assertEqual(res_class[0]["similarity_score"], res_func[0]["similarity_score"])

    def test_empty_corpus_handling(self):
        """Verify retriever handles empty or whitespace-only chunk collections gracefully."""
        empty_retriever = TFIDFRetriever([])
        self.assertEqual(empty_retriever.search("any query"), [])

        blank_retriever = TFIDFRetriever([{"chunk_id": "p1", "cleaned_text": "   "}])
        self.assertEqual(blank_retriever.search("any query"), [])


if __name__ == "__main__":
    unittest.main()
