"""Automated Test Suite for Phase 4: Question Answering (Extractive QA Engine)."""

import unittest
from src.retriever import TFIDFRetriever
from src.qa_engine import ExtractiveQAEngine, answer_question


class TestPhase4QuestionAnswering(unittest.TestCase):
    """Test suite covering all requirements of Phase 4."""

    def setUp(self):
        """Set up realistic document chunks for QA evaluation."""
        self.chunks = [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "NLP_Syllabus.pdf",
                "page": 1,
                "original_text": (
                    "Natural Language Processing is a branch of artificial intelligence that "
                    "enables computers to understand and process human language."
                ),
                "cleaned_text": "natural language processing branch artificial intelligence enables computers understand process human language",
                "char_count": 139,
                "word_count": 19,
                "similarity_score": 0.85,
            },
            {
                "chunk_id": "page_2_chunk_1",
                "source": "NLP_Syllabus.pdf",
                "page": 2,
                "original_text": (
                    "Students must maintain a minimum attendance of 75 percent. "
                    "Attendance below this requirement may result in academic restrictions. "
                    "Campus facilities are open daily."
                ),
                "cleaned_text": "students maintain minimum attendance 75 percent attendance requirement result academic restrictions campus facilities open daily",
                "char_count": 166,
                "word_count": 22,
                "similarity_score": 0.78,
            },
            {
                "chunk_id": "page_3_chunk_1",
                "source": "NLP_Syllabus.pdf",
                "page": 3,
                "original_text": (
                    "TF-IDF measures the lexical importance of a term within a document relative to a corpus. "
                    "Cosine similarity evaluates the angular alignment between two normalized vector representations."
                ),
                "cleaned_text": "tf idf measures lexical importance term document relative corpus cosine similarity evaluates angular alignment two normalized vector representations",
                "char_count": 188,
                "word_count": 24,
                "similarity_score": 0.72,
            },
        ]
        self.qa_engine = ExtractiveQAEngine()
        self.retriever = TFIDFRetriever(self.chunks)

    def test_1_direct_factual_question(self):
        """Test 1: Direct factual question answered by one sentence (Attendance minimum)."""
        question = "What is the minimum attendance requirement?"
        # Retrieve chunks using Phase 3
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)
        self.assertGreater(len(retrieved), 0)

        # Answer using Phase 4 QA
        result = self.qa_engine.answer(question, retrieved, max_sentences=1)

        self.assertTrue(result["found"])
        self.assertIsNotNone(result["answer"])
        # Expected answer must be the first sentence containing 75 percent
        expected_sentence = "Students must maintain a minimum attendance of 75 percent."
        self.assertIn(expected_sentence, result["answer"])
        # Must not contain the unrelated campus sentence
        self.assertNotIn("Campus facilities are open daily.", result["answer"])

    def test_2_question_requiring_context(self):
        """Test 2: Question requiring multi-sentence context (attendance policy)."""
        question = "What are the rules and requirements for student attendance?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        result = self.qa_engine.answer(question, retrieved, max_sentences=2)

        self.assertTrue(result["found"])
        # Should include both attendance sentences
        self.assertIn("Students must maintain a minimum attendance of 75 percent.", result["answer"])
        self.assertIn("Attendance below this requirement may result in academic restrictions.", result["answer"])
        # Should not include unrelated sentence
        self.assertNotIn("Campus facilities are open daily.", result["answer"])

    def test_3_irrelevant_question(self):
        """Test 3: Irrelevant question not covered by document returns no answer."""
        question = "What is the capital of France?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        # Phase 3 returns empty list for unrelated query
        self.assertEqual(retrieved, [])

        result = self.qa_engine.answer(question, retrieved)

        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])
        self.assertIn("No sufficiently relevant information", result["reason"])

    def test_4_empty_question(self):
        """Test 4: Graceful handling of empty, whitespace, and punctuation-only questions."""
        # Empty string
        res_empty = self.qa_engine.answer("", self.chunks)
        self.assertFalse(res_empty["found"])
        self.assertIsNone(res_empty["answer"])

        # Whitespace
        res_ws = self.qa_engine.answer("   \t  \n ", self.chunks)
        self.assertFalse(res_ws["found"])

        # Punctuation only
        res_punct = self.qa_engine.answer("??? ... !!!", self.chunks)
        self.assertFalse(res_punct["found"])

    def test_5_source_preservation(self):
        """Test 5: Verify source filename, page number, and chunk ID remain intact."""
        question = "What is Natural Language Processing?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        result = self.qa_engine.answer(question, retrieved)

        self.assertTrue(result["found"])
        self.assertEqual(result["source"], "NLP_Syllabus.pdf")
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["chunk_id"], "page_1_chunk_1")
        self.assertGreater(result["similarity_score"], 0.0)

    def test_6_evidence(self):
        """Test 6: Verify answer is derived strictly from original_text evidence."""
        question = "What is Natural Language Processing?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        result = self.qa_engine.answer(question, retrieved)

        self.assertTrue(result["found"])
        self.assertIsNotNone(result["evidence"])
        # Answer must be a substring of original_text evidence
        self.assertIn(result["answer"], result["evidence"])
        # Original text has proper casing and punctuation
        self.assertTrue(result["answer"].startswith("Natural Language Processing"))

    def test_7_grounding(self):
        """Test 7: Ensure QA engine never returns information that does not exist in retrieved evidence."""
        question = "What is quantum entanglement and teleportation?"
        # Empty retrieval
        result = self.qa_engine.answer(question, [])
        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])
        # Never hallucinates
        self.assertNotEqual(result["answer"], "Quantum entanglement is...")

    def test_8_retrieval_failure(self):
        """Test 8: If retrieval returns no results, QA engine does not execute answer extraction."""
        result = self.qa_engine.answer("Any query", retrieved_chunks=[])
        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["similarity_score"], 0.0)

    def test_9_multiple_retrieved_chunks(self):
        """Test 9: Verify answer is selected from the most relevant chunk, not blindly using the first result."""
        # Query targeting chunk 3 (TF-IDF and cosine similarity)
        question = "How does cosine similarity evaluate vector representations?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        result = self.qa_engine.answer(question, retrieved)

        self.assertTrue(result["found"])
        self.assertEqual(result["chunk_id"], "page_3_chunk_1")
        self.assertEqual(result["page"], 3)
        self.assertIn("Cosine similarity evaluates", result["answer"])

    def test_10_functional_api(self):
        """Test 10: Functional interface answer_question matches class behavior."""
        question = "What is Natural Language Processing?"
        retrieved = self.retriever.search(question, top_k=3, min_similarity=0.10)

        res_func = answer_question(question, retrieved)
        self.assertTrue(res_func["found"])
        self.assertEqual(res_func["chunk_id"], "page_1_chunk_1")
        self.assertIn("Natural Language Processing is a branch", res_func["answer"])


if __name__ == "__main__":
    unittest.main()
