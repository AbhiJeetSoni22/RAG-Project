"""Regression Test Suite for Question Answering Relevance and Evidence Grounding.

Specifically tests:
1. The reproduced failure: 'How much attendance required?' on a document mentioning
   TF-IDF, Section 5, and 4.2 must NOT return the unrelated TF-IDF sentence.
2. Direct factual questions with attendance explicitly present.
3. Paraphrased attendance questions ('minimum attendance', 'attendance requirement',
   'required attendance percentage').
4. Unrelated questions ('What is quantum entanglement and teleportation?').
5. Questions whose answers are genuinely absent from the uploaded document.
6. Evidence preservation and strict citation metadata (source, page, chunk_id, evidence).
7. All 3 retrieval modes (TF-IDF, Semantic, Hybrid).
"""

import unittest
from src.retriever import TFIDFRetriever
from src.semantic_retriever import SemanticRetriever
from src.hybrid_retriever import HybridRetriever
from src.qa_engine import ExtractiveQAEngine, answer_question


class MockSemanticEngine:
    """Mock semantic retriever for fast, deterministic unit test verification."""

    def __init__(self, mapping: dict[str, list[dict]]) -> None:
        self.mapping = mapping

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict]:
        candidates = self.mapping.get(query, [])
        return [c for c in candidates if float(c.get("similarity_score", 0.0)) >= min_similarity][:top_k]


class TestQARelevanceRegression(unittest.TestCase):
    """Test suite ensuring candidate answers are strictly relevant to the question."""

    def setUp(self):
        """Set up test chunks representing syllabus and lecture notes."""
        self.engine = ExtractiveQAEngine()

        # Chunk with TF-IDF and numbers (reproduced bug scenario)
        self.tfidf_chunk = {
            "chunk_id": "notes_chunk_1",
            "source": "NLP_Lecture_Notes.pdf",
            "page": 4,
            "original_text": (
                "Techniques described in Section 5, matches meaning. "
                "4.2 TF-IDF TF-IDF stands for term frequency-inverse document frequency."
            ),
            "cleaned_text": "techniques described section 5 matches meaning 4 2 tf idf tf idf stands term frequency inverse document frequency",
            "char_count": 128,
            "word_count": 18,
            "similarity_score": 0.35,
        }

        # Chunk with explicit attendance requirement
        self.attendance_chunk = {
            "chunk_id": "policy_chunk_1",
            "source": "Student_Handbook.pdf",
            "page": 2,
            "original_text": (
                "Students are required to maintain a minimum attendance of 75 percent across all enrolled courses. "
                "Failure to meet attendance requirements may result in disciplinary action."
            ),
            "cleaned_text": "students required maintain minimum attendance 75 percent across enrolled courses failure meet attendance requirements result disciplinary action",
            "char_count": 180,
            "word_count": 23,
            "similarity_score": 0.82,
        }

        # Chunk with NLP definitions
        self.nlp_chunk = {
            "chunk_id": "nlp_chunk_1",
            "source": "Course_Intro.pdf",
            "page": 1,
            "original_text": (
                "Natural Language Processing is a multidisciplinary field at the intersection of "
                "computer science and artificial intelligence."
            ),
            "cleaned_text": "natural language processing multidisciplinary field intersection computer science artificial intelligence",
            "char_count": 128,
            "word_count": 16,
            "similarity_score": 0.75,
        }

    # -------------------------------------------------------------------------
    # 1. Regression Test for the Reported Reproduction Case
    # -------------------------------------------------------------------------
    def test_reproduced_failure_unrelated_sentence_rejected(self):
        """Ensure 'How much attendance required?' rejects unrelated sentence with numbers."""
        question = "How much attendance required?"
        # The document contains only the TF-IDF chunk (unrelated to attendance)
        retrieved_chunks = [self.tfidf_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        # Must NOT return the Section 5 / TF-IDF sentence
        self.assertFalse(result["found"], "Engine should not claim an answer was found for unrelated chunk.")
        self.assertIsNone(result["answer"], "Answer must be None when no relevant evidence exists.")
        self.assertIn(
            "I couldn't find a sufficiently relevant answer in this document.",
            result["reason"],
            "Must provide clear fallback message."
        )

    # -------------------------------------------------------------------------
    # 2. Explicit Presence of Attendance in Document
    # -------------------------------------------------------------------------
    def test_explicit_attendance_present(self):
        """Ensure 'How much attendance required?' extracts 75 percent when passage is present."""
        question = "How much attendance required?"
        retrieved_chunks = [self.attendance_chunk, self.tfidf_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertTrue(result["found"])
        self.assertIsNotNone(result["answer"])
        self.assertIn("75 percent", result["answer"])
        self.assertIn("minimum attendance of 75 percent", result["answer"])
        self.assertEqual(result["source"], "Student_Handbook.pdf")
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["chunk_id"], "policy_chunk_1")
        self.assertEqual(result["evidence"], self.attendance_chunk["original_text"])

    # -------------------------------------------------------------------------
    # 3. Paraphrased Attendance Questions
    # -------------------------------------------------------------------------
    def test_paraphrased_questions(self):
        """Ensure morphological variations and paraphrased questions match correctly."""
        paraphrases = [
            "What is the minimum attendance?",
            "What is the attendance requirement?",
            "What is the required attendance percentage?",
            "How much attendance is mandatory for students?",
        ]
        retrieved_chunks = [self.attendance_chunk]

        for q in paraphrases:
            with self.subTest(question=q):
                res = self.engine.answer(q, retrieved_chunks)
                self.assertTrue(res["found"], f"Failed to answer paraphrased query: {q}")
                self.assertIn("75 percent", res["answer"])
                self.assertIn("minimum attendance", res["answer"].lower())

    # -------------------------------------------------------------------------
    # 4. Question Unrelated to Document
    # -------------------------------------------------------------------------
    def test_unrelated_question(self):
        """Ensure unrelated question returns no answer without fabricating."""
        question = "What is quantum entanglement and teleportation?"
        retrieved_chunks = [self.nlp_chunk, self.attendance_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])
        self.assertIn("I couldn't find a sufficiently relevant answer in this document.", result["reason"])

    # -------------------------------------------------------------------------
    # 5. Genuinely Absent Answer from Document
    # -------------------------------------------------------------------------
    def test_genuinely_absent_question(self):
        """Ensure query about attendance returns not found when document lacks attendance info."""
        question = "How much attendance required?"
        # Document only has NLP intro
        retrieved_chunks = [self.nlp_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])
        self.assertIn("I couldn't find a sufficiently relevant answer in this document.", result["reason"])

    # -------------------------------------------------------------------------
    # 6. Evidence Grounding and Attribution Preservation
    # -------------------------------------------------------------------------
    def test_evidence_grounding_preservation(self):
        """Verify returned answer is an exact substring of the original text evidence."""
        question = "What is Natural Language Processing?"
        retrieved_chunks = [self.nlp_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertTrue(result["found"])
        self.assertIn(result["answer"], result["evidence"])
        self.assertEqual(result["source"], "Course_Intro.pdf")
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["chunk_id"], "nlp_chunk_1")

    # -------------------------------------------------------------------------
    # 7. Testing All Retrieval Modes (TF-IDF, Mock Semantic, Hybrid)
    # -------------------------------------------------------------------------
    def test_all_retrieval_modes_reproduction_safety(self):
        """Test the reproduction case across TF-IDF, Semantic, and Hybrid retrieval."""
        all_chunks = [self.tfidf_chunk, self.nlp_chunk]
        question = "How much attendance required?"

        # A. TF-IDF Retrieval
        tfidf_retriever = TFIDFRetriever(all_chunks)
        tfidf_results = tfidf_retriever.search(question, top_k=3, min_similarity=0.10)
        ans_tfidf = self.engine.answer(question, tfidf_results)
        self.assertFalse(ans_tfidf["found"], "TF-IDF mode should not return answer for absent attendance topic.")

        # B. Mock Semantic Retrieval (simulating dense embedding baseline)
        mock_semantic = MockSemanticEngine({
            question: [
                {**self.tfidf_chunk, "similarity_score": 0.22},
                {**self.nlp_chunk, "similarity_score": 0.15},
            ]
        })
        sem_results = mock_semantic.search(question, top_k=3, min_similarity=0.10)
        ans_sem = self.engine.answer(question, sem_results)
        self.assertFalse(ans_sem["found"], "Semantic mode should not accept unrelated candidate even if retrieved.")

        # C. Hybrid Retrieval
        hybrid_retriever = HybridRetriever(
            chunks=all_chunks,
            alpha=0.5,
            tfidf_retriever=tfidf_retriever,
            semantic_retriever=mock_semantic,
        )
        hybrid_results = hybrid_retriever.search(question, top_k=3, min_similarity=0.10)
        ans_hyb = self.engine.answer(question, hybrid_results)
        self.assertFalse(ans_hyb["found"], "Hybrid mode should not accept unrelated candidate.")


if __name__ == "__main__":
    unittest.main()
