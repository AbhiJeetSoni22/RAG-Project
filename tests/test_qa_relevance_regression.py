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

        # Chunk with Machine Learning (reproduced bug scenario for 'what is database?')
        self.ml_chunk = {
            "chunk_id": "ml_chunk_1",
            "source": "Machine_Learning_Guide.pdf",
            "page": 2,
            "original_text": (
                "Machine Learning: Supervised, Unsupervised, and Reinforcement Learning "
                "2.1 What Machine Learning Is "
                "Machine learning (ML) is a subfield of AI in which algorithms improve their performance "
                "on a task by learning from data rather than by following explicitly programmed rules. "
                "The result of training is a model, which is a mathematical function whose parameters "
                "have been adjusted to fit the data."
            ),
            "cleaned_text": (
                "machine learning supervised unsupervised reinforcement learning 2.1 what machine learning is "
                "machine learning ml is a subfield of ai algorithms improve performance task learning data "
                "explicitly programmed rules result training model mathematical function parameters adjusted fit data"
            ),
            "char_count": 395,
            "word_count": 55,
            "similarity_score": 0.45,
        }

        # Chunk with explicit database definition
        self.database_chunk = {
            "chunk_id": "db_chunk_1",
            "source": "Database_Systems.pdf",
            "page": 1,
            "original_text": (
                "A database is an organized collection of structured data stored electronically in a computer system. "
                "Databases are typically controlled by a database management system (DBMS)."
            ),
            "cleaned_text": "database organized collection structured data stored electronically computer system databases typically controlled database management system dbms",
            "char_count": 182,
            "word_count": 23,
            "similarity_score": 0.88,
        }

    # -------------------------------------------------------------------------
    # Test A: Definition Questions (what is database?)
    # -------------------------------------------------------------------------
    def test_database_question_abstains_on_unrelated_ml_document(self):
        """Test A1: 'what is database?' on ML document must abstain, not return ML definition."""
        question = "what is database?"
        retrieved_chunks = [self.ml_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertFalse(result["found"], "Engine must not return ML passage for 'what is database?'.")
        self.assertIsNone(result["answer"], "Answer must be None when no database evidence exists.")
        self.assertIn("I couldn't find a sufficiently relevant answer in this document.", result["reason"])
        self.assertIsNotNone(result.get("diagnostics"))
        self.assertIn("abstention_reason", result["diagnostics"])

    def test_database_question_extracts_when_present(self):
        """Test A2: 'what is database?' extracts definition when database passage is present."""
        question = "what is database?"
        retrieved_chunks = [self.database_chunk, self.ml_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertTrue(result["found"])
        self.assertIsNotNone(result["answer"])
        self.assertIn("database is an organized collection", result["answer"].lower())
        self.assertEqual(result["chunk_id"], "db_chunk_1")
        self.assertEqual(result["source"], "Database_Systems.pdf")

    # -------------------------------------------------------------------------
    # Test B: Attendance Requirement Questions
    # -------------------------------------------------------------------------
    def test_attendance_question_abstains_on_unrelated_document(self):
        """Test B1: 'How much attendance required?' on TF-IDF chunk must abstain."""
        question = "How much attendance required?"
        retrieved_chunks = [self.tfidf_chunk, self.ml_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertFalse(result["found"], "Must not return TF-IDF sentence for attendance query.")
        self.assertIsNone(result["answer"])
        self.assertIn("I couldn't find a sufficiently relevant answer in this document.", result["reason"])

    def test_attendance_question_extracts_when_present(self):
        """Test B2: 'How much attendance is required?' extracts 75 percent when present."""
        question = "How much attendance is required?"
        retrieved_chunks = [self.attendance_chunk, self.ml_chunk]

        result = self.engine.answer(question, retrieved_chunks)

        self.assertTrue(result["found"])
        self.assertIsNotNone(result["answer"])
        self.assertIn("75 percent", result["answer"])
        self.assertIn("minimum attendance", result["answer"].lower())
        self.assertEqual(result["source"], "Student_Handbook.pdf")

    # -------------------------------------------------------------------------
    # Test C: Paraphrased Questions
    # -------------------------------------------------------------------------
    def test_paraphrased_questions_match_correctly(self):
        """Test C: Paraphrased questions match their grounded source passages."""
        test_pairs = [
            ("What is the minimum attendance?", "75 percent", self.attendance_chunk),
            ("What is the attendance requirement?", "attendance requirements", self.attendance_chunk),
            ("What is the required attendance percentage?", "75 percent", self.attendance_chunk),
            ("Explain the concept of Natural Language Processing", "Natural Language Processing", self.nlp_chunk),
        ]

        for query, expected_snippet, chunk in test_pairs:
            with self.subTest(query=query):
                res = self.engine.answer(query, [chunk])
                self.assertTrue(res["found"], f"Should find answer for '{query}'")
                self.assertIn(expected_snippet.lower(), res["answer"].lower())

    # -------------------------------------------------------------------------
    # Test D: Unanswerable & Absent Questions
    # -------------------------------------------------------------------------
    def test_unanswerable_question_abstains(self):
        """Test D: Asking about facts absent from document strictly abstains."""
        unanswerable = [
            "What is quantum entanglement and teleportation?",
            "What is the capital of France?",
            "How much is the university tuition fee per credit hour?",
        ]
        retrieved_chunks = [self.nlp_chunk, self.ml_chunk]

        for q in unanswerable:
            with self.subTest(question=q):
                res = self.engine.answer(q, retrieved_chunks)
                self.assertFalse(res["found"], f"Must abstain on absent question: '{q}'")
                self.assertIsNone(res["answer"])
                self.assertIn("I couldn't find a sufficiently relevant answer in this document.", res["reason"])

    # -------------------------------------------------------------------------
    # Test E: All Retrieval Modes (TF-IDF, Mock Semantic, Hybrid)
    # -------------------------------------------------------------------------
    def test_all_retrieval_modes_database_and_attendance(self):
        """Test E: Verify all retrieval modes handle present vs absent queries correctly."""
        all_chunks = [self.ml_chunk, self.nlp_chunk, self.attendance_chunk, self.database_chunk]

        # 1. Database query on document without database chunk
        chunks_no_db = [self.ml_chunk, self.nlp_chunk]
        tfidf_no_db = TFIDFRetriever(chunks_no_db)
        mock_sem_no_db = MockSemanticEngine({
            "what is database?": [{**self.ml_chunk, "similarity_score": 0.28}]
        })
        hybrid_no_db = HybridRetriever(
            chunks=chunks_no_db, alpha=0.5,
            tfidf_retriever=tfidf_no_db, semantic_retriever=mock_sem_no_db
        )

        for mode_name, ret in [("TF-IDF", tfidf_no_db), ("Semantic", mock_sem_no_db), ("Hybrid", hybrid_no_db)]:
            with self.subTest(mode=mode_name, test="absent_database"):
                passages = ret.search("what is database?", top_k=3, min_similarity=0.10)
                ans = self.engine.answer("what is database?", passages)
                self.assertFalse(ans["found"], f"{mode_name} should abstain on absent database topic.")
                self.assertIsNone(ans["answer"])

        # 2. Database query on document with database chunk
        tfidf_with_db = TFIDFRetriever(all_chunks)
        mock_sem_with_db = MockSemanticEngine({
            "what is database?": [
                {**self.database_chunk, "similarity_score": 0.85},
                {**self.ml_chunk, "similarity_score": 0.28},
            ]
        })
        hybrid_with_db = HybridRetriever(
            chunks=all_chunks, alpha=0.5,
            tfidf_retriever=tfidf_with_db, semantic_retriever=mock_sem_with_db
        )

        for mode_name, ret in [("TF-IDF", tfidf_with_db), ("Semantic", mock_sem_with_db), ("Hybrid", hybrid_with_db)]:
            with self.subTest(mode=mode_name, test="present_database"):
                passages = ret.search("what is database?", top_k=3, min_similarity=0.10)
                ans = self.engine.answer("what is database?", passages)
                self.assertTrue(ans["found"], f"{mode_name} should find answer for present database.")
                self.assertIn("database is an organized collection", ans["answer"].lower())
                self.assertEqual(ans["chunk_id"], "db_chunk_1")
                self.assertEqual(ans["source"], "Database_Systems.pdf")


if __name__ == "__main__":
    unittest.main()
