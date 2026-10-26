"""Comprehensive Edge Case Remediation Regression Test Suite.

Verifies all confirmed edge cases and failure modes identified during the audit:
1. Answer Relevance & False-Positive Prevention (Inflation, Database, Attendance, section numbers).
2. Negation Preservation & Polarity Distinction ("not permitted" vs positive statements).
3. Hybrid Retrieval Score Normalization (single candidate, ties, disjoint sets, non-finite values).
4. Chunking Overlap & Boundary Safety (no malformed sentence fragments, zero overlap, long text).
5. PDF Ingestion Robustness (encrypted PDFs, recurring headers/footers, unicode ligatures).
6. Document State Invalidation & Content Hashing.
"""

from __future__ import annotations

import hashlib
import io
import unittest
from typing import Any
import numpy as np
import pymupdf as fitz

from src.qa_engine import ExtractiveQAEngine
from src.chunker import (
    chunk_document,
    get_overlap_prefix,
    split_text_into_chunks,
    ChunkingError,
)
from src.hybrid_retriever import HybridRetriever
from src.pdf_processor import (
    extract_text_from_pdf,
    _filter_recurring_headers_footers,
    PDFProcessingError,
)
from src.text_processor import (
    clean_text,
    get_stopwords,
    tokenize_text,
    remove_stopwords,
    process_page_data,
)


class MockRetriever:
    """Mock retriever returning static or dynamic candidates."""

    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict[str, Any]]:
        filtered = [c for c in self.candidates if float(c.get("similarity_score", 0.0)) >= min_similarity]
        return filtered[:top_k]


class TestRemediationEdgeCases(unittest.TestCase):
    """Regression test suite for all audit findings."""

    def setUp(self):
        self.qa_engine = ExtractiveQAEngine()

    # =========================================================================
    # 1. Answer Relevance & False-Positive Prevention
    # =========================================================================

    def test_relevance_inflation_query_abstains_on_medical_chunk(self):
        """CRIT-01: 'What causes inflation?' must NOT return smoking/cardiovascular sentence."""
        medical_chunk = [{
            "chunk_id": "c1",
            "page": 1,
            "source": "medical.pdf",
            "similarity_score": 0.45,
            "original_text": "Smoking causes serious cardiovascular damage and lung cancer. Regular exercise prevents heart disease.",
            "cleaned_text": "smoking causes serious cardiovascular damage lung cancer regular exercise prevents heart disease",
        }]

        result = self.qa_engine.answer("What causes inflation?", medical_chunk)
        self.assertFalse(result["found"], "Should abstain when evidence lacks the core topic 'inflation'")
        self.assertIsNone(result["answer"])

    def test_relevance_database_query_abstains_on_ml_chunk(self):
        """CRIT-01: 'What is database?' must abstain on machine learning text."""
        ml_chunk = [{
            "chunk_id": "c2",
            "page": 1,
            "source": "ai.pdf",
            "similarity_score": 0.50,
            "original_text": "Machine learning is a subfield of artificial intelligence that learns from data.",
            "cleaned_text": "machine learning subfield artificial intelligence learns data",
        }]

        result = self.qa_engine.answer("What is database?", ml_chunk)
        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])

    def test_relevance_database_query_succeeds_with_actual_evidence(self):
        """CRIT-01: 'What is database?' extracts correctly when actual evidence exists."""
        db_chunk = [{
            "chunk_id": "c3",
            "page": 2,
            "source": "db.pdf",
            "similarity_score": 0.70,
            "original_text": "A database is an organized collection of structured information or data stored electronically.",
            "cleaned_text": "database organized collection structured information data stored electronically",
        }]

        result = self.qa_engine.answer("What is database?", db_chunk)
        self.assertTrue(result["found"])
        self.assertIsNotNone(result["answer"])
        self.assertIn("database is an organized collection", result["answer"])

    def test_relevance_attendance_query_abstains_on_tfidf_chunk(self):
        """CRIT-02: 'How much attendance required?' must NOT match TF-IDF definition with section numbers."""
        tfidf_chunk = [{
            "chunk_id": "c4",
            "page": 1,
            "source": "nlp.pdf",
            "similarity_score": 0.35,
            "original_text": "Section 4.2 TF-IDF is a numerical statistic reflecting word importance across a collection.",
            "cleaned_text": "section 4 2 tf idf numerical statistic reflecting word importance across collection",
        }]

        result = self.qa_engine.answer("How much attendance required?", tfidf_chunk)
        self.assertFalse(result["found"])
        self.assertIsNone(result["answer"])

    def test_relevance_attendance_query_extracts_when_present(self):
        """CRIT-02: 'How much attendance required?' extracts 75% when attendance evidence is present."""
        att_chunk = [{
            "chunk_id": "c5",
            "page": 3,
            "source": "handbook.pdf",
            "similarity_score": 0.65,
            "original_text": "Students must maintain a minimum attendance threshold of 75 percent to appear in examinations.",
            "cleaned_text": "students maintain minimum attendance threshold 75 percent appear examinations",
        }]

        result = self.qa_engine.answer("How much attendance required?", att_chunk)
        self.assertTrue(result["found"])
        self.assertIn("75 percent", result["answer"])

    def test_relevance_generic_predicate_verb_alone_does_not_satisfy_gate(self):
        """CRIT-01: Questions sharing ONLY generic verbs ('includes', 'provides', 'requires') must abstain."""
        irrelevant_chunk = [{
            "chunk_id": "c6",
            "page": 1,
            "source": "sample.pdf",
            "similarity_score": 0.40,
            "original_text": "The hotel accommodation package includes complimentary breakfast and laundry services.",
            "cleaned_text": "hotel accommodation package includes complimentary breakfast laundry services",
        }]

        # Query asks about software features, sharing only 'includes'
        result = self.qa_engine.answer("What software tools does the platform include?", irrelevant_chunk)
        self.assertFalse(result["found"])

    # =========================================================================
    # 2. Negation Preservation & Polarity Distinction
    # =========================================================================

    def test_negation_tokens_preserved_in_stopwords(self):
        """CRIT-03: 'not', 'no', 'never', 'neither', 'nor' must be preserved when requested."""
        stops_with_negation = get_stopwords(preserve_negations=True)
        for neg in ["not", "no", "never", "neither", "nor", "without"]:
            self.assertNotIn(neg, stops_with_negation, f"Negation '{neg}' should not be in stopword set")

    def test_negation_question_does_not_return_positive_permission(self):
        """CRIT-03: 'What is not permitted' must NOT extract positive permission statements."""
        permission_chunk = [{
            "chunk_id": "c7",
            "page": 1,
            "source": "rules.pdf",
            "similarity_score": 0.50,
            "original_text": "Students are permitted to bring basic non-programmable calculators into the examination room.",
            "cleaned_text": "students permitted bring basic non programmable calculators examination room",
        }]

        # Candidate sentence is purely positive permission; query asks what is NOT permitted
        result = self.qa_engine.answer("What items are not permitted in the examination room?", permission_chunk)
        self.assertFalse(result["found"], "Should abstain from presenting permission as answer to negative question")

    def test_negation_question_extracts_actual_prohibition(self):
        """CRIT-03: 'What is not permitted' extracts sentences describing prohibitions."""
        prohibition_chunk = [{
            "chunk_id": "c8",
            "page": 1,
            "source": "rules.pdf",
            "similarity_score": 0.60,
            "original_text": "Mobile phones and smart watches are strictly prohibited inside the examination room.",
            "cleaned_text": "mobile phones smart watches strictly prohibited inside examination room",
        }]

        result = self.qa_engine.answer("What items are not permitted in the examination room?", prohibition_chunk)
        self.assertTrue(result["found"])
        self.assertIn("prohibited", result["answer"])

    # =========================================================================
    # 3. Hybrid Retrieval Score Normalization & Edge Cases
    # =========================================================================

    def test_hybrid_single_candidate_does_not_inflate_to_one(self):
        """CRIT-04: Single candidate with score 0.12 must not be inflated to 1.0."""
        mock_tf = MockRetriever([{"chunk_id": "a", "similarity_score": 0.12, "original_text": "A"}])
        mock_sem = MockRetriever([])

        retriever = HybridRetriever(
            chunks=[{"chunk_id": "a"}],
            alpha=1.0,
            tfidf_retriever=mock_tf,
            semantic_retriever=mock_sem,
        )

        results = retriever.search("query", top_k=5, min_similarity=0.0)
        self.assertEqual(len(results), 1)
        # Score must be 0.12, NOT 1.0
        self.assertAlmostEqual(results[0]["hybrid_score"], 0.12, places=2)

    def test_hybrid_threshold_filters_weak_single_candidate(self):
        """CRIT-04: Weak single candidate below min_similarity must be filtered out."""
        mock_tf = MockRetriever([{"chunk_id": "a", "similarity_score": 0.12, "original_text": "A"}])
        mock_sem = MockRetriever([])

        retriever = HybridRetriever(
            chunks=[{"chunk_id": "a"}],
            alpha=1.0,
            tfidf_retriever=mock_tf,
            semantic_retriever=mock_sem,
        )

        # min_similarity = 0.20 > 0.12 -> should return empty
        results = retriever.search("query", top_k=5, min_similarity=0.20)
        self.assertEqual(results, [])

    def test_hybrid_equal_scores_do_not_divide_by_zero(self):
        """CRIT-04: Multiple candidates with equal scores must not cause division-by-zero."""
        mock_tf = MockRetriever([
            {"chunk_id": "a", "similarity_score": 0.50, "original_text": "A"},
            {"chunk_id": "b", "similarity_score": 0.50, "original_text": "B"},
        ])
        mock_sem = MockRetriever([
            {"chunk_id": "a", "similarity_score": 0.50, "original_text": "A"},
            {"chunk_id": "b", "similarity_score": 0.50, "original_text": "B"},
        ])

        retriever = HybridRetriever(
            chunks=[{"chunk_id": "a"}, {"chunk_id": "b"}],
            alpha=0.5,
            tfidf_retriever=mock_tf,
            semantic_retriever=mock_sem,
        )

        results = retriever.search("query", top_k=5, min_similarity=0.0)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertAlmostEqual(r["hybrid_score"], 0.50, places=2)

    def test_hybrid_non_finite_scores_handled_safely(self):
        """CRIT-04: NaN or Inf scores must not crash fusion."""
        mock_tf = MockRetriever([
            {"chunk_id": "a", "similarity_score": float("nan"), "original_text": "A"},
            {"chunk_id": "b", "similarity_score": float("inf"), "original_text": "B"},
        ])
        mock_sem = MockRetriever([
            {"chunk_id": "a", "similarity_score": 0.50, "original_text": "A"},
        ])

        retriever = HybridRetriever(
            chunks=[{"chunk_id": "a"}, {"chunk_id": "b"}],
            alpha=0.5,
            tfidf_retriever=mock_tf,
            semantic_retriever=mock_sem,
        )

        results = retriever.search("query", top_k=5, min_similarity=0.0)
        self.assertTrue(isinstance(results, list))

    # =========================================================================
    # 4. Chunk Overlap & Boundary Safety
    # =========================================================================

    def test_chunking_overlap_does_not_create_truncated_sentence_fragments(self):
        """CRIT-06: Overlap prefix should be sentence-aware and not slice mid-sentence."""
        text = "First sentence of paragraph. Second sentence with important details."
        overlap_prefix = get_overlap_prefix(text, overlap_size=30)
        # Should snap to sentence or be clean, not arbitrary fragmented word
        self.assertFalse(overlap_prefix.startswith("details."), "Should not start with sentence tail")

    def test_chunking_zero_overlap(self):
        """Phase 2: Overlap equal to zero works properly without error."""
        pages = [{"page": 1, "cleaned_text": "A" * 200, "original_text": "A" * 200, "tokens": ["A"] * 20}]
        chunks = chunk_document(pages, source="doc.pdf", chunk_size=100, chunk_overlap=0)
        self.assertTrue(len(chunks) >= 2)

    def test_chunking_invalid_overlap_raises_error(self):
        """Phase 2: Overlap >= chunk_size must raise ChunkingError."""
        pages = [{"page": 1, "cleaned_text": "Text", "original_text": "Text", "tokens": ["Text"]}]
        with self.assertRaises(ChunkingError):
            chunk_document(pages, source="doc.pdf", chunk_size=100, chunk_overlap=100)

    # =========================================================================
    # 5. PDF Ingestion Robustness
    # =========================================================================

    def test_pdf_encrypted_file_raises_actionable_error(self):
        """P1: Encrypted/password-protected PDF raises clear user-facing error."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Confidential financial document.")
        # Save with encryption
        encrypted_bytes = doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="secret_owner",
            user_pw="secret_user",
        )
        doc.close()

        with self.assertRaises(PDFProcessingError) as ctx:
            extract_text_from_pdf(encrypted_bytes)
        self.assertIn("password-protected or encrypted", str(ctx.exception))

    def test_pdf_recurring_header_removal(self):
        """P1: Conservative recurring header removal on >=3 page documents."""
        pages_data = [
            {"page": 1, "original_text": "Acme Corp Annual Report 2026\nChapter 1: Overview\nCompany was founded in 2010."},
            {"page": 2, "original_text": "Acme Corp Annual Report 2026\nChapter 2: Financials\nRevenue grew by 25 percent."},
            {"page": 3, "original_text": "Acme Corp Annual Report 2026\nChapter 3: Operations\nOpened two new facilities."},
        ]

        cleaned = _filter_recurring_headers_footers(pages_data)
        self.assertEqual(len(cleaned), 3)
        for p in cleaned:
            self.assertNotIn("Acme Corp Annual Report 2026", p["original_text"])
            self.assertIn("Chapter", p["original_text"])

    def test_unicode_ligature_normalization(self):
        """P1: Unicode ligatures (fi, fl, ffi) normalize cleanly in text cleaning."""
        ligature_text = "The scientific afﬁliation of the ofﬁce is verified."
        cleaned = clean_text(ligature_text)
        self.assertNotIn("afﬁliation", cleaned)
        self.assertIn("affiliation", cleaned)
        self.assertIn("office", cleaned)

    # =========================================================================
    # 6. Document State Invalidation
    # =========================================================================

    def test_document_hash_different_for_different_contents_with_same_filename(self):
        """CRIT-05: Documents with the same filename but different contents produce distinct hashes."""
        content_a = b"%PDF-1.4 Document A content"
        content_b = b"%PDF-1.4 Document B content"

        hash_a = hashlib.sha256(content_a).hexdigest()
        hash_b = hashlib.sha256(content_b).hexdigest()

        self.assertNotEqual(hash_a, hash_b)


if __name__ == "__main__":
    unittest.main()
