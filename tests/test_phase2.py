"""Automated Test Suite for Phase 2: Text Chunking & Retrieval Data Preparation."""

import unittest
import pymupdf as fitz  # PyMuPDF

from src.pdf_processor import extract_text_from_pdf
from src.text_processor import process_page_data
from src.chunker import (
    chunk_document,
    chunk_page,
    split_text_into_chunks,
    clean_chunk_text,
    validate_chunks,
    calculate_chunk_stats,
    ChunkingError,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF from a list of page strings."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=2000)
        if text:
            rect = fitz.Rect(50, 50, 545, 1950)
            page.insert_textbox(rect, text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPhase2TextChunking(unittest.TestCase):
    """Test suite covering all requirements of Phase 2."""

    def test_1_small_pdf_with_several_paragraphs(self):
        """Test 1: A small PDF with several paragraphs."""
        para1 = "Natural Language Processing enables computers to understand human language."
        para2 = "Tokenization breaks continuous text into individual words, phrases, or symbols."
        para3 = "Chunking groups individual tokens and sentences into larger semantic units."
        full_text = f"{para1}\n\n{para2}\n\n{para3}"

        pdf_bytes = create_in_memory_pdf([full_text])
        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed_pages = process_page_data(raw_pages)

        # Chunk with small size to create multiple paragraph chunks
        chunks = chunk_document(
            processed_pages=processed_pages,
            source="TestDoc_Paragraphs.pdf",
            chunk_size=120,
            chunk_overlap=30,
        )

        self.assertGreaterEqual(len(chunks), 3)

        # Check metadata
        for c in chunks:
            self.assertEqual(c["source"], "TestDoc_Paragraphs.pdf")
            self.assertEqual(c["page"], 1)
            self.assertTrue(c["chunk_id"].startswith("page_1_chunk_"))
            self.assertIn("original_text", c)
            self.assertIn("cleaned_text", c)
            self.assertGreater(c["char_count"], 0)

        # Check that all 3 concepts are represented in the chunks
        joined_orig = " ".join(c["original_text"] for c in chunks)
        self.assertIn("Natural Language Processing", joined_orig)
        self.assertIn("Tokenization", joined_orig)
        self.assertIn("Chunking", joined_orig)

        # Validate chunk collection
        is_valid, issues = validate_chunks(chunks)
        self.assertTrue(is_valid, f"Validation failed: {issues}")

    def test_2_multipage_pdf(self):
        """Test 2: A multi-page PDF."""
        p1 = "Page one content discusses retrieval mechanisms and inverted indexes."
        p2 = "Page two focuses on vector embeddings, semantic search, and similarity scoring."
        p3 = "Page three explains evaluation metrics such as Mean Reciprocal Rank and NDCG."
        pdf_bytes = create_in_memory_pdf([p1, p2, p3])

        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed_pages = process_page_data(raw_pages)

        chunks = chunk_document(
            processed_pages=processed_pages,
            source="MultiPage_NLP.pdf",
            chunk_size=500,
            chunk_overlap=50,
        )

        # Verify page mapping
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["chunk_id"], "page_1_chunk_1")
        self.assertEqual(chunks[1]["page"], 2)
        self.assertEqual(chunks[1]["chunk_id"], "page_2_chunk_1")
        self.assertEqual(chunks[2]["page"], 3)
        self.assertEqual(chunks[2]["chunk_id"], "page_3_chunk_1")

        # Verify all sources match
        for c in chunks:
            self.assertEqual(c["source"], "MultiPage_NLP.pdf")

        # Verify chunk IDs are unique
        chunk_ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

        # Test statistics calculation
        stats = calculate_chunk_stats(chunks, processed_pages)
        self.assertEqual(stats["total_pages"], 3)
        self.assertEqual(stats["pages_with_text"], 3)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["chunks_per_page"], {1: 1, 2: 1, 3: 1})
        self.assertGreater(stats["avg_chunk_size"], 0)

    def test_3_very_long_paragraph(self):
        """Test 3: A PDF containing a very long paragraph (larger than chunk_size)."""
        sentences = [
            f"Sentence number {i} explains how large language models and information retrieval systems interact."
            for i in range(1, 25)
        ]
        long_paragraph = " ".join(sentences)
        self.assertGreater(len(long_paragraph), 2000)

        pdf_bytes = create_in_memory_pdf([long_paragraph])
        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed_pages = process_page_data(raw_pages)

        chunk_size = 500
        chunk_overlap = 50
        chunks = chunk_document(
            processed_pages=processed_pages,
            source="LongPara.pdf",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Should split long paragraph into multiple chunks
        self.assertGreater(len(chunks), 4)

        # Verify every chunk respects chunk_size (or tiny margin if word boundary)
        for c in chunks:
            # Hierarchy ensures sentences or words are packed within chunk_size
            self.assertLessEqual(len(c["original_text"]), chunk_size + 15)
            self.assertGreater(len(c["original_text"]), 20)

        # Verify no sentence words were broken in half (e.g. "Sentence" not "Sente")
        for c in chunks:
            words = c["original_text"].split()
            for w in words:
                self.assertNotIn("Sentenc", [w])  # not truncated

    def test_4_pdf_with_empty_pages(self):
        """Test 4: A PDF with empty pages and whitespace-only pages."""
        p1 = "Page one has valuable content about text tokenization."
        p2 = ""  # Completely blank
        p3 = "   \n\n\t  \n  "  # Whitespace only
        p4 = "Page four concludes the document with summary findings."
        pdf_bytes = create_in_memory_pdf([p1, p2, p3, p4])

        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed_pages = process_page_data(raw_pages)

        chunks = chunk_document(
            processed_pages=processed_pages,
            source="EmptyPages.pdf",
            chunk_size=500,
            chunk_overlap=50,
        )

        # Only page 1 and page 4 should produce chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["chunk_id"], "page_1_chunk_1")
        self.assertEqual(chunks[1]["page"], 4)
        self.assertEqual(chunks[1]["chunk_id"], "page_4_chunk_1")

        # Stats should accurately show 4 total pages, 2 pages with text
        stats = calculate_chunk_stats(chunks, processed_pages)
        self.assertEqual(stats["total_pages"], 4)
        self.assertEqual(stats["pages_with_text"], 2)
        self.assertEqual(stats["total_chunks"], 2)
        self.assertEqual(stats["chunks_per_page"], {1: 1, 4: 1})

    def test_5_headings_and_short_paragraphs(self):
        """Test 5: A PDF containing headings and short paragraphs."""
        doc_text = """Chapter 1: Foundations of NLP

Natural Language Processing is a subfield of computer science.

Section 1.1: Tokenization

Tokens represent words or subwords.

Section 1.2: Stop Words

Stop words are filtered during preprocessing."""

        pdf_bytes = create_in_memory_pdf([doc_text])
        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed_pages = process_page_data(raw_pages)

        chunks = chunk_document(
            processed_pages=processed_pages,
            source="Headings_Doc.pdf",
            chunk_size=150,
            chunk_overlap=30,
        )

        self.assertGreater(len(chunks), 1)

        # Verify no empty chunks or single-character chunks
        for c in chunks:
            self.assertTrue(len(c["original_text"].strip()) >= 15)
            self.assertTrue(any(ch.isalnum() for ch in c["original_text"]))

        # Verify unique IDs
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_6_chunk_overlap_at_boundary(self):
        """Test 6: A document where important information occurs near a chunk boundary."""
        sentence1 = "Natural Language Processing is a field of artificial intelligence."
        sentence2 = "It is used to analyze textual data."

        # Chunk with size that forces split between sentence 1 and sentence 2
        chunk_size = 90
        chunk_overlap = 35
        chunks = split_text_into_chunks(
            text=f"{sentence1} {sentence2}",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.assertEqual(len(chunks), 2)
        chunk1, chunk2 = chunks[0], chunks[1]

        # Chunk 1 contains sentence 1
        self.assertIn("Natural Language Processing", chunk1)
        self.assertIn("artificial intelligence.", chunk1)

        # Chunk 2 contains overlap from end of chunk 1 and sentence 2
        self.assertIn("field of artificial intelligence.", chunk2)
        self.assertIn("It is used to analyze", chunk2)

    def test_section_8_cleaned_vs_original_text(self):
        """Verify Requirement 8: original_text vs cleaned_text distinction."""
        sample_query = "What is Natural Language Processing?"
        cleaned = clean_chunk_text(sample_query)

        # Matches prompt example: 'What is Natural Language Processing?' -> 'what natural language processing'
        self.assertEqual(cleaned, "what natural language processing")

        # Ensure casing and punctuation are preserved in original_text
        page_data = {"page": 1, "cleaned_text": sample_query, "original_text": sample_query}
        chunks = chunk_page(page_data, source="test.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["original_text"], sample_query)
        self.assertEqual(chunks[0]["cleaned_text"], "what natural language processing")

    def test_chunk_validation_and_errors(self):
        """Verify validate_chunks identifies missing keys and duplicates."""
        valid_chunks = [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "doc.pdf",
                "page": 1,
                "original_text": "Sample text here.",
                "cleaned_text": "sample text here",
            }
        ]
        is_val, issues = validate_chunks(valid_chunks)
        self.assertTrue(is_val)
        self.assertEqual(issues, [])

        # Duplicate ID
        duplicate_chunks = valid_chunks + [valid_chunks[0].copy()]
        is_val_dup, issues_dup = validate_chunks(duplicate_chunks)
        self.assertFalse(is_val_dup)
        self.assertTrue(any("Duplicate chunk_id" in err for err in issues_dup))

        # Missing key
        invalid_chunk = [{"chunk_id": "p_1", "page": 1}]
        is_val_inv, issues_inv = validate_chunks(invalid_chunk)
        self.assertFalse(is_val_inv)
        self.assertTrue(any("missing required keys" in err for err in issues_inv))

    def test_parameter_validation(self):
        """Verify invalid chunking parameters raise ChunkingError."""
        with self.assertRaises(ChunkingError):
            split_text_into_chunks("Text", chunk_size=0)

        with self.assertRaises(ChunkingError):
            split_text_into_chunks("Text", chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ChunkingError):
            split_text_into_chunks("Text", chunk_size=100, chunk_overlap=-10)


if __name__ == "__main__":
    unittest.main()
