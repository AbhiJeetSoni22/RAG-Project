"""Automated Test Suite for Phase 1: Document Ingestion & Text Preprocessing."""

import io
import unittest
import pymupdf as fitz  # PyMuPDF

from src.pdf_processor import extract_text_from_pdf, PDFProcessingError
from src.text_processor import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    process_page_data,
    calculate_document_stats,
    get_stopwords,
)


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper function to create a PDF in memory with specified page texts."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            # Insert text at top-left margin (point 50, 72)
            page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPhase1DocumentIngestion(unittest.TestCase):
    """Test suite covering all requirements of Phase 1."""

    def test_1_normal_text_pdf(self):
        """Test 1: A normal text-based PDF."""
        text = "Artificial intelligence and machine learning are revolutionizing modern computing."
        pdf_bytes = create_in_memory_pdf([text])

        # Extract text
        raw_pages = extract_text_from_pdf(pdf_bytes)
        self.assertEqual(len(raw_pages), 1)
        self.assertEqual(raw_pages[0]["page"], 1)
        self.assertIn("Artificial intelligence", raw_pages[0]["original_text"])

        # Process page data
        processed = process_page_data(raw_pages)
        self.assertEqual(len(processed), 1)
        page1 = processed[0]

        # Verify page number is preserved
        self.assertEqual(page1["page"], 1)
        # Verify original text is not destroyed
        self.assertEqual(page1["original_text"], raw_pages[0]["original_text"])
        # Verify cleaned text exists
        self.assertTrue(len(page1["cleaned_text"]) > 0)
        # Verify tokens exist
        self.assertIn("Artificial", page1["tokens"])
        self.assertIn("intelligence", page1["tokens"])
        # Verify stop words removed ('and', 'are' should be excluded)
        self.assertNotIn("and", page1["filtered_tokens"])
        self.assertNotIn("are", page1["filtered_tokens"])
        self.assertIn("Artificial", page1["filtered_tokens"])
        self.assertIn("revolutionizing", page1["filtered_tokens"])

    def test_2_multipage_pdf(self):
        """Test 2: A multi-page PDF."""
        page1_content = "This is the content of Page One."
        page2_content = "Here is the content for Page Two."
        page3_content = "Finally, this is the text on Page Three."
        pdf_bytes = create_in_memory_pdf([page1_content, page2_content, page3_content])

        raw_pages = extract_text_from_pdf(pdf_bytes)
        self.assertEqual(len(raw_pages), 3)

        # Verify page numbers are preserved and 1-indexed
        self.assertEqual([p["page"] for p in raw_pages], [1, 2, 3])

        # Verify text was NOT merged into one string
        self.assertIn("Page One", raw_pages[0]["original_text"])
        self.assertNotIn("Page Two", raw_pages[0]["original_text"])
        self.assertIn("Page Two", raw_pages[1]["original_text"])
        self.assertNotIn("Page Three", raw_pages[1]["original_text"])
        self.assertIn("Page Three", raw_pages[2]["original_text"])

        processed = process_page_data(raw_pages)
        self.assertEqual(len(processed), 3)
        self.assertEqual([p["page"] for p in processed], [1, 2, 3])

        stats = calculate_document_stats(processed)
        self.assertEqual(stats["total_pages"], 3)
        self.assertEqual(stats["pages_with_text"], 3)

    def test_3_pdf_with_page_containing_no_text(self):
        """Test 3: A PDF with a page containing little/no text."""
        page1_content = "This page has substantial content."
        page2_content = ""  # Completely blank page
        page3_content = "   \n\n   "  # Whitespace only
        pdf_bytes = create_in_memory_pdf([page1_content, page2_content, page3_content])

        raw_pages = extract_text_from_pdf(pdf_bytes)
        self.assertEqual(len(raw_pages), 3)

        processed = process_page_data(raw_pages)
        self.assertEqual(len(processed), 3)

        # Page 2 should be empty
        self.assertEqual(processed[1]["cleaned_text"], "")
        self.assertEqual(processed[1]["tokens"], [])
        self.assertEqual(processed[1]["filtered_tokens"], [])

        # Page 3 should also clean to empty
        self.assertEqual(processed[2]["cleaned_text"], "")
        self.assertEqual(processed[2]["tokens"], [])
        self.assertEqual(processed[2]["filtered_tokens"], [])

        # Check stats
        stats = calculate_document_stats(processed)
        self.assertEqual(stats["total_pages"], 3)
        self.assertEqual(stats["pages_with_text"], 1)

    def test_4_invalid_non_pdf_upload(self):
        """Test 4: Invalid/non-PDF upload."""
        # 1. Plain text file masquerading as PDF
        fake_pdf = b"This is just a plain text file, not a real PDF format."
        with self.assertRaises(PDFProcessingError) as ctx1:
            extract_text_from_pdf(fake_pdf)
        self.assertIn("Failed to open PDF document", str(ctx1.exception))

        # 2. Empty byte buffer (0 bytes)
        empty_bytes = b""
        with self.assertRaises(PDFProcessingError) as ctx2:
            extract_text_from_pdf(empty_bytes)
        self.assertIn("empty", str(ctx2.exception).lower())

        # 3. None input
        with self.assertRaises(PDFProcessingError) as ctx3:
            extract_text_from_pdf(None)
        self.assertIn("No PDF file provided", str(ctx3.exception))

    def test_5_punctuation_and_stopwords(self):
        """Test 5: A PDF containing punctuation and common stop words."""
        sample_sentence = "Natural Language Processing is a branch of Artificial Intelligence."
        pdf_bytes = create_in_memory_pdf([sample_sentence])

        raw_pages = extract_text_from_pdf(pdf_bytes)
        processed = process_page_data(raw_pages)
        page1 = processed[0]

        tokens = page1["tokens"]
        filtered_tokens = page1["filtered_tokens"]

        # Verify tokens contain words and punctuation
        self.assertIn("Natural", tokens)
        self.assertIn("is", tokens)
        self.assertIn("a", tokens)
        self.assertIn("of", tokens)
        self.assertIn(".", tokens)

        # Verify stop words are removed ('is', 'a', 'of')
        self.assertNotIn("is", filtered_tokens)
        self.assertNotIn("a", filtered_tokens)
        self.assertNotIn("of", filtered_tokens)

        # Verify non-stop-words are retained with exact original casing
        expected_words = ["Natural", "Language", "Processing", "branch", "Artificial", "Intelligence"]
        for word in expected_words:
            self.assertIn(word, filtered_tokens)

        # Verify original text is completely preserved
        self.assertIn(sample_sentence, page1["original_text"])

    def test_text_cleaning_behavior(self):
        """Verify text cleaning normalizes spaces and excessive newlines without altering content."""
        raw_messy = "   Natural   Language\tProcessing\n\n\n\n\nis\t\ta   branch.\x00\xa0  "
        cleaned = clean_text(raw_messy)

        # Null byte removed
        self.assertNotIn("\x00", cleaned)
        # Non-breaking space normalized
        self.assertNotIn("\xa0", cleaned)
        # Multiple spaces/tabs collapsed to single space
        self.assertIn("Natural Language Processing", cleaned)
        self.assertIn("is a branch.", cleaned)
        # Consecutive newlines collapsed to 2 (paragraph break)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("\n\n", cleaned)

    def test_stopword_loading(self):
        """Verify NLTK stopwords load properly."""
        stops = get_stopwords("english")
        self.assertTrue(len(stops) > 100)
        self.assertIn("the", stops)
        self.assertIn("is", stops)
        self.assertIn("in", stops)


if __name__ == "__main__":
    unittest.main()
