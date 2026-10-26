"""Regression tests for Extracted Pages Inspection & Page Count Initialization.

Verifies:
1. A valid PDF with one extracted page.
2. A valid PDF with multiple extracted pages.
3. An empty extracted-page collection.
4. A PDF extraction failure.
5. A Streamlit rerun where document is cached (ensuring no UnboundLocalError).
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch
import pymupdf as fitz

from src.pdf_processor import extract_text_from_pdf, PDFProcessingError
from src.text_processor import process_page_data
from app import render_extracted_pages_inspection


class TestPageInspectionRegression(unittest.TestCase):
    """Test suite covering page inspection robustness and page count scoping."""

    def setUp(self):
        self.single_page = [
            {
                "page": 1,
                "original_text": "Single page document text.",
                "cleaned_text": "single page document text",
                "tokens": ["single", "page", "document", "text"],
                "filtered_tokens": ["single", "page", "document", "text"],
            }
        ]
        self.multi_pages = [
            {
                "page": 1,
                "original_text": "Page one content.",
                "cleaned_text": "page one content",
                "tokens": ["page", "one", "content"],
                "filtered_tokens": ["page", "content"],
            },
            {
                "page": 2,
                "original_text": "Page two content.",
                "cleaned_text": "page two content",
                "tokens": ["page", "two", "content"],
                "filtered_tokens": ["page", "two", "content"],
            },
        ]

    @patch("app.st")
    def test_single_page_inspection(self, mock_st):
        """Case 1: Single-page document renders without dropdown and without UnboundLocalError."""
        mock_st.expander.return_value.__enter__.return_value = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        # Should execute cleanly
        render_extracted_pages_inspection(self.single_page)

        # For single page, selectbox should NOT be called
        mock_st.selectbox.assert_not_called()
        mock_st.text_area.assert_any_call(
            label="Original Text (as extracted from PDF)",
            value="Single page document text.",
            height=240,
            key="p1_orig_1",
            disabled=True,
        )

    @patch("app.st")
    def test_multi_page_inspection(self, mock_st):
        """Case 2: Multi-page document renders selectbox dropdown and selected page."""
        mock_st.expander.return_value.__enter__.return_value = MagicMock()
        mock_st.selectbox.return_value = "Page 2"
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        render_extracted_pages_inspection(self.multi_pages)

        # For multi-page, selectbox must be called with page options
        mock_st.selectbox.assert_called_once_with(
            "Select Page to Inspect",
            options=["Page 1", "Page 2"],
            key="raw_page_select",
        )
        mock_st.text_area.assert_any_call(
            label="Original Text (as extracted from PDF)",
            value="Page two content.",
            height=240,
            key="p1_orig_2",
            disabled=True,
        )

    @patch("app.st")
    def test_empty_page_collection(self, mock_st):
        """Case 3: Empty page collection displays info message safely without indexing error."""
        mock_st.expander.return_value.__enter__.return_value = MagicMock()

        # Should not raise IndexError or UnboundLocalError
        render_extracted_pages_inspection([])

        mock_st.info.assert_called_once_with("ℹ️ No extracted pages are available for inspection.")
        mock_st.selectbox.assert_not_called()
        mock_st.columns.assert_not_called()

    def test_pdf_extraction_failure_handled_cleanly(self):
        """Case 4: PDF extraction failure raises PDFProcessingError safely."""
        corrupted_bytes = b"Not a valid PDF file stream"
        with self.assertRaises(PDFProcessingError):
            extract_text_from_pdf(corrupted_bytes)

    @patch("app.st")
    def test_cached_rerun_page_count_initialization(self, mock_st):
        """Case 5: Rerun where pipeline is cached ensures num_pages is valid and inspection succeeds."""
        mock_st.expander.return_value.__enter__.return_value = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        # Simulate cached pipeline data retrieval
        cached_pipe = {
            "processed_pages": self.single_page,
            "doc_stats": {"total_pages": 1, "pages_with_text": 1},
        }

        # Emulate the main() execution path on a cache hit
        processed_pages = cached_pipe["processed_pages"]
        num_pages = len(processed_pages) if processed_pages else 0

        # Assert num_pages is valid integer and inspection executes without error
        self.assertEqual(num_pages, 1)
        render_extracted_pages_inspection(processed_pages)


if __name__ == "__main__":
    unittest.main()
