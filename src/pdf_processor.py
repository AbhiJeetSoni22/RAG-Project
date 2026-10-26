"""PDF Processing module for extracting text page-by-page using PyMuPDF (fitz).

Provides robust text extraction with encryption detection, multi-column block ordering,
conservative recurring header/footer suppression, and deterministic resource safety.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Union
import pymupdf as fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when PDF extraction or parsing fails."""
    pass


def _filter_recurring_headers_footers(pages_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservatively remove repeated headers and footers that pollute multi-page documents.

    Only applies if the document has 3 or more pages, the repeated line appears on >= 80%
    of pages with content, and is between 3 and 120 characters long.
    """
    total_pages = len(pages_data)
    if total_pages < 3:
        return pages_data

    # Collect first and last lines of pages with content
    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}
    pages_with_content = 0

    for p in pages_data:
        text = p.get("original_text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:  # Only consider pages with at least 2 lines so single-line pages are never wiped
            pages_with_content += 1
            first_line = lines[0]
            last_line = lines[-1]
            if 3 <= len(first_line) <= 120:
                header_counts[first_line] = header_counts.get(first_line, 0) + 1
            if 3 <= len(last_line) <= 120:
                footer_counts[last_line] = footer_counts.get(last_line, 0) + 1

    if pages_with_content < 3:
        return pages_data

    threshold = 0.80 * pages_with_content
    recurring_headers = {line for line, count in header_counts.items() if count >= threshold}
    recurring_footers = {line for line, count in footer_counts.items() if count >= threshold}

    if not recurring_headers and not recurring_footers:
        return pages_data

    cleaned_pages: list[dict[str, Any]] = []
    for p in pages_data:
        text = p.get("original_text", "")
        lines = text.splitlines()
        filtered_lines: list[str] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            # If top line matches recurring header, skip it
            if stripped in recurring_headers and idx < 3:
                continue
            # If bottom line matches recurring footer, skip it
            if stripped in recurring_footers and idx >= len(lines) - 3:
                continue
            filtered_lines.append(line)
        cleaned_pages.append({
            "page": p["page"],
            "original_text": "\n".join(filtered_lines),
        })

    return cleaned_pages


def extract_text_from_pdf(
    file_source: Union[str, bytes, io.BytesIO]
) -> list[dict[str, Any]]:
    """Extract text from a PDF file page by page.

    Args:
        file_source: File path (str), raw PDF bytes (bytes), or a stream (BytesIO).

    Returns:
        A list of dictionaries with structure:
        [
            {
                "page": 1,
                "original_text": "..."
            },
            ...
        ]

    Raises:
        PDFProcessingError: If the file is invalid, corrupted, empty, encrypted, or unreadable.
    """
    if file_source is None:
        raise PDFProcessingError("No PDF file provided.")

    # Check for empty byte buffers
    if isinstance(file_source, (bytes, bytearray)):
        if len(file_source) == 0:
            raise PDFProcessingError("Uploaded file is empty (0 bytes).")
    elif isinstance(file_source, io.BytesIO):
        buffer_val = file_source.getvalue()
        if len(buffer_val) == 0:
            raise PDFProcessingError("Uploaded file buffer is empty (0 bytes).")
        file_source.seek(0)

    doc = None
    try:
        if isinstance(file_source, str):
            doc = fitz.open(file_source)
        elif isinstance(file_source, (bytes, bytearray)):
            doc = fitz.open(stream=file_source, filetype="pdf")
        elif hasattr(file_source, "read"):
            data = file_source.read()
            if len(data) == 0:
                raise PDFProcessingError("Uploaded file stream is empty (0 bytes).")
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise PDFProcessingError(f"Unsupported file source type: {type(file_source)}")
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open PDF document: {exc}") from exc

    try:
        # Detect encrypted or password-protected PDFs
        if getattr(doc, "is_encrypted", False):
            if getattr(doc, "needs_pass", False) or not doc.authenticate(""):
                raise PDFProcessingError(
                    "The PDF file is password-protected or encrypted. Please provide an unencrypted PDF document."
                )

        total_pages = len(doc)
        if total_pages == 0:
            raise PDFProcessingError("PDF contains 0 pages.")

        pages_data: list[dict[str, Any]] = []

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            text = ""
            try:
                # get_text("blocks") returns tuples: (x0, y0, x1, y1, text, block_no, block_type)
                # Sort blocks top-to-bottom, left-to-right for consistent multi-column flow
                blocks = page.get_text("blocks")
                text_blocks = [
                    b[4] for b in blocks
                    if len(b) > 4 and isinstance(b[4], str) and b[4].strip()
                ]
                text = "\n".join(text_blocks)
            except Exception:
                text = ""

            # Fallback to plain get_text if blocks are empty
            if not text.strip():
                text = page.get_text() or ""

            pages_data.append({
                "page": page_idx + 1,  # 1-indexed page number
                "original_text": text
            })

        # Apply conservative recurring header/footer removal
        return _filter_recurring_headers_footers(pages_data)

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Error extracting text from PDF: {exc}") from exc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
