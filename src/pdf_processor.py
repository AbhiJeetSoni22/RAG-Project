"""PDF Processing module for extracting text page-by-page using PyMuPDF (fitz)."""

from __future__ import annotations

import io
from typing import Any, Union
import pymupdf as fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Raised when PDF extraction or parsing fails."""
    pass


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
        PDFProcessingError: If the file is invalid, corrupted, empty, or unreadable.
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
        total_pages = len(doc)
        if total_pages == 0:
            raise PDFProcessingError("PDF contains 0 pages.")

        pages_data: list[dict[str, Any]] = []

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            text = page.get_text() or ""
            pages_data.append({
                "page": page_idx + 1,  # 1-indexed page number
                "original_text": text
            })

        return pages_data
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Error extracting text from PDF: {exc}") from exc
    finally:
        try:
            doc.close()
        except Exception:
            pass
