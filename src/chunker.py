"""Chunking and Retrieval Data Preparation module for Phase 2.

This module partitions page-level extracted text into paragraph-aware,
sentence-bounded chunks with optional overlap, preserving full metadata
(page numbers, source document, chunk IDs, original text, and cleaned text)
for downstream vectorization and similarity retrieval in Phase 3.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from src.text_processor import ensure_nltk_resources

# Default configuration parameters (configurable in one place)
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_MIN_CHUNK_SIZE: int = 20

# Common interrogatives to preserve in retrieval text for QA query-document matching
QUESTION_WORDS: set[str] = {
    "what", "which", "who", "whom", "where", "when", "why", "how"
}


class ChunkingError(Exception):
    """Raised when text chunking or validation fails."""
    pass


def get_retrieval_stopwords(language: str = "english") -> set[str]:
    """Return stop words for retrieval, retaining interrogatives for QA matching.

    Args:
        language: Language of stopwords (default: 'english').

    Returns:
        Set of stop words excluding question words.
    """
    ensure_nltk_resources()
    try:
        stops = set(stopwords.words(language))
        return stops - QUESTION_WORDS
    except Exception as exc:
        raise ChunkingError(f"Error loading stopwords for retrieval: {exc}") from exc


def clean_chunk_text(text: str, language: str = "english") -> str:
    """Prepare cleaned text for retrieval/vectorization.

    Converts text to lowercase, removes punctuation and standard stop-words
    (while retaining question words), and normalizes spacing. Falls back to all
    lowercased alphanumeric tokens if all words are stop-words.

    Args:
        text: Original chunk text string.
        language: Language for stop-word filtering.

    Returns:
        Cleaned, normalized string ready for downstream vectorization.
    """
    if not text or not text.strip():
        return ""

    # Normalize special characters and spacing
    text_clean = text.replace("\x00", "").replace("\xad", "").replace("\u200b", "")
    text_clean = text_clean.replace("\xa0", " ").replace("\u202f", " ")
    text_clean = re.sub(r"[ \t]+", " ", text_clean)

    ensure_nltk_resources()
    tokens = word_tokenize(text_clean.lower())
    stops = get_retrieval_stopwords(language)

    # Filter out stopwords and non-alphanumeric punctuation tokens
    filtered = [t for t in tokens if t not in stops and any(c.isalnum() for c in t)]

    # Fallback if text consists solely of stop words (e.g., 'To be or not to be')
    if not filtered:
        filtered = [t for t in tokens if any(c.isalnum() for c in t)]

    return " ".join(filtered)


def get_overlap_prefix(text: str, overlap_size: int) -> str:
    """Extract an overlap prefix from the tail of text at clean word boundaries.

    Args:
        text: The source chunk text.
        overlap_size: Approximate number of characters to retain for overlap.

    Returns:
        Extracted overlap prefix string.
    """
    if overlap_size <= 0 or not text:
        return ""

    text = text.strip()
    if len(text) <= overlap_size:
        return text

    tail = text[-overlap_size:]
    # Snap forward to next whitespace to avoid breaking words in the middle
    space_pos = tail.find(" ")
    if space_pos != -1 and space_pos < len(tail) - 1:
        prefix = tail[space_pos + 1:].strip()
    else:
        prefix = tail.strip()

    # Strip awkward leading punctuation or symbols
    prefix = prefix.lstrip(".,;:?!-—()[]{}'\" \t\n\r")
    return prefix.strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[str]:
    """Split a page's text into chunks respecting the hierarchy: Paragraph -> Sentence -> Word.

    Ensures:
    1. Text boundaries are preserved (paragraphs and sentences preferred).
    2. Words are never broken in half (unless an individual word exceeds chunk_size).
    3. Overlap between adjacent chunks is maintained without excessive duplication.
    4. Meaningless whitespace and empty chunks are filtered out.

    Args:
        text: The text to chunk.
        chunk_size: Maximum character size per chunk (default 500).
        chunk_overlap: Overlap character size between consecutive chunks (default 50).
        min_chunk_size: Minimum character length for valid chunks (default 20).

    Returns:
        List of chunk strings.

    Raises:
        ChunkingError: If chunk configuration parameters are invalid.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ChunkingError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    ensure_nltk_resources()

    # Clean whitespace and control characters
    cleaned = text.replace("\x00", "").replace("\xad", "").replace("\u200b", "")
    cleaned = cleaned.replace("\xa0", " ").replace("\u202f", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 1. Paragraph boundary splitting
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not raw_paras:
        return []

    # 2. Deconstruct paragraphs into atomic units (sentences, or sub-sentence word groups)
    units: list[tuple[str, bool]] = []  # (unit_text, is_new_paragraph)

    for p_idx, raw_p in enumerate(raw_paras):
        # Unwrap wrapped lines within the same paragraph into a single line
        lines = [line.strip() for line in raw_p.splitlines() if line.strip()]
        p_text = " ".join(lines)
        if not p_text:
            continue

        # Sentence tokenization
        sentences = sent_tokenize(p_text)
        if not sentences:
            sentences = [p_text]

        for s_idx, sent in enumerate(sentences):
            is_new_p = (p_idx > 0 and s_idx == 0)

            # If sentence fits in chunk_size, keep as single unit
            if len(sent) <= chunk_size:
                units.append((sent, is_new_p))
            else:
                # 3. Word-level fallback for oversized sentences
                words = sent.split()
                current_w: list[str] = []
                cur_w_len = 0

                for w in words:
                    w_len = len(w)
                    if w_len > chunk_size:
                        # Flush accumulated words
                        if current_w:
                            units.append((" ".join(current_w), is_new_p and len(units) == 0))
                            current_w = []
                            cur_w_len = 0
                        # 4. Character-level fallback for giant single word (e.g. continuous symbols/URL)
                        for start in range(0, w_len, chunk_size):
                            units.append((w[start:start + chunk_size], is_new_p and len(units) == 0))
                    elif cur_w_len + (1 if current_w else 0) + w_len <= chunk_size:
                        current_w.append(w)
                        cur_w_len += (1 if cur_w_len > 0 else 0) + w_len
                    else:
                        units.append((" ".join(current_w), is_new_p and len(units) == 0))
                        current_w = [w]
                        cur_w_len = w_len

                if current_w:
                    units.append((" ".join(current_w), is_new_p and len(units) == 0))

    if not units:
        return []

    # 5. Assemble units into chunks with target size and overlap
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    overlap_prefix = ""

    for unit_text, is_new_p in units:
        # Determine separator
        if not current_parts:
            if overlap_prefix:
                current_parts.append(overlap_prefix)
                current_len = len(overlap_prefix)
                sep = " "
            else:
                sep = ""
        else:
            sep = "\n\n" if is_new_p else " "

        added_len = len(sep) + len(unit_text)

        if current_len + added_len <= chunk_size:
            if sep:
                current_parts.append(sep)
            current_parts.append(unit_text)
            current_len += added_len
        else:
            # Current chunk has reached capacity, emit it
            if current_parts:
                chunk_str = "".join(current_parts).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                    overlap_prefix = get_overlap_prefix(chunk_str, chunk_overlap)
                else:
                    overlap_prefix = ""

            current_parts = []
            current_len = 0

            # Adapt overlap_prefix so that the new unit_text fits within chunk_size
            target_overlap = overlap_prefix
            if target_overlap and (len(target_overlap) + 1 + len(unit_text) > chunk_size):
                available_space = chunk_size - len(unit_text) - 1
                if available_space >= 10:
                    target_overlap = get_overlap_prefix(chunks[-1], available_space)
                else:
                    target_overlap = ""

            if target_overlap:
                current_parts = [target_overlap, " ", unit_text]
                current_len = len(target_overlap) + 1 + len(unit_text)
            else:
                current_parts = [unit_text]
                current_len = len(unit_text)

    # Emit final chunk
    if current_parts:
        chunk_str = "".join(current_parts).strip()
        if chunk_str:
            chunks.append(chunk_str)

    # Post-process: Filter out chunks that contain no alphanumeric characters
    valid_chunks: list[str] = []
    for c in chunks:
        c_stripped = c.strip()
        if any(char.isalnum() for char in c_stripped):
            valid_chunks.append(c_stripped)

    # Merge small trailing orphan fragment into previous chunk if feasible
    if len(valid_chunks) > 1 and len(valid_chunks[-1]) < min_chunk_size:
        last = valid_chunks.pop()
        prev = valid_chunks[-1]
        if len(prev) + 1 + len(last) <= int(chunk_size * 1.2):
            valid_chunks[-1] = f"{prev} {last}"
        else:
            valid_chunks.append(last)

    return valid_chunks


def chunk_page(
    page_data: dict[str, Any],
    source: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Generate structured chunks for a single page of text.

    Args:
        page_data: Dictionary containing 'page', 'original_text', and optionally 'cleaned_text'.
        source: Name or identifier of the source document.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap size in characters between consecutive chunks.
        min_chunk_size: Minimum characters required for a chunk.

    Returns:
        List of chunk dictionaries with complete metadata:
        [
            {
                "chunk_id": "page_1_chunk_1",
                "source": "NLP_Syllabus.pdf",
                "page": 1,
                "original_text": "...",
                "cleaned_text": "...",
                "char_count": 421,
                "word_count": 65,
            },
            ...
        ]
    """
    page_num = page_data.get("page", 1)
    text = page_data.get("cleaned_text") or page_data.get("original_text", "")

    chunk_strings = split_text_into_chunks(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    page_chunks: list[dict[str, Any]] = []
    for idx, c_text in enumerate(chunk_strings, start=1):
        cleaned_c_text = clean_chunk_text(c_text)
        page_chunks.append({
            "chunk_id": f"page_{page_num}_chunk_{idx}",
            "source": source,
            "page": page_num,
            "original_text": c_text,
            "cleaned_text": cleaned_c_text,
            "char_count": len(c_text),
            "word_count": len(c_text.split()),
        })

    return page_chunks


def chunk_document(
    processed_pages: list[dict[str, Any]],
    source: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Convert page-level processed text into a collection of retrieval-ready chunks.

    Args:
        processed_pages: List of page dictionaries from Phase 1.
        source: Document name (e.g. 'NLP_Syllabus.pdf').
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between chunks on the same page.
        min_chunk_size: Minimum character length for valid chunks.

    Returns:
        Flat list of chunk dictionaries ready for Phase 3 retrieval.
    """
    all_chunks: list[dict[str, Any]] = []

    for page_data in processed_pages:
        page_chunks = chunk_page(
            page_data=page_data,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def validate_chunks(chunks: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Validate the integrity and completeness of a chunk collection.

    Verifies:
    - All chunks are valid dictionaries with required keys.
    - Chunk IDs are unique across the collection.
    - Page numbers are positive integers.
    - Original text and cleaned text are non-empty.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        A tuple of (is_valid: bool, issues: list[str]).
    """
    issues: list[str] = []
    seen_ids: set[str] = set()

    if not isinstance(chunks, list):
        return False, ["Chunks collection must be a list."]

    required_keys = {"chunk_id", "source", "page", "original_text", "cleaned_text"}

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            issues.append(f"Chunk at index {idx} is not a dictionary.")
            continue

        missing = required_keys - set(chunk.keys())
        if missing:
            issues.append(f"Chunk at index {idx} is missing required keys: {sorted(missing)}.")

        c_id = chunk.get("chunk_id", "")
        if not c_id:
            issues.append(f"Chunk at index {idx} has an empty chunk_id.")
        elif c_id in seen_ids:
            issues.append(f"Duplicate chunk_id '{c_id}' detected at index {idx}.")
        else:
            seen_ids.add(c_id)

        page_val = chunk.get("page")
        if not isinstance(page_val, int) or page_val < 1:
            issues.append(f"Chunk '{c_id}' has invalid page number: {page_val}.")

        orig = chunk.get("original_text", "")
        if not orig or not orig.strip():
            issues.append(f"Chunk '{c_id}' has empty original_text.")

        cleaned = chunk.get("cleaned_text", "")
        if not cleaned or not cleaned.strip():
            issues.append(f"Chunk '{c_id}' has empty cleaned_text.")

    return (len(issues) == 0, issues)


def calculate_chunk_stats(
    chunks: list[dict[str, Any]],
    processed_pages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Calculate aggregate chunk and document statistics for inspection and UI display.

    Args:
        chunks: List of chunk dictionaries.
        processed_pages: Optional list of processed page dictionaries from Phase 1.

    Returns:
        Dictionary containing:
        - total_pages
        - pages_with_text
        - pages_with_chunks
        - total_chunks
        - avg_chunk_size
        - min_chunk_size
        - max_chunk_size
        - avg_cleaned_chunk_size
        - chunks_per_page
        - total_chunk_characters
    """
    total_chunks = len(chunks)

    # Page-level counts
    if processed_pages is not None:
        total_pages = len(processed_pages)
        pages_with_text = sum(
            1 for p in processed_pages
            if (p.get("cleaned_text") or p.get("original_text", "")).strip()
        )
    else:
        unique_pages = {c["page"] for c in chunks if "page" in c}
        total_pages = len(unique_pages)
        pages_with_text = total_pages

    # Chunks per page mapping
    chunks_per_page: dict[int, int] = {}
    for c in chunks:
        pg = c.get("page", 1)
        chunks_per_page[pg] = chunks_per_page.get(pg, 0) + 1

    pages_with_chunks = len(chunks_per_page)

    if total_chunks == 0:
        return {
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "pages_with_chunks": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "avg_cleaned_chunk_size": 0,
            "chunks_per_page": {},
            "total_chunk_characters": 0,
        }

    orig_lengths = [len(c.get("original_text", "")) for c in chunks]
    cleaned_lengths = [len(c.get("cleaned_text", "")) for c in chunks]

    return {
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "pages_with_chunks": pages_with_chunks,
        "total_chunks": total_chunks,
        "avg_chunk_size": round(sum(orig_lengths) / total_chunks, 1),
        "min_chunk_size": min(orig_lengths),
        "max_chunk_size": max(orig_lengths),
        "avg_cleaned_chunk_size": round(sum(cleaned_lengths) / total_chunks, 1),
        "chunks_per_page": chunks_per_page,
        "total_chunk_characters": sum(orig_lengths),
    }
