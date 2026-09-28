"""Text Preprocessing module for cleaning, tokenization, and stop-word removal using NLTK."""

from __future__ import annotations

import re
from typing import Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextProcessingError(Exception):
    """Raised when text processing or NLTK operations fail."""
    pass


def ensure_nltk_resources() -> None:
    """Ensure required NLTK datasets and tokenizers are downloaded.

    Downloads 'punkt', 'punkt_tab', and 'stopwords' if not already available.
    """
    required_packages = ["punkt", "punkt_tab", "stopwords"]
    for pkg in required_packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except (LookupError, AttributeError):
            try:
                nltk.download(pkg, quiet=True)
            except Exception as exc:
                raise TextProcessingError(
                    f"Failed to download required NLTK resource '{pkg}': {exc}"
                ) from exc


def get_stopwords(language: str = "english") -> set[str]:
    """Retrieve the set of stop words for a given language.

    Args:
        language: Language for the stopwords corpus (default: 'english').

    Returns:
        A set of lowercase stop words.
    """
    ensure_nltk_resources()
    try:
        return set(stopwords.words(language))
    except Exception as exc:
        raise TextProcessingError(f"Error loading stopwords for '{language}': {exc}") from exc


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace, newlines, and non-printable characters.

    Keeps the original casing, vocabulary, and sentence structure intact.

    Args:
        text: The raw text string to clean.

    Returns:
        The cleaned text string.
    """
    if not text:
        return ""

    # Replace null bytes, soft hyphens, and zero-width spaces
    cleaned = text.replace("\x00", "").replace("\xad", "").replace("\u200b", "")

    # Normalize non-breaking spaces and special horizontal whitespace to standard space
    cleaned = cleaned.replace("\xa0", " ").replace("\u202f", " ")

    # Collapse multiple horizontal spaces and tabs into a single space (preserving newlines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Clean individual lines: strip trailing/leading spaces on each line
    lines = [line.strip() for line in cleaned.splitlines()]

    # Collapse 3 or more consecutive empty lines down to a maximum of 2 newlines (paragraph break)
    cleaned_lines: list[str] = []
    consecutive_empty = 0
    for line in lines:
        if not line:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                cleaned_lines.append("")
        else:
            consecutive_empty = 0
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    return result


def tokenize_text(text: str) -> list[str]:
    """Tokenize a text string into individual word and punctuation tokens using NLTK.

    Args:
        text: The text string to tokenize.

    Returns:
        A list of string tokens.
    """
    if not text or not text.strip():
        return []

    ensure_nltk_resources()
    try:
        return word_tokenize(text)
    except Exception as exc:
        raise TextProcessingError(f"Tokenization failed: {exc}") from exc


def remove_stopwords(
    tokens: list[str],
    language: str = "english",
    remove_punctuation: bool = False
) -> list[str]:
    """Remove stop words from a list of tokens while preserving token casing.

    Args:
        tokens: List of string tokens.
        language: Language of stop words (default: 'english').
        remove_punctuation: If True, also filters out punctuation-only tokens.

    Returns:
        A list of filtered tokens.
    """
    if not tokens:
        return []

    stop_words = get_stopwords(language)
    filtered: list[str] = []

    for token in tokens:
        # Check lowercase version against stopwords
        if token.lower() in stop_words:
            continue

        # If requested, filter out tokens that contain no alphanumeric characters (punctuation only)
        if remove_punctuation and not any(char.isalnum() for char in token):
            continue

        filtered.append(token)

    return filtered


def process_page_data(
    raw_pages: list[dict[str, Any]],
    remove_punctuation_in_filter: bool = False
) -> list[dict[str, Any]]:
    """Process raw extracted pages through cleaning, tokenization, and stop-word removal.

    Args:
        raw_pages: List of dictionaries with keys 'page' and 'original_text'.
        remove_punctuation_in_filter: Whether to omit punctuation-only tokens in filtered_tokens.

    Returns:
        Structured page data ready for Phase 2:
        [
            {
                "page": 1,
                "original_text": "...",
                "cleaned_text": "...",
                "tokens": [...],
                "filtered_tokens": [...]
            },
            ...
        ]
    """
    processed_pages: list[dict[str, Any]] = []

    for item in raw_pages:
        page_num = item["page"]
        orig_text = item.get("original_text", "")

        cleaned = clean_text(orig_text)
        tokens = tokenize_text(cleaned)
        filtered = remove_stopwords(
            tokens,
            remove_punctuation=remove_punctuation_in_filter
        )

        processed_pages.append({
            "page": page_num,
            "original_text": orig_text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "filtered_tokens": filtered,
        })

    return processed_pages


def calculate_document_stats(processed_pages: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate aggregate statistics across processed pages.

    Returns:
        Dict with total_pages, pages_with_text, total_characters, total_tokens, total_filtered_tokens.
    """
    total_pages = len(processed_pages)
    pages_with_text = sum(1 for p in processed_pages if p["cleaned_text"].strip())
    total_characters = sum(len(p["original_text"]) for p in processed_pages)
    total_tokens = sum(len(p["tokens"]) for p in processed_pages)
    total_filtered_tokens = sum(len(p["filtered_tokens"]) for p in processed_pages)

    return {
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "total_filtered_tokens": total_filtered_tokens,
    }
