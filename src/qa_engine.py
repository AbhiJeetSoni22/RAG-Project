"""Question Answering Engine for Phase 4.

Extractive, retrieval-grounded Question Answering engine that consumes
relevant chunks retrieved by Phase 3, identifies the most relevant sentence(s)
as the concise answer, and preserves full source attribution (source file,
page number, chunk ID, retrieval similarity, and original evidence passage).
"""

from __future__ import annotations

import re
from typing import Any, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import ensure_nltk_resources


class QAError(Exception):
    """Raised when an unrecoverable error occurs in the QA engine."""
    pass


def get_content_words(text: str, language: str = "english") -> set[str]:
    """Extract lowercase alphanumeric content words excluding stop words.

    Args:
        text: Input string.
        language: Language of stop words.

    Returns:
        Set of content words.
    """
    if not text or not text.strip():
        return set()

    ensure_nltk_resources()
    try:
        stops = set(stopwords.words(language))
    except Exception:
        stops = set()

    tokens = word_tokenize(text.lower())
    return {t for t in tokens if t not in stops and any(c.isalnum() for c in t)}


class ExtractiveQAEngine:
    """Extractive Question Answering Engine grounded in retrieved document evidence."""

    def __init__(self, language: str = "english") -> None:
        """Initialize the QA engine.

        Args:
            language: Corpus language for stop-word handling.
        """
        self.language = language
        ensure_nltk_resources()

    def score_sentence(
        self,
        sentence: str,
        question: str,
        chunk_similarity: float = 0.5,
    ) -> float:
        """Compute a relevance score for an individual candidate sentence against a question.

        Combines:
        1. Lexical content-word overlap ratio.
        2. TF-IDF cosine similarity between question and sentence.
        3. Informative cues (definition markers, quantity/numeric markers).
        4. Prior retrieval similarity from the parent chunk.

        Args:
            sentence: Candidate sentence text.
            question: User question text.
            chunk_similarity: Retrieval similarity score of the parent chunk.

        Returns:
            Continuous relevance score float.
        """
        q_words = get_content_words(question, self.language)
        if not q_words:
            return 0.0

        s_words = get_content_words(sentence, self.language)
        if not s_words:
            return 0.0

        # 1. Lexical Overlap Ratio
        overlap = len(q_words & s_words)
        overlap_ratio = overlap / len(q_words)

        # 2. Sentence-level TF-IDF Cosine Similarity
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit([sentence, question])
            q_vec = vec.transform([question])
            s_vec = vec.transform([sentence])
            tfidf_sim = float(cosine_similarity(q_vec, s_vec)[0][0])
        except Exception:
            tfidf_sim = 0.0

        # 3. Informative / Answer-type heuristics
        bonus = 0.0
        q_lower = question.lower()
        s_lower = sentence.lower()

        # Quantity / Numeric matching: "minimum", "how many", "percentage", etc.
        has_quantity_q = any(
            w in q_lower
            for w in ["minimum", "maximum", "how many", "how much", "percent", "percentage", "number", "rate", "fee", "cost"]
        )
        has_quantity_s = bool(re.search(r"\b\d+(\.\d+)?%?\b|\b(percent|percentage)\b", s_lower))
        if has_quantity_q and has_quantity_s:
            bonus += 0.20

        # Definition matching: "what is", "define", "meaning of"
        has_def_q = any(
            q_lower.startswith(prefix)
            for prefix in ["what is", "what are", "define", "meaning of", "explain", "who is", "who are"]
        )
        has_def_s = bool(
            re.search(
                r"\b(is a|are a|is an|are an|is the|are the|refers to|defined as|stands for|means)\b",
                s_lower,
            )
        )
        if has_def_q and has_def_s:
            bonus += 0.15

        # Weighted aggregate score
        total_score = (
            (0.45 * overlap_ratio)
            + (0.30 * tfidf_sim)
            + (0.15 * max(0.0, chunk_similarity))
            + bonus
        )

        return float(total_score)

    def answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        max_sentences: int = 2,
        min_sentence_score: float = 0.15,
    ) -> dict[str, Any]:
        """Extract a grounded answer from the retrieved document chunks.

        Args:
            question: The user's natural language question.
            retrieved_chunks: List of ranked chunk dicts from Phase 3 retriever.
            max_sentences: Maximum number of consecutive sentences to return.
            min_sentence_score: Minimum sentence score required to accept an answer.

        Returns:
            Dictionary containing:
            - answer: Extracted answer string (or None if no answer found)
            - found: Boolean flag indicating if answer was found
            - reason: Explanation if answer could not be found
            - source: Source PDF filename
            - page: 1-indexed page number
            - chunk_id: Unique chunk identifier
            - similarity_score: Retrieval similarity score
            - sentence_score: Sentence relevance score
            - evidence: Full original chunk text (ground truth passage)
        """
        # Validate input question
        if not question or not question.strip():
            return {
                "answer": None,
                "found": False,
                "reason": "Question is empty or contains no text.",
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
            }

        q_words = get_content_words(question, self.language)
        if not q_words:
            return {
                "answer": None,
                "found": False,
                "reason": "Question contains only punctuation or stop words with no searchable keywords.",
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
            }

        # If retrieval returned no chunks (failed threshold or no match)
        if not retrieved_chunks:
            return {
                "answer": None,
                "found": False,
                "reason": "No sufficiently relevant information was found in the uploaded document.",
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
            }

        # Consider top retrieved chunks (up to top 3 for passage analysis)
        candidate_chunks = retrieved_chunks[:3]

        best_candidate: Optional[dict[str, Any]] = None
        best_sentence_score = -1.0
        best_chunk: Optional[dict[str, Any]] = None
        best_s_idx = -1
        best_sentences_list: list[str] = []

        for chunk in candidate_chunks:
            orig_text = chunk.get("original_text", "")
            if not orig_text.strip():
                continue

            chunk_sim = float(chunk.get("similarity_score", 0.0))
            sentences = sent_tokenize(orig_text)
            if not sentences:
                continue

            for idx, sent in enumerate(sentences):
                s_score = self.score_sentence(sent, question, chunk_sim)

                # Prioritize sentence with higher score
                if s_score > best_sentence_score:
                    best_sentence_score = s_score
                    best_chunk = chunk
                    best_s_idx = idx
                    best_sentences_list = sentences

        # Check if the best sentence meets the minimum confidence threshold
        if (
            best_chunk is None
            or best_sentence_score < min_sentence_score
            or best_s_idx < 0
        ):
            return {
                "answer": None,
                "found": False,
                "reason": "No sufficiently relevant passage was retrieved from the uploaded document.",
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
            }

        # Formulate answer span
        chosen_sentences = [best_sentences_list[best_s_idx]]

        # Multi-sentence expansion: check immediate neighbor if max_sentences > 1
        if max_sentences > 1 and len(best_sentences_list) > 1:
            next_idx = best_s_idx + 1
            if next_idx < len(best_sentences_list):
                next_sent = best_sentences_list[next_idx]
                next_score = self.score_sentence(
                    next_sent, question, float(best_chunk.get("similarity_score", 0.0))
                )
                # Include next sentence if it has strong relevance (at least 60% of top score)
                if next_score >= max(min_sentence_score, 0.60 * best_sentence_score):
                    chosen_sentences.append(next_sent)
            elif best_s_idx > 0:
                prev_idx = best_s_idx - 1
                prev_sent = best_sentences_list[prev_idx]
                prev_score = self.score_sentence(
                    prev_sent, question, float(best_chunk.get("similarity_score", 0.0))
                )
                if prev_score >= max(min_sentence_score, 0.60 * best_sentence_score):
                    chosen_sentences.insert(0, prev_sent)

        answer_text = " ".join(s.strip() for s in chosen_sentences).strip()

        return {
            "answer": answer_text,
            "found": True,
            "reason": None,
            "source": best_chunk.get("source"),
            "page": best_chunk.get("page"),
            "chunk_id": best_chunk.get("chunk_id"),
            "similarity_score": round(float(best_chunk.get("similarity_score", 0.0)), 4),
            "sentence_score": round(best_sentence_score, 4),
            "evidence": best_chunk.get("original_text"),
        }


def answer_question(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_sentences: int = 2,
    min_sentence_score: float = 0.15,
) -> dict[str, Any]:
    """Functional interface for extractive question answering.

    Args:
        question: User's question.
        retrieved_chunks: List of chunk dictionaries from Phase 3 retriever.
        max_sentences: Maximum number of sentences in the extracted answer.
        min_sentence_score: Minimum sentence score threshold.

    Returns:
        Structured answer dictionary.
    """
    engine = ExtractiveQAEngine()
    return engine.answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
    )
