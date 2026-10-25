"""Question Answering Engine for Phase 4.

Extractive, retrieval-grounded Question Answering engine that consumes
relevant chunks retrieved by Phase 3, identifies the most relevant sentence(s)
as the concise answer, and preserves full source attribution (source file,
page number, chunk ID, retrieval similarity, and original evidence passage).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import ensure_nltk_resources

logger = logging.getLogger(__name__)

FALLBACK_NO_ANSWER_MSG: str = (
    "I couldn't find a sufficiently relevant answer in this document. "
    "Try rephrasing your question or check whether the document contains the information."
)


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


def strip_leading_heading(text: str) -> str:
    """Strip prepended section headings or chapter numbers from an extracted answer sentence.

    Example:
        '2.1 What Machine Learning Is Machine learning is...' -> 'Machine learning is...'

    Args:
        text: Extracted sentence text.

    Returns:
        Cleaned sentence text without leading heading noise.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Match section number and title preceding the start of a standard capitalized sentence
    match = re.match(
        r"^(?:(?:Chapter|Section|\b)\s*\d+(\.\d+)*\s+[A-Za-z0-9\s:,\-]{2,50}?\s+)(?=[A-Z][a-z])",
        cleaned,
    )
    if match:
        cleaned = cleaned[match.end():].strip()
    return cleaned


class ExtractiveQAEngine:
    """Extractive Question Answering Engine grounded in retrieved document evidence."""

    def __init__(self, language: str = "english") -> None:
        """Initialize the QA engine.

        Args:
            language: Corpus language for stop-word handling.
        """
        self.language = language
        self.stemmer = PorterStemmer()
        ensure_nltk_resources()

    @staticmethod
    def _normalize_stem(stem: str) -> str:
        """Normalize specific English morphological quirks without loose substring matching."""
        if stem in ("percentag", "percentage", "percent"):
            return "percent"
        return stem

    def score_sentence(
        self,
        sentence: str,
        question: str,
        chunk_similarity: float = 0.5,
    ) -> float:
        """Compute a relevance score for an individual candidate sentence against a question.

        Combines:
        1. Lexical content-word overlap ratio with exact morphological stemming.
        2. Sentence-level TF-IDF cosine similarity excluding generic stop-words.
        3. Strict topical gating: candidate must share substantive keywords or stems.
        4. Informative cues (definition markers, quantity/numeric markers) strictly gated
           on topic match and target entity alignment.
        5. Prior retrieval similarity from the parent chunk.

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

        # Exact stems for morphological flexibility (e.g. required vs requirement, attend vs attendance)
        q_stems = {self._normalize_stem(self.stemmer.stem(w)) for w in q_words}
        s_stems = {self._normalize_stem(self.stemmer.stem(w)) for w in s_words}

        # 1. Lexical Overlap Ratio (exact word match and exact stem match)
        raw_overlap = len(q_words & s_words)
        raw_overlap_ratio = raw_overlap / len(q_words)

        stem_overlap = len(q_stems & s_stems)
        stem_overlap_ratio = stem_overlap / len(q_stems)
        effective_overlap = max(raw_overlap_ratio, stem_overlap_ratio)

        # 2. Sentence-level TF-IDF Cosine Similarity with stop words removed
        # Generic words like 'what', 'is', 'the' must NEVER generate sentence similarity!
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit([sentence, question])
            q_vec = vec.transform([question])
            s_vec = vec.transform([sentence])
            tfidf_sim = float(cosine_similarity(q_vec, s_vec)[0][0])
        except Exception:
            tfidf_sim = 0.0

        # 3. Strict Topical Grounding Gate:
        # A candidate sentence MUST have at least one substantive matching content word or stem.
        # Generic stop words (e.g. 'what', 'is', 'the') never satisfy this gate.
        has_topic_match = (stem_overlap > 0) or (raw_overlap > 0)

        if not has_topic_match:
            # Off-topic sentence: bounded strictly below min_sentence_score (0.15)
            return float(min(0.08, 0.08 * max(0.0, chunk_similarity)))

        # 4. Informative / Answer-type heuristics (Strictly GATED on topical relevance)
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
        # Definition bonus applies ONLY if the sentence is defining one of the question's target stems
        defines_target = has_def_s and bool(q_stems & s_stems)
        if has_def_q and defines_target:
            bonus += 0.15

        # Weighted aggregate score
        total_score = (
            (0.45 * effective_overlap)
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
            logger.info("QA engine abstained: retrieved_chunks is empty for query '%s'", question)
            return {
                "answer": None,
                "found": False,
                "reason": (
                    f"{FALLBACK_NO_ANSWER_MSG} "
                    "No sufficiently relevant information was found in the uploaded document."
                ),
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
                "diagnostics": {
                    "question": question,
                    "q_words": sorted(q_words),
                    "retrieved_chunk_count": 0,
                    "abstention_reason": "No chunks retrieved above similarity cutoff.",
                },
            }

        # Consider top retrieved chunks (up to top 3 for passage analysis)
        candidate_chunks = retrieved_chunks[:3]

        best_candidate: Optional[dict[str, Any]] = None
        best_sentence_score = -1.0
        best_chunk: Optional[dict[str, Any]] = None
        best_s_idx = -1
        best_sentences_list: list[str] = []

        candidate_diagnostics: list[dict[str, Any]] = []

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

                candidate_diagnostics.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "sentence_idx": idx,
                    "sentence_preview": sent[:60] + ("..." if len(sent) > 60 else ""),
                    "score": round(s_score, 4),
                    "chunk_similarity": round(chunk_sim, 4),
                })

                # Prioritize sentence with higher score
                if s_score > best_sentence_score:
                    best_sentence_score = s_score
                    best_chunk = chunk
                    best_s_idx = idx
                    best_sentences_list = sentences

        diagnostics_payload = {
            "question": question,
            "q_words": sorted(q_words),
            "retrieved_chunk_count": len(retrieved_chunks),
            "candidate_chunk_ids": [c.get("chunk_id") for c in candidate_chunks],
            "best_sentence_score": round(max(0.0, best_sentence_score), 4),
            "min_sentence_score_threshold": min_sentence_score,
            "evaluated_sentences_count": len(candidate_diagnostics),
            "top_sentence_candidates": sorted(
                candidate_diagnostics, key=lambda x: x["score"], reverse=True
            )[:5],
        }

        # Check if the best sentence meets the minimum confidence threshold
        if (
            best_chunk is None
            or best_sentence_score < min_sentence_score
            or best_s_idx < 0
        ):
            logger.info(
                "QA engine abstained: best_sentence_score (%.4f) < min_sentence_score (%.4f) for query '%s'",
                best_sentence_score,
                min_sentence_score,
                question,
            )
            diagnostics_payload["abstention_reason"] = (
                f"Best sentence score ({max(0.0, best_sentence_score):.4f}) was below acceptance threshold ({min_sentence_score:.4f})."
            )
            return {
                "answer": None,
                "found": False,
                "reason": (
                    f"{FALLBACK_NO_ANSWER_MSG} "
                    "No sufficiently relevant passage was retrieved from the uploaded document."
                ),
                "source": None,
                "page": None,
                "chunk_id": None,
                "similarity_score": 0.0,
                "sentence_score": 0.0,
                "evidence": None,
                "diagnostics": diagnostics_payload,
            }

        # Formulate answer span
        chosen_sentences = [best_sentences_list[best_s_idx]]

        # Multi-sentence expansion: check immediate neighbor if max_sentences > 1
        q_stems = {self._normalize_stem(self.stemmer.stem(w)) for w in q_words}
        if max_sentences > 1 and len(best_sentences_list) > 1:
            next_idx = best_s_idx + 1
            if next_idx < len(best_sentences_list):
                next_sent = best_sentences_list[next_idx]
                next_score = self.score_sentence(
                    next_sent, question, float(best_chunk.get("similarity_score", 0.0))
                )
                next_words = get_content_words(next_sent, self.language)
                next_stems = {self._normalize_stem(self.stemmer.stem(w)) for w in next_words}
                # Include next sentence if it is topic-relevant to the question and sufficiently high scoring
                if bool(q_stems & next_stems) and next_score >= max(min_sentence_score, 0.60 * best_sentence_score):
                    chosen_sentences.append(next_sent)

            # If next sentence was not included or capacity permits, check previous sentence
            if len(chosen_sentences) < max_sentences and best_s_idx > 0:
                prev_idx = best_s_idx - 1
                prev_sent = best_sentences_list[prev_idx]
                prev_score = self.score_sentence(
                    prev_sent, question, float(best_chunk.get("similarity_score", 0.0))
                )
                prev_words = get_content_words(prev_sent, self.language)
                prev_stems = {self._normalize_stem(self.stemmer.stem(w)) for w in prev_words}
                if bool(q_stems & prev_stems) and prev_score >= max(min_sentence_score, 0.60 * best_sentence_score):
                    chosen_sentences.insert(0, prev_sent)

        # Clean section heading prefix if prepended to the first answer sentence
        chosen_sentences[0] = strip_leading_heading(chosen_sentences[0])
        answer_text = " ".join(s.strip() for s in chosen_sentences).strip()

        diagnostics_payload["abstention_reason"] = None
        diagnostics_payload["selected_chunk_id"] = best_chunk.get("chunk_id")
        diagnostics_payload["selected_sentence_count"] = len(chosen_sentences)

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
            "diagnostics": diagnostics_payload,
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
