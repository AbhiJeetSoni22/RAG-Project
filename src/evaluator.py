"""Retrieval Evaluation Module for Phase 6.

This module provides objective, standardized benchmarking and evaluation for
passage retrieval engines (TF-IDF vs. Semantic Embeddings). It computes standard
Information Retrieval (IR) metrics:
- Hit@K
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

The evaluator operates independently of the UI and works with any retriever
implementing the `.search(query, top_k, min_similarity)` interface.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES: list[int] = [1, 3, 5]


class EvaluationError(Exception):
    """Raised when an error occurs during dataset evaluation or validation."""
    pass


# =============================================================================
# Core Metric Calculation Functions
# =============================================================================

def compute_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> int:
    """Compute Hit@K for a single query.

    A retrieval is considered a hit (1) if at least one ground-truth relevant
    chunk appears in the top K retrieved chunks; otherwise 0.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (rank 1 first).
        relevant_chunk_ids: List of ground-truth relevant chunk IDs.
        k: Cutoff rank for evaluation (must be >= 1).

    Returns:
        1 if any relevant chunk is in the top K retrieved chunks, else 0.
    """
    if k <= 0 or not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)

    return 1 if bool(top_k_retrieved & relevant_set) else 0


def compute_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Compute Precision@K for a single query.

    Precision@K = (number of relevant retrieved chunks in top K) / K.

    Standard IR convention divides by K (the evaluation cutoff budget).
    If k <= 0 or relevant_chunk_ids is empty, returns 0.0.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevant_chunk_ids: List of ground-truth relevant chunk IDs.
        k: Evaluation cutoff budget.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    hits = len(top_k_retrieved & relevant_set)

    return hits / float(k)


def compute_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Recall@K = (number of relevant retrieved chunks in top K) / (total relevant chunks).

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevant_chunk_ids: List of ground-truth relevant chunk IDs.
        k: Evaluation cutoff rank.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    hits = len(top_k_retrieved & relevant_set)

    return hits / float(len(relevant_set))


def compute_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: Optional[int] = None,
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    If the first relevant chunk appears at 1-based rank r (within top k if specified):
        RR = 1.0 / r
    If no relevant chunk appears:
        RR = 0.0

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevant_chunk_ids: List of ground-truth relevant chunk IDs.
        k: Optional maximum rank cutoff to consider. If None, considers all retrieved.

    Returns:
        Reciprocal rank value in [0.0, 1.0].
    """
    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    limit = len(retrieved_chunk_ids) if k is None else min(k, len(retrieved_chunk_ids))

    for rank_0, chunk_id in enumerate(retrieved_chunk_ids[:limit]):
        if chunk_id in relevant_set:
            return 1.0 / float(rank_0 + 1)

    return 0.0


# =============================================================================
# Structured Result Containers
# =============================================================================

@dataclass
class QueryEvaluationResult:
    """Evaluation result for a single question at a specific Top-K."""
    question_id: str
    question: str
    category: str
    method: str
    top_k: int
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    hit: int
    precision: float
    recall: float
    reciprocal_rank: float
    retrieved_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert query evaluation result to a serializable dictionary."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "method": self.method,
            "top_k": self.top_k,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "hit": self.hit,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "retrieved_scores": self.retrieved_scores,
        }


@dataclass
class CategoryMetric:
    """Aggregate metrics for a specific question category."""
    category: str
    query_count: int
    hit_rates: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        """Convert category metrics to dictionary."""
        return {
            "category": self.category,
            "query_count": self.query_count,
            "hit_rates": {f"Hit@{k}": round(v, 4) for k, v in self.hit_rates.items()},
            "precision_at_k": {f"Precision@{k}": round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {f"Recall@{k}": round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr": round(self.mrr, 4),
        }


@dataclass
class EvaluationSummary:
    """Comprehensive summary of evaluation results for a retrieval method."""
    method: str
    total_queries: int
    k_values: list[int]
    hit_rates: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    category_metrics: dict[str, CategoryMetric]
    query_results: list[QueryEvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "method": self.method,
            "total_queries": self.total_queries,
            "k_values": self.k_values,
            "hit_rates": {f"Hit@{k}": round(v, 4) for k, v in self.hit_rates.items()},
            "precision_at_k": {f"Precision@{k}": round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {f"Recall@{k}": round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr": round(self.mrr, 4),
            "category_metrics": {cat: m.to_dict() for cat, m in self.category_metrics.items()},
            "query_results": [q.to_dict() for q in self.query_results],
        }


@dataclass
class ComparisonResult:
    """Side-by-side comparison between multiple retrieval methods."""
    methods: list[str]
    k_values: list[int]
    summaries: dict[str, EvaluationSummary]
    overall_metrics_table: list[dict[str, Any]]
    category_metrics_table: list[dict[str, Any]]
    per_question_table: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to dictionary."""
        return {
            "methods": self.methods,
            "k_values": self.k_values,
            "summaries": {m: s.to_dict() for m, s in self.summaries.items()},
            "overall_metrics_table": self.overall_metrics_table,
            "category_metrics_table": self.category_metrics_table,
            "per_question_table": self.per_question_table,
        }


# =============================================================================
# Dataset Loader & Validator
# =============================================================================

def load_evaluation_dataset(
    dataset_source: Union[str, Path, Sequence[dict[str, Any]], io.IOBase],
) -> list[dict[str, Any]]:
    """Load and validate an evaluation dataset from JSON file, path, or object list.

    Args:
        dataset_source: File path, Path object, file-like object, or list of dictionaries.

    Returns:
        Validated list of evaluation question dictionaries.

    Raises:
        EvaluationError: If the format is invalid or required fields are missing.
    """
    if isinstance(dataset_source, list):
        raw_items = dataset_source
    elif isinstance(dataset_source, (str, Path)):
        path = Path(dataset_source)
        if not path.is_file():
            raise EvaluationError(f"Evaluation dataset file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
        except Exception as exc:
            raise EvaluationError(f"Failed to parse JSON dataset from {path}: {exc}") from exc
    elif hasattr(dataset_source, "read"):
        try:
            raw_items = json.load(dataset_source)
        except Exception as exc:
            raise EvaluationError(f"Failed to read evaluation dataset stream: {exc}") from exc
    else:
        raise EvaluationError(f"Unsupported dataset source type: {type(dataset_source)}")

    if not isinstance(raw_items, list):
        raise EvaluationError(f"Dataset root must be a JSON array, got {type(raw_items).__name__}")

    validated_items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise EvaluationError(f"Dataset item at index {idx} must be an object.")

        q_id = str(item.get("id", f"q{idx + 1:03d}")).strip()
        question = str(item.get("question", "")).strip()
        relevant_chunk_ids = item.get("relevant_chunk_ids")

        if not question:
            raise EvaluationError(f"Item {q_id} (index {idx}) is missing required 'question' field.")

        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise EvaluationError(
                f"Item {q_id} (index {idx}) must contain a non-empty 'relevant_chunk_ids' list."
            )

        category = str(item.get("category", "general")).strip() or "general"
        relevant_pages = item.get("relevant_pages", [])
        source = str(item.get("source", "")).strip()

        validated_items.append({
            "id": q_id,
            "question": question,
            "category": category,
            "relevant_chunk_ids": [str(cid).strip() for cid in relevant_chunk_ids],
            "relevant_pages": relevant_pages,
            "source": source,
        })

    return validated_items


def validate_dataset_against_chunks(
    questions: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Check alignment between ground-truth chunk IDs and available document chunks.

    Args:
        questions: List of validated evaluation question dicts.
        chunks: List of processed document chunk dicts from Phase 2.

    Returns:
        Dictionary with validation statistics and missing chunk details.
    """
    chunk_ids = {c["chunk_id"] for c in chunks}
    all_ground_truth_ids: set[str] = set()
    for q in questions:
        all_ground_truth_ids.update(q.get("relevant_chunk_ids", []))

    missing_chunk_ids = sorted(list(all_ground_truth_ids - chunk_ids))
    matching_chunk_ids = sorted(list(all_ground_truth_ids & chunk_ids))

    aligned_questions = [
        q for q in questions
        if any(cid in chunk_ids for cid in q.get("relevant_chunk_ids", []))
    ]

    return {
        "total_questions": len(questions),
        "aligned_questions_count": len(aligned_questions),
        "total_chunks_in_doc": len(chunk_ids),
        "total_ground_truth_chunks": len(all_ground_truth_ids),
        "matching_chunks_count": len(matching_chunk_ids),
        "missing_chunks_count": len(missing_chunk_ids),
        "missing_chunk_ids": missing_chunk_ids,
        "is_fully_aligned": len(missing_chunk_ids) == 0,
    }


# =============================================================================
# Retriever Evaluation Logic
# =============================================================================

def evaluate_retriever(
    retriever: Any,
    questions: Sequence[dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5),
    min_similarity: float = 0.0,
    method_name: Optional[str] = None,
) -> EvaluationSummary:
    """Evaluate a single retrieval engine against an evaluation dataset.

    Args:
        retriever: Retriever instance implementing `.search(query, top_k, min_similarity)`.
        questions: Sequence of evaluation question dictionaries.
        k_values: Cutoff values K for evaluation (e.g. [1, 3, 5]).
        min_similarity: Minimum cosine similarity threshold (default 0.0 for evaluation).
        method_name: Label for the retriever (defaults to retriever class name).

    Returns:
        EvaluationSummary object containing overall, category, and per-query metrics.
    """
    if method_name is None:
        method_name = retriever.__class__.__name__

    sorted_k_values = sorted(list(set(int(k) for k in k_values if int(k) > 0)))
    if not sorted_k_values:
        sorted_k_values = [1, 3, 5]

    max_k = max(sorted_k_values)
    total_queries = len(questions)

    if total_queries == 0:
        return EvaluationSummary(
            method=method_name,
            total_queries=0,
            k_values=sorted_k_values,
            hit_rates={k: 0.0 for k in sorted_k_values},
            precision_at_k={k: 0.0 for k in sorted_k_values},
            recall_at_k={k: 0.0 for k in sorted_k_values},
            mrr=0.0,
            category_metrics={},
            query_results=[],
        )

    # We evaluate for each query at max_k once to get ranked retrieved results
    all_query_results: list[QueryEvaluationResult] = []

    # Accumulators for aggregate metrics
    hit_accumulators: dict[int, list[int]] = {k: [] for k in sorted_k_values}
    precision_accumulators: dict[int, list[float]] = {k: [] for k in sorted_k_values}
    recall_accumulators: dict[int, list[float]] = {k: [] for k in sorted_k_values}
    rr_accumulator: list[float] = []

    # Category grouping: category -> list of question records
    category_map: dict[str, list[dict[str, Any]]] = {}

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        q_category = q.get("category", "general")
        relevant_ids = q["relevant_chunk_ids"]

        # Run retrieval up to max_k
        try:
            retrieved = retriever.search(
                query=q_text,
                top_k=max_k,
                min_similarity=min_similarity,
            )
        except Exception as exc:
            logger.warning(f"Retriever {method_name} failed on query '{q_id}': {exc}")
            retrieved = []

        retrieved_ids = [r.get("chunk_id", "") for r in retrieved]
        retrieved_scores = [float(r.get("similarity_score", 0.0)) for r in retrieved]

        # Compute reciprocal rank over all retrieved candidates up to max_k
        query_rr = compute_reciprocal_rank(retrieved_ids, relevant_ids, k=max_k)
        rr_accumulator.append(query_rr)

        # Compute metrics across all target K values
        for k in sorted_k_values:
            k_hit = compute_hit_at_k(retrieved_ids, relevant_ids, k=k)
            k_prec = compute_precision_at_k(retrieved_ids, relevant_ids, k=k)
            k_rec = compute_recall_at_k(retrieved_ids, relevant_ids, k=k)
            k_rr = compute_reciprocal_rank(retrieved_ids, relevant_ids, k=k)

            hit_accumulators[k].append(k_hit)
            precision_accumulators[k].append(k_prec)
            recall_accumulators[k].append(k_rec)

            # Store query evaluation result for this K
            all_query_results.append(
                QueryEvaluationResult(
                    question_id=q_id,
                    question=q_text,
                    category=q_category,
                    method=method_name,
                    top_k=k,
                    retrieved_chunk_ids=retrieved_ids[:k],
                    relevant_chunk_ids=relevant_ids,
                    hit=k_hit,
                    precision=k_prec,
                    recall=k_rec,
                    reciprocal_rank=k_rr,
                    retrieved_scores=retrieved_scores[:k],
                )
            )

        # Track per-category results at max_k
        if q_category not in category_map:
            category_map[q_category] = []
        category_map[q_category].append({
            "q_id": q_id,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "rr": query_rr,
        })

    # Overall aggregate metrics
    overall_hit_rates = {
        k: sum(hit_accumulators[k]) / float(total_queries)
        for k in sorted_k_values
    }
    overall_precision = {
        k: sum(precision_accumulators[k]) / float(total_queries)
        for k in sorted_k_values
    }
    overall_recall = {
        k: sum(recall_accumulators[k]) / float(total_queries)
        for k in sorted_k_values
    }
    overall_mrr = sum(rr_accumulator) / float(total_queries)

    # Per-category aggregate metrics
    category_metrics: dict[str, CategoryMetric] = {}
    for cat, records in category_map.items():
        cat_count = len(records)
        cat_hits: dict[int, float] = {}
        cat_precs: dict[int, float] = {}
        cat_recs: dict[int, float] = {}

        for k in sorted_k_values:
            cat_hits[k] = sum(
                compute_hit_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in records
            ) / float(cat_count)
            cat_precs[k] = sum(
                compute_precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in records
            ) / float(cat_count)
            cat_recs[k] = sum(
                compute_recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in records
            ) / float(cat_count)

        cat_mrr = sum(r["rr"] for r in records) / float(cat_count)

        category_metrics[cat] = CategoryMetric(
            category=cat,
            query_count=cat_count,
            hit_rates=cat_hits,
            precision_at_k=cat_precs,
            recall_at_k=cat_recs,
            mrr=cat_mrr,
        )

    return EvaluationSummary(
        method=method_name,
        total_queries=total_queries,
        k_values=sorted_k_values,
        hit_rates=overall_hit_rates,
        precision_at_k=overall_precision,
        recall_at_k=overall_recall,
        mrr=overall_mrr,
        category_metrics=category_metrics,
        query_results=all_query_results,
    )


# =============================================================================
# Retrieval Comparison Function
# =============================================================================

def compare_retrieval_methods(
    questions: Sequence[dict[str, Any]],
    retrievers: dict[str, Any],
    k_values: Sequence[int] = (1, 3, 5),
    min_similarity: float = 0.0,
) -> ComparisonResult:
    """Benchmark and compare multiple retrieval methods side-by-side.

    Args:
        questions: Validated evaluation question sequence.
        retrievers: Dictionary of {method_name: retriever_instance}.
        k_values: Cutoff values K to benchmark (e.g. [1, 3, 5]).
        min_similarity: Minimum cosine similarity threshold.

    Returns:
        ComparisonResult containing summaries, metric tables, and drill-down results.
    """
    sorted_k_values = sorted(list(set(int(k) for k in k_values if int(k) > 0)))
    if not sorted_k_values:
        sorted_k_values = [1, 3, 5]

    method_names = list(retrievers.keys())
    summaries: dict[str, EvaluationSummary] = {}

    for name, ret in retrievers.items():
        summaries[name] = evaluate_retriever(
            retriever=ret,
            questions=questions,
            k_values=sorted_k_values,
            min_similarity=min_similarity,
            method_name=name,
        )

    # 1. Overall Metrics Comparison Table
    metrics_table: list[dict[str, Any]] = []

    # Hit@K rows
    for k in sorted_k_values:
        row: dict[str, Any] = {"Metric": f"Hit@{k}"}
        for name in method_names:
            row[name] = round(summaries[name].hit_rates.get(k, 0.0), 4)
        metrics_table.append(row)

    # Precision@max_k and Recall@max_k
    max_k = max(sorted_k_values)
    prec_row: dict[str, Any] = {"Metric": f"Precision@{max_k}"}
    rec_row: dict[str, Any] = {"Metric": f"Recall@{max_k}"}
    for name in method_names:
        prec_row[name] = round(summaries[name].precision_at_k.get(max_k, 0.0), 4)
        rec_row[name] = round(summaries[name].recall_at_k.get(max_k, 0.0), 4)
    metrics_table.append(prec_row)
    metrics_table.append(rec_row)

    # MRR row
    mrr_row: dict[str, Any] = {"Metric": "MRR"}
    for name in method_names:
        mrr_row[name] = round(summaries[name].mrr, 4)
    metrics_table.append(mrr_row)

    # 2. Category Breakdown Table (reporting Hit@max_k and MRR per category)
    categories = sorted(list({q.get("category", "general") for q in questions}))
    category_table: list[dict[str, Any]] = []

    for cat in categories:
        cat_count = sum(1 for q in questions if q.get("category", "general") == cat)
        cat_row: dict[str, Any] = {
            "Category": cat,
            "Count": cat_count,
        }
        for name in method_names:
            cat_metric = summaries[name].category_metrics.get(cat)
            if cat_metric:
                cat_row[f"{name} Hit@{max_k}"] = round(cat_metric.hit_rates.get(max_k, 0.0), 4)
                cat_row[f"{name} MRR"] = round(cat_metric.mrr, 4)
            else:
                cat_row[f"{name} Hit@{max_k}"] = 0.0
                cat_row[f"{name} MRR"] = 0.0
        category_table.append(cat_row)

    # 3. Per-Question Side-by-Side Breakdown (at max_k)
    per_question_table: list[dict[str, Any]] = []
    for q in questions:
        q_id = q["id"]
        q_row: dict[str, Any] = {
            "id": q_id,
            "question": q["question"],
            "category": q.get("category", "general"),
            "relevant_chunk_ids": ", ".join(q["relevant_chunk_ids"]),
        }
        for name in method_names:
            # Find the query result at max_k for this method
            q_res = next(
                (r for r in summaries[name].query_results if r.question_id == q_id and r.top_k == max_k),
                None,
            )
            if q_res:
                q_row[f"{name} Retrieved"] = ", ".join(q_res.retrieved_chunk_ids)
                q_row[f"{name} Hit@{max_k}"] = q_res.hit
                q_row[f"{name} RR"] = round(q_res.reciprocal_rank, 4)
            else:
                q_row[f"{name} Retrieved"] = ""
                q_row[f"{name} Hit@{max_k}"] = 0
                q_row[f"{name} RR"] = 0.0
        per_question_table.append(q_row)

    return ComparisonResult(
        methods=method_names,
        k_values=sorted_k_values,
        summaries=summaries,
        overall_metrics_table=metrics_table,
        category_metrics_table=category_table,
        per_question_table=per_question_table,
    )


# =============================================================================
# CSV Export Utilities
# =============================================================================

def export_results_to_csv(
    query_results: Sequence[Union[QueryEvaluationResult, dict[str, Any]]],
    filepath: Optional[Union[str, Path]] = None,
) -> str:
    """Export query-level evaluation results to CSV string and optionally write to file.

    Columns:
        question_id, question, category, method, k, hit, precision, recall, reciprocal_rank,
        retrieved_chunk_ids, relevant_chunk_ids

    Args:
        query_results: List of QueryEvaluationResult instances or dicts.
        filepath: Optional path to save the CSV file (e.g. 'evaluation/results.csv').

    Returns:
        Generated CSV content string.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    writer.writerow([
        "question_id",
        "question",
        "category",
        "method",
        "k",
        "hit",
        "precision",
        "recall",
        "reciprocal_rank",
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
    ])

    for item in query_results:
        if isinstance(item, QueryEvaluationResult):
            row = [
                item.question_id,
                item.question,
                item.category,
                item.method,
                item.top_k,
                item.hit,
                round(item.precision, 4),
                round(item.recall, 4),
                round(item.reciprocal_rank, 4),
                "; ".join(item.retrieved_chunk_ids),
                "; ".join(item.relevant_chunk_ids),
            ]
        elif isinstance(item, dict):
            row = [
                item.get("question_id", ""),
                item.get("question", ""),
                item.get("category", ""),
                item.get("method", ""),
                item.get("top_k", item.get("k", "")),
                item.get("hit", 0),
                round(float(item.get("precision", 0.0)), 4),
                round(float(item.get("recall", 0.0)), 4),
                round(float(item.get("reciprocal_rank", 0.0)), 4),
                "; ".join(item.get("retrieved_chunk_ids", [])),
                "; ".join(item.get("relevant_chunk_ids", [])),
            ]
        else:
            continue

        writer.writerow(row)

    csv_text = output.getvalue()

    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)

    return csv_text


def export_summary_to_csv(
    summaries: Union[ComparisonResult, dict[str, EvaluationSummary]],
    filepath: Optional[Union[str, Path]] = None,
) -> str:
    """Export summary metrics across methods to CSV string and optionally write to file.

    Columns:
        method, total_queries, hit@1, hit@3, hit@5, precision@5, recall@5, mrr

    Args:
        summaries: ComparisonResult or dict of {method_name: EvaluationSummary}.
        filepath: Optional path to save the CSV file (e.g. 'evaluation/summary.csv').

    Returns:
        Generated CSV content string.
    """
    if isinstance(summaries, ComparisonResult):
        summary_map = summaries.summaries
    else:
        summary_map = summaries

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    # Find union of all K values
    all_k = sorted(list({k for s in summary_map.values() for k in s.k_values}))
    if not all_k:
        all_k = [1, 3, 5]
    max_k = max(all_k)

    header = ["method", "total_queries"]
    for k in all_k:
        header.append(f"hit@{k}")
    header.extend([f"precision@{max_k}", f"recall@{max_k}", "mrr"])
    writer.writerow(header)

    for method_name, s in summary_map.items():
        row = [method_name, s.total_queries]
        for k in all_k:
            row.append(round(s.hit_rates.get(k, 0.0), 4))
        row.append(round(s.precision_at_k.get(max_k, 0.0), 4))
        row.append(round(s.recall_at_k.get(max_k, 0.0), 4))
        row.append(round(s.mrr, 4))
        writer.writerow(row)

    csv_text = output.getvalue()

    if filepath is not None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)

    return csv_text
