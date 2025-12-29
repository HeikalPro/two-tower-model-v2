"""Evaluation module for Two-Tower recommendation model."""

from src.evaluation.metrics import (
    Evaluator,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_ndcg_at_k,
    compute_mrr,
    compute_hit_rate_at_k,
    compute_diversity,
    compute_coverage,
    compute_embedding_stats
)

__all__ = [
    'Evaluator',
    'compute_recall_at_k',
    'compute_precision_at_k',
    'compute_ndcg_at_k',
    'compute_mrr',
    'compute_hit_rate_at_k',
    'compute_diversity',
    'compute_coverage',
    'compute_embedding_stats'
]

