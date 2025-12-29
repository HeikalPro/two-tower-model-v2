"""Comprehensive evaluation metrics for Two-Tower recommendation model."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

from src.inference.encoder import EmbeddingEncoder
from src.inference.vector_db import VectorDatabase
from src.data.processor import DataProcessor
from src.utils.config import load_config, get_event_weight


def compute_recall_at_k(
    retrieved_items: List[str],
    relevant_items: Set[str],
    k: int
) -> float:
    """Compute Recall@K.
    
    Recall@K = |relevant_items ∩ retrieved_items[:k]| / |relevant_items|
    
    Args:
        retrieved_items: List of retrieved product IDs (ordered by relevance)
        relevant_items: Set of relevant product IDs
        k: Number of top items to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = set(retrieved_items[:k])
    intersection = len(top_k & relevant_items)
    return intersection / len(relevant_items)


def compute_precision_at_k(
    retrieved_items: List[str],
    relevant_items: Set[str],
    k: int
) -> float:
    """Compute Precision@K.
    
    Precision@K = |relevant_items ∩ retrieved_items[:k]| / k
    
    Args:
        retrieved_items: List of retrieved product IDs (ordered by relevance)
        relevant_items: Set of relevant product IDs
        k: Number of top items to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved_items[:k])
    intersection = len(top_k & relevant_items)
    return intersection / k


def compute_ndcg_at_k(
    retrieved_items: List[str],
    relevant_items: Set[str],
    k: int
) -> float:
    """Compute Normalized Discounted Cumulative Gain@K.
    
    NDCG@K measures ranking quality by giving higher weight to relevant items
    appearing earlier in the ranking.
    
    Args:
        retrieved_items: List of retrieved product IDs (ordered by relevance)
        relevant_items: Set of relevant product IDs
        k: Number of top items to consider
        
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # Compute DCG@K
    dcg = 0.0
    for i, item in enumerate(retrieved_items[:k], 1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 1)
    
    # Compute IDCG@K (ideal DCG - all relevant items at top)
    idcg = 0.0
    num_relevant = min(len(relevant_items), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(
    retrieved_items: List[str],
    relevant_items: Set[str]
) -> float:
    """Compute Mean Reciprocal Rank.
    
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved_items: List of retrieved product IDs (ordered by relevance)
        relevant_items: Set of relevant product IDs
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, item in enumerate(retrieved_items, 1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def compute_hit_rate_at_k(
    retrieved_items: List[str],
    relevant_items: Set[str],
    k: int
) -> float:
    """Compute Hit Rate@K.
    
    Hit Rate@K = 1 if at least one relevant item is in top-K, else 0.
    
    Args:
        retrieved_items: List of retrieved product IDs (ordered by relevance)
        relevant_items: Set of relevant product IDs
        k: Number of top items to consider
        
    Returns:
        Hit Rate@K score (0.0 or 1.0)
    """
    top_k = set(retrieved_items[:k])
    return 1.0 if len(top_k & relevant_items) > 0 else 0.0


def compute_diversity(
    retrieved_items: List[str],
    product_metadata: Dict[str, Dict],
    attribute: str = 'category'
) -> float:
    """Compute diversity of retrieved items based on an attribute.
    
    Diversity = number of unique attribute values / number of retrieved items
    
    Args:
        retrieved_items: List of retrieved product IDs
        product_metadata: Dictionary mapping product_id to metadata
        attribute: Attribute to use for diversity ('category' or 'brand')
        
    Returns:
        Diversity score (0.0 to 1.0)
    """
    if len(retrieved_items) == 0:
        return 0.0
    
    unique_values = set()
    for product_id in retrieved_items:
        metadata = product_metadata.get(product_id, {})
        value = metadata.get(attribute)
        if value:
            unique_values.add(value)
    
    return len(unique_values) / len(retrieved_items)


def compute_coverage(
    all_retrieved_items: Set[str],
    all_product_ids: Set[str]
) -> float:
    """Compute catalog coverage.
    
    Coverage = |unique retrieved items| / |all products|
    
    Args:
        all_retrieved_items: Set of all unique product IDs that were retrieved
        all_product_ids: Set of all product IDs in catalog
        
    Returns:
        Coverage score (0.0 to 1.0)
    """
    if len(all_product_ids) == 0:
        return 0.0
    
    return len(all_retrieved_items) / len(all_product_ids)


def compute_embedding_stats(
    embeddings: np.ndarray
) -> Dict[str, float]:
    """Compute statistics about embeddings.
    
    Args:
        embeddings: Embedding array [n_samples, embedding_dim]
        
    Returns:
        Dictionary with embedding statistics
    """
    # Compute norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Compute pairwise cosine similarities (sample for large datasets)
    n_samples = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    # Normalize for cosine similarity
    normalized = sample_embeddings / (np.linalg.norm(sample_embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(normalized, normalized.T)
    # Remove diagonal (self-similarity = 1.0)
    mask = ~np.eye(n_samples, dtype=bool)
    similarities = similarities[mask]
    
    return {
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms)),
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities))
    }


class Evaluator:
    """Comprehensive evaluator for Two-Tower recommendation model."""
    
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        vector_db: VectorDatabase,
        config_path: str = "configs/config.yaml"
    ):
        """Initialize evaluator.
        
        Args:
            encoder: Embedding encoder
            vector_db: Vector database for retrieval
            config_path: Path to configuration file
        """
        self.encoder = encoder
        self.vector_db = vector_db
        self.config = load_config(config_path)
        self.product_metadata = None
    
    def set_product_metadata(self, product_metadata: Dict):
        """Set product metadata.
        
        Args:
            product_metadata: Dictionary mapping product_id to metadata
        """
        self.product_metadata = product_metadata
    
    def evaluate_retrieval(
        self,
        test_pairs: List[Tuple[str, List[Dict], Set[str]]],
        k_values: List[int] = [1, 5, 10, 20, 50],
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            test_pairs: List of (buyer_id, interactions, relevant_items) tuples
                - buyer_id: Buyer identifier
                - interactions: List of interaction dicts (product_id, event_type, timestamp)
                - relevant_items: Set of relevant product IDs (ground truth)
            k_values: List of K values to evaluate
            verbose: Whether to show progress
            
        Returns:
            Dictionary with metrics for each K value
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before evaluation")
        
        # Initialize metrics
        metrics = {}
        for k in k_values:
            metrics[f'recall@{k}'] = []
            metrics[f'precision@{k}'] = []
            metrics[f'ndcg@{k}'] = []
            metrics[f'hit_rate@{k}'] = []
        metrics['mrr'] = []
        
        # Evaluate each test pair
        iterator = tqdm(test_pairs, desc="Evaluating retrieval") if verbose else test_pairs
        
        for buyer_id, interactions, relevant_items in iterator:
            try:
                # Encode buyer
                buyer_embedding = self.encoder.encode_buyer(interactions)
                
                # Retrieve top-K items (use max K)
                max_k = max(k_values)
                results = self.vector_db.retrieve(buyer_embedding, k=max_k)
                retrieved_items = [product_id for product_id, _ in results]
                
                # Compute metrics for each K
                for k in k_values:
                    metrics[f'recall@{k}'].append(
                        compute_recall_at_k(retrieved_items, relevant_items, k)
                    )
                    metrics[f'precision@{k}'].append(
                        compute_precision_at_k(retrieved_items, relevant_items, k)
                    )
                    metrics[f'ndcg@{k}'].append(
                        compute_ndcg_at_k(retrieved_items, relevant_items, k)
                    )
                    metrics[f'hit_rate@{k}'].append(
                        compute_hit_rate_at_k(retrieved_items, relevant_items, k)
                    )
                
                metrics['mrr'].append(
                    compute_mrr(retrieved_items, relevant_items)
                )
                
            except Exception as e:
                if verbose:
                    print(f"Error evaluating buyer {buyer_id}: {e}")
                continue
        
        # Aggregate metrics
        aggregated = {}
        for key, values in metrics.items():
            if len(values) > 0:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_median'] = float(np.median(values))
        
        return aggregated
    
    def evaluate_embedding_quality(
        self,
        product_ids: Optional[List[str]] = None,
        sample_size: int = 10000
    ) -> Dict[str, float]:
        """Evaluate embedding quality.
        
        Args:
            product_ids: List of product IDs to evaluate (None = all)
            sample_size: Number of products to sample for similarity computation
            
        Returns:
            Dictionary with embedding statistics
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before evaluation")
        
        # Get product IDs
        if product_ids is None:
            product_ids = list(self.product_metadata.keys())
        
        # Sample if needed
        if len(product_ids) > sample_size:
            product_ids = np.random.choice(product_ids, sample_size, replace=False).tolist()
        
        # Encode products
        print(f"Encoding {len(product_ids)} products...")
        embeddings = self.encoder.encode_items(product_ids, batch_size=32)
        
        # Compute statistics
        stats = compute_embedding_stats(embeddings)
        
        return stats
    
    def evaluate_diversity(
        self,
        test_pairs: List[Tuple[str, List[Dict], Set[str]]],
        k: int = 10,
        attribute: str = 'category'
    ) -> Dict[str, float]:
        """Evaluate diversity of recommendations.
        
        Args:
            test_pairs: List of (buyer_id, interactions, relevant_items) tuples
            k: Number of items to retrieve
            attribute: Attribute to use for diversity ('category' or 'brand')
            
        Returns:
            Dictionary with diversity metrics
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before evaluation")
        
        diversities = []
        
        for buyer_id, interactions, _ in tqdm(test_pairs, desc="Evaluating diversity"):
            try:
                # Encode buyer
                buyer_embedding = self.encoder.encode_buyer(interactions)
                
                # Retrieve items
                results = self.vector_db.retrieve(buyer_embedding, k=k)
                retrieved_items = [product_id for product_id, _ in results]
                
                # Compute diversity
                div = compute_diversity(retrieved_items, self.product_metadata, attribute)
                diversities.append(div)
                
            except Exception as e:
                print(f"Error evaluating diversity for buyer {buyer_id}: {e}")
                continue
        
        if len(diversities) == 0:
            return {}
        
        return {
            f'diversity_{attribute}_mean': float(np.mean(diversities)),
            f'diversity_{attribute}_std': float(np.std(diversities)),
            f'diversity_{attribute}_median': float(np.median(diversities))
        }
    
    def evaluate_coverage(
        self,
        test_pairs: List[Tuple[str, List[Dict], Set[str]]],
        k: int = 10,
        all_product_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate catalog coverage.
        
        Args:
            test_pairs: List of (buyer_id, interactions, relevant_items) tuples
            k: Number of items to retrieve per buyer
            all_product_ids: List of all product IDs in catalog (None = use metadata keys)
            
        Returns:
            Dictionary with coverage metrics
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before evaluation")
        
        if all_product_ids is None:
            all_product_ids = list(self.product_metadata.keys())
        
        all_retrieved = set()
        
        for buyer_id, interactions, _ in tqdm(test_pairs, desc="Evaluating coverage"):
            try:
                # Encode buyer
                buyer_embedding = self.encoder.encode_buyer(interactions)
                
                # Retrieve items
                results = self.vector_db.retrieve(buyer_embedding, k=k)
                retrieved_items = [product_id for product_id, _ in results]
                all_retrieved.update(retrieved_items)
                
            except Exception as e:
                print(f"Error evaluating coverage for buyer {buyer_id}: {e}")
                continue
        
        coverage = compute_coverage(all_retrieved, set(all_product_ids))
        
        return {
            'coverage': coverage,
            'unique_retrieved': len(all_retrieved),
            'total_products': len(all_product_ids)
        }
    
    def evaluate_all(
        self,
        test_pairs: List[Tuple[str, List[Dict], Set[str]]],
        k_values: List[int] = [1, 5, 10, 20, 50],
        all_product_ids: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, any]:
        """Run comprehensive evaluation.
        
        Args:
            test_pairs: List of (buyer_id, interactions, relevant_items) tuples
            k_values: List of K values to evaluate
            all_product_ids: List of all product IDs in catalog
            output_path: Optional path to save results JSON
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("=" * 60)
        print("Starting Comprehensive Evaluation")
        print("=" * 60)
        
        results = {}
        
        # 1. Retrieval metrics
        print("\n1. Evaluating Retrieval Performance...")
        retrieval_metrics = self.evaluate_retrieval(test_pairs, k_values)
        results['retrieval'] = retrieval_metrics
        
        # 2. Embedding quality
        print("\n2. Evaluating Embedding Quality...")
        embedding_stats = self.evaluate_embedding_quality()
        results['embedding_quality'] = embedding_stats
        
        # 3. Diversity
        print("\n3. Evaluating Diversity...")
        diversity_category = self.evaluate_diversity(test_pairs, k=max(k_values), attribute='category')
        diversity_brand = self.evaluate_diversity(test_pairs, k=max(k_values), attribute='brand')
        results['diversity'] = {**diversity_category, **diversity_brand}
        
        # 4. Coverage
        print("\n4. Evaluating Coverage...")
        coverage_metrics = self.evaluate_coverage(test_pairs, k=max(k_values), all_product_ids=all_product_ids)
        results['coverage'] = coverage_metrics
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        self._print_summary(results)
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary.
        
        Args:
            results: Evaluation results dictionary
        """
        # Retrieval metrics
        if 'retrieval' in results:
            print("\nRetrieval Metrics:")
            for key, value in sorted(results['retrieval'].items()):
                if key.endswith('_mean'):
                    metric_name = key.replace('_mean', '')
                    print(f"  {metric_name:30s}: {value:.4f}")
        
        # Embedding quality
        if 'embedding_quality' in results:
            print("\nEmbedding Quality:")
            for key, value in results['embedding_quality'].items():
                print(f"  {key:30s}: {value:.4f}")
        
        # Diversity
        if 'diversity' in results:
            print("\nDiversity:")
            for key, value in sorted(results['diversity'].items()):
                if key.endswith('_mean'):
                    metric_name = key.replace('_mean', '')
                    print(f"  {metric_name:30s}: {value:.4f}")
        
        # Coverage
        if 'coverage' in results:
            print("\nCoverage:")
            for key, value in results['coverage'].items():
                print(f"  {key:30s}: {value}")

