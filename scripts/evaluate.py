"""Script to evaluate the Two-Tower recommendation model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import Evaluator
from src.inference.encoder import EmbeddingEncoder
from src.inference.vector_db import VectorDatabase
from src.data.processor import DataProcessor
from src.utils.config import load_config, get_event_weight


def prepare_test_data(
    events_df: pd.DataFrame,
    products_df: pd.DataFrame,
    test_split: float = 0.2,
    min_interactions: int = 3
) -> List[Tuple[str, List[Dict], Set[str]]]:
    """Prepare test data for evaluation.
    
    For each buyer, we use their first (1-test_split) interactions as context
    and the remaining test_split interactions as ground truth relevant items.
    
    Args:
        events_df: Events dataframe
        products_df: Products dataframe
        test_split: Fraction of interactions to use as test
        min_interactions: Minimum interactions required per buyer
        
    Returns:
        List of (buyer_id, interactions, relevant_items) tuples
    """
    # Sort events by timestamp
    if 'timestamp' in events_df.columns:
        events_df = events_df.sort_values('timestamp')
    elif 'created_at' in events_df.columns:
        events_df = events_df.sort_values('created_at')
    
    # Group by buyer
    buyer_events = defaultdict(list)
    for _, row in events_df.iterrows():
        buyer_id = row.get('buyer_id') or row.get('distinct_id')
        product_id = row.get('product_id')
        event_type = row.get('event_type') or row.get('event_name', 'view')
        timestamp = row.get('timestamp') or row.get('created_at')
        
        if buyer_id and product_id:
            buyer_events[buyer_id].append({
                'product_id': product_id,
                'event_type': event_type.lower(),
                'timestamp': str(timestamp) if pd.notna(timestamp) else None
            })
    
    # Create test pairs
    test_pairs = []
    
    for buyer_id, events in buyer_events.items():
        if len(events) < min_interactions:
            continue
        
        # Split into train/test
        split_idx = int(len(events) * (1 - test_split))
        train_events = events[:split_idx]
        test_events = events[split_idx:]
        
        if len(train_events) == 0 or len(test_events) == 0:
            continue
        
        # Get relevant items (products interacted with in test set)
        relevant_items = set(event['product_id'] for event in test_events)
        
        # Use train events as context
        test_pairs.append((buyer_id, train_events, relevant_items))
    
    return test_pairs


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower recommendation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of interactions to use as test'
    )
    parser.add_argument(
        '--min-interactions',
        type=int,
        default=3,
        help='Minimum interactions required per buyer'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 5, 10, 20, 50],
        help='K values for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation_results.json',
        help='Output path for results'
    )
    parser.add_argument(
        '--max-test-samples',
        type=int,
        default=None,
        help='Maximum number of test samples to evaluate'
    )
    parser.add_argument(
        '--skip-exact-metrics',
        action='store_true',
        help='Skip exact match metrics (only compute similarity-based metrics)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    print("Loading data...")
    processor = DataProcessor()
    events_df = processor.load_events()
    products_df = processor.load_products()
    
    # Prepare test data
    print("Preparing test data...")
    test_pairs = prepare_test_data(
        events_df,
        products_df,
        test_split=args.test_split,
        min_interactions=args.min_interactions
    )
    
    if args.max_test_samples:
        test_pairs = test_pairs[:args.max_test_samples]
    
    print(f"Prepared {len(test_pairs)} test samples")
    
    # Initialize encoder
    print("Loading model...")
    model_path = config['inference']['model_checkpoint']
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    encoder = EmbeddingEncoder(
        model_path=model_path,
        config_path=args.config
    )
    
    # Load vector database
    print("Loading vector database...")
    index_dir = Path(config['inference']['index_dir'])
    index_path = index_dir / "product_index.faiss"
    product_ids_path = index_dir / "product_ids.npy"
    mapping_path = index_dir / "product_id_to_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    vector_db = VectorDatabase(embedding_dim=config['model']['embedding_dim'])
    vector_db.load_index(
        str(index_path),
        str(product_ids_path) if product_ids_path.exists() else None,
        str(mapping_path) if mapping_path.exists() else None
    )
    
    # Set product metadata
    print("Loading product metadata...")
    product_metadata = processor.get_product_metadata(products_df)
    encoder.set_product_metadata(product_metadata)
    
    # Initialize evaluator
    evaluator = Evaluator(encoder, vector_db, config_path=args.config)
    evaluator.set_product_metadata(product_metadata)
    
    # Get all product IDs
    all_product_ids = list(product_metadata.keys())
    
    # Run evaluation
    results = evaluator.evaluate_all(
        test_pairs=test_pairs,
        k_values=args.k_values,
        all_product_ids=all_product_ids,
        output_path=args.output
    )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

