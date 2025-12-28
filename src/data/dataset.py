"""PyTorch dataset for training."""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import random
import numpy as np


class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training."""
    
    def __init__(
        self,
        positive_pairs: List[Tuple[str, str, int]],
        product_metadata: Dict[str, Dict],
        buyer_sequences: Dict[str, List[Tuple[str, int]]],
        all_product_ids: List[str],
        num_negatives: int = 4
    ):
        """Initialize dataset.
        
        Args:
            positive_pairs: List of (buyer_id, product_id, weight) tuples
            product_metadata: Dictionary mapping product_id to metadata
            buyer_sequences: Dictionary mapping buyer_id to interaction sequences
            all_product_ids: List of all product IDs for negative sampling
            num_negatives: Number of negative samples per positive
        """
        self.positive_pairs = positive_pairs
        self.product_metadata = product_metadata
        self.buyer_sequences = buyer_sequences
        self.all_product_ids = all_product_ids
        self.num_negatives = num_negatives
        
        # Filter out pairs with missing data
        self.valid_pairs = [
            (buyer_id, product_id, weight)
            for buyer_id, product_id, weight in positive_pairs
            if buyer_id in buyer_sequences and product_id in product_metadata
        ]
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with buyer_id, positive_product_id, negative_product_ids,
            buyer_sequence, positive_product_text, negative_product_texts
        """
        buyer_id, positive_product_id, weight = self.valid_pairs[idx]
        
        # Get buyer interaction sequence
        buyer_sequence = self.buyer_sequences[buyer_id]
        
        # Sample negative products
        negative_product_ids = self._sample_negatives(positive_product_id)
        
        # Get product texts
        positive_text = self.product_metadata[positive_product_id]['text']
        negative_texts = [
            self.product_metadata[neg_id]['text']
            for neg_id in negative_product_ids
        ]
        
        return {
            'buyer_id': buyer_id,
            'positive_product_id': positive_product_id,
            'negative_product_ids': negative_product_ids,
            'buyer_sequence': buyer_sequence,
            'positive_product_text': positive_text,
            'negative_product_texts': negative_texts,
            'weight': weight
        }
    
    def _sample_negatives(self, positive_product_id: str) -> List[str]:
        """Sample negative product IDs.
        
        Args:
            positive_product_id: Positive product ID to avoid
            
        Returns:
            List of negative product IDs
        """
        candidates = [pid for pid in self.all_product_ids if pid != positive_product_id]
        return random.sample(candidates, min(self.num_negatives, len(candidates)))


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    return {
        'buyer_ids': [item['buyer_id'] for item in batch],
        'positive_product_ids': [item['positive_product_id'] for item in batch],
        'negative_product_ids': [item['negative_product_ids'] for item in batch],
        'buyer_sequences': [item['buyer_sequence'] for item in batch],
        'positive_product_texts': [item['positive_product_text'] for item in batch],
        'negative_product_texts': [item['negative_product_texts'] for item in batch],
        'weights': torch.tensor([item['weight'] for item in batch], dtype=torch.float32)
    }


