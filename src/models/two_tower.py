"""Combined Two-Tower model for recommendation retrieval."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from .item_tower import ItemTower
from .buyer_tower import BuyerTower


class TwoTowerModel(nn.Module):
    """Combined Two-Tower model for recommendation retrieval."""
    
    def __init__(
        self,
        item_tower: ItemTower,
        buyer_tower: BuyerTower
    ):
        """Initialize Two-Tower model.
        
        Args:
            item_tower: Item Tower instance
            buyer_tower: Buyer Tower instance
        """
        super(TwoTowerModel, self).__init__()
        
        self.item_tower = item_tower
        self.buyer_tower = buyer_tower
        
        # Ensure embedding dimensions match
        assert item_tower.embedding_dim == buyer_tower.embedding_dim, \
            "Item and Buyer towers must have the same embedding dimension"
    
    def encode_items(
        self,
        texts: List[str],
        brands: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Encode products using Item Tower.
        
        Args:
            texts: List of product texts
            brands: Optional list of brands
            categories: Optional list of categories
            
        Returns:
            Item embeddings [batch_size, embedding_dim]
        """
        return self.item_tower(texts, brands, categories)
    
    def encode_buyer(
        self,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Encode buyer using Buyer Tower.
        
        Args:
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            weights: Event weights [batch_size, seq_len]
            
        Returns:
            Buyer embeddings [batch_size, embedding_dim]
        """
        return self.buyer_tower(item_embeddings, weights)
    
    def forward(
        self,
        buyer_sequences: List[List[Tuple[str, int]]],
        positive_texts: List[str],
        negative_texts: List[List[str]],
        positive_brands: Optional[List[str]] = None,
        positive_categories: Optional[List[str]] = None,
        negative_brands: Optional[List[List[str]]] = None,
        negative_categories: Optional[List[List[str]]] = None,
        product_embeddings_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            buyer_sequences: List of buyer interaction sequences [(product_id, weight), ...]
            positive_texts: List of positive product texts
            negative_texts: List of lists of negative product texts
            positive_brands: Optional list of positive product brands
            positive_categories: Optional list of positive product categories
            negative_brands: Optional list of lists of negative product brands
            negative_categories: Optional list of lists of negative product categories
            product_embeddings_cache: Optional cache of product embeddings
            
        Returns:
            Dictionary with buyer_embeddings, positive_embeddings, negative_embeddings
        """
        batch_size = len(buyer_sequences)
        device = next(self.parameters()).device
        
        # Encode positive products
        positive_embeddings = self.item_tower(
            positive_texts,
            positive_brands,
            positive_categories
        )  # [batch_size, embedding_dim]
        
        # Encode negative products
        all_negative_texts = [text for neg_list in negative_texts for text in neg_list]
        all_negative_brands = None
        all_negative_categories = None
        
        if negative_brands:
            all_negative_brands = [brand for neg_list in negative_brands for brand in neg_list]
        if negative_categories:
            all_negative_categories = [cat for neg_list in negative_categories for cat in neg_list]
        
        negative_embeddings = self.item_tower(
            all_negative_texts,
            all_negative_brands,
            all_negative_categories
        )  # [total_negatives, embedding_dim]
        
        # Reshape negative embeddings
        num_negatives_per_sample = len(negative_texts[0]) if negative_texts else 0
        negative_embeddings = negative_embeddings.view(
            batch_size, num_negatives_per_sample, -1
        )  # [batch_size, num_negatives, embedding_dim]
        
        # Encode buyers from sequences
        buyer_embeddings_list = []
        for seq, weights in zip(buyer_sequences, [torch.tensor([w for _, w in seq], dtype=torch.float32, device=device) for seq in buyer_sequences]):
            # Get item embeddings for sequence
            if product_embeddings_cache:
                # Use cache if available
                seq_item_embeddings = torch.stack([
                    product_embeddings_cache.get(pid, torch.zeros(self.item_tower.embedding_dim, device=device))
                    for pid, _ in seq
                ]).to(device)
            else:
                # This is a simplified version - in practice, we'd need product texts
                # For now, we'll use a placeholder that should be handled differently
                seq_item_embeddings = torch.zeros(len(seq), self.item_tower.embedding_dim, device=device)
            
            # Add batch dimension
            seq_item_embeddings = seq_item_embeddings.unsqueeze(0)
            weights = weights.unsqueeze(0).to(device)
            
            buyer_emb = self.buyer_tower(seq_item_embeddings, weights)
            buyer_embeddings_list.append(buyer_emb.squeeze(0))
        
        buyer_embeddings = torch.stack(buyer_embeddings_list)  # [batch_size, embedding_dim]
        
        return {
            'buyer_embeddings': buyer_embeddings,
            'positive_embeddings': positive_embeddings,
            'negative_embeddings': negative_embeddings
        }
    
    def forward_simplified(
        self,
        buyer_item_embeddings: torch.Tensor,
        buyer_weights: torch.Tensor,
        positive_texts: List[str],
        negative_texts: List[List[str]],
        positive_brands: Optional[List[str]] = None,
        positive_categories: Optional[List[str]] = None,
        negative_brands: Optional[List[List[str]]] = None,
        negative_categories: Optional[List[List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Simplified forward pass that takes pre-computed item embeddings for buyer sequence.
        
        Args:
            buyer_item_embeddings: Item embeddings for buyer sequences [batch_size, seq_len, embedding_dim]
            buyer_weights: Event weights [batch_size, seq_len]
            positive_texts: List of positive product texts
            negative_texts: List of lists of negative product texts
            positive_brands: Optional list of positive product brands
            positive_categories: Optional list of positive product categories
            negative_brands: Optional list of lists of negative product brands
            negative_categories: Optional list of lists of negative product categories
            
        Returns:
            Dictionary with buyer_embeddings, positive_embeddings, negative_embeddings
        """
        # Encode positive products
        positive_embeddings = self.item_tower(
            positive_texts,
            positive_brands,
            positive_categories
        )  # [batch_size, embedding_dim]
        
        # Encode negative products
        all_negative_texts = [text for neg_list in negative_texts for text in neg_list]
        all_negative_brands = None
        all_negative_categories = None
        
        if negative_brands:
            all_negative_brands = [brand for neg_list in negative_brands for brand in neg_list]
        if negative_categories:
            all_negative_categories = [cat for neg_list in negative_categories for cat in neg_list]
        
        negative_embeddings = self.item_tower(
            all_negative_texts,
            all_negative_brands,
            all_negative_categories
        )  # [total_negatives, embedding_dim]
        
        # Reshape negative embeddings
        batch_size = buyer_item_embeddings.shape[0]
        num_negatives_per_sample = len(negative_texts[0]) if negative_texts else 0
        negative_embeddings = negative_embeddings.view(
            batch_size, num_negatives_per_sample, -1
        )  # [batch_size, num_negatives, embedding_dim]
        
        # Encode buyers
        buyer_embeddings = self.buyer_tower(buyer_item_embeddings, buyer_weights)
        
        return {
            'buyer_embeddings': buyer_embeddings,
            'positive_embeddings': positive_embeddings,
            'negative_embeddings': negative_embeddings
        }

