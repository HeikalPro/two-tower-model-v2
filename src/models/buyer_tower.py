"""Buyer Tower model for behavior-based buyer embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BuyerTower(nn.Module):
    """Buyer Tower for encoding buyer behavior into embeddings."""
    
    def __init__(
        self,
        embedding_dim: int = 384,
        aggregation_method: str = "attention",
        attention_hidden_dim: int = 128
    ):
        """Initialize Buyer Tower.
        
        Args:
            embedding_dim: Dimension of item embeddings (input/output)
            aggregation_method: "weighted_avg" or "attention"
            attention_hidden_dim: Hidden dimension for attention mechanism
        """
        super(BuyerTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        
        if aggregation_method == "attention":
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(attention_hidden_dim, 1)
            )
        elif aggregation_method == "weighted_avg":
            # No learnable parameters for weighted average
            pass
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def weighted_average(
        self,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted average of item embeddings.
        
        Args:
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            weights: Event weights [batch_size, seq_len]
            
        Returns:
            Aggregated buyer embeddings [batch_size, embedding_dim]
        """
        # Normalize weights
        weights = weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_weights = weights / weights_sum
        
        # Weighted sum
        buyer_embeddings = (item_embeddings * normalized_weights).sum(dim=1)
        
        # L2 normalization
        buyer_embeddings = F.normalize(buyer_embeddings, p=2, dim=1)
        
        return buyer_embeddings
    
    def attention_aggregation(
        self,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention-weighted aggregation of item embeddings.
        
        Args:
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            weights: Event weights [batch_size, seq_len]
            
        Returns:
            Aggregated buyer embeddings [batch_size, embedding_dim]
        """
        # Compute attention scores
        attention_scores = self.attention(item_embeddings)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Combine with event weights (element-wise multiplication)
        combined_scores = attention_scores * weights
        
        # Softmax over sequence
        attention_weights = F.softmax(combined_scores, dim=1)  # [batch_size, seq_len]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Weighted sum
        buyer_embeddings = (item_embeddings * attention_weights).sum(dim=1)
        
        # L2 normalization
        buyer_embeddings = F.normalize(buyer_embeddings, p=2, dim=1)
        
        return buyer_embeddings
    
    def forward(
        self,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            weights: Event weights [batch_size, seq_len]
            
        Returns:
            Buyer embeddings [batch_size, embedding_dim], L2-normalized
        """
        if self.aggregation_method == "weighted_avg":
            return self.weighted_average(item_embeddings, weights)
        elif self.aggregation_method == "attention":
            return self.attention_aggregation(item_embeddings, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def encode_from_sequence(
        self,
        item_embeddings: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Encode buyer from interaction sequence.
        
        Args:
            item_embeddings: Item embeddings [seq_len, embedding_dim] or [1, seq_len, embedding_dim]
            weights: Event weights [seq_len] or [1, seq_len]
            
        Returns:
            Buyer embedding [embedding_dim] or [1, embedding_dim]
        """
        # Ensure batch dimension
        if item_embeddings.dim() == 2:
            item_embeddings = item_embeddings.unsqueeze(0)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        
        return self.forward(item_embeddings, weights)

