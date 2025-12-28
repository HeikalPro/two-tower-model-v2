"""Tests for Buyer Tower model."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.buyer_tower import BuyerTower


def test_buyer_tower_weighted_avg():
    """Test Buyer Tower with weighted average."""
    embedding_dim = 384
    buyer_tower = BuyerTower(
        embedding_dim=embedding_dim,
        aggregation_method="weighted_avg"
    )
    buyer_tower.eval()
    
    batch_size = 2
    seq_len = 5
    
    item_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    weights = torch.tensor([[1.0, 5.0, 10.0, 1.0, 1.0], [1.0, 1.0, 5.0, 5.0, 1.0]])
    
    with torch.no_grad():
        buyer_embeddings = buyer_tower(item_embeddings, weights)
    
    assert buyer_embeddings.shape == (batch_size, embedding_dim)
    
    # Check L2 normalization
    norms = torch.norm(buyer_embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    print("✓ Buyer Tower weighted average test passed")


def test_buyer_tower_attention():
    """Test Buyer Tower with attention."""
    embedding_dim = 384
    buyer_tower = BuyerTower(
        embedding_dim=embedding_dim,
        aggregation_method="attention"
    )
    buyer_tower.eval()
    
    batch_size = 2
    seq_len = 5
    
    item_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    weights = torch.tensor([[1.0, 5.0, 10.0, 1.0, 1.0], [1.0, 1.0, 5.0, 5.0, 1.0]])
    
    with torch.no_grad():
        buyer_embeddings = buyer_tower(item_embeddings, weights)
    
    assert buyer_embeddings.shape == (batch_size, embedding_dim)
    
    # Check L2 normalization
    norms = torch.norm(buyer_embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    print("✓ Buyer Tower attention test passed")


if __name__ == "__main__":
    test_buyer_tower_weighted_avg()
    test_buyer_tower_attention()
    print("All Buyer Tower tests passed!")


