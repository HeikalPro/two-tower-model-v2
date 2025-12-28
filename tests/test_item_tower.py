"""Tests for Item Tower model."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.item_tower import ItemTower
from src.utils.config import load_config


def test_item_tower_initialization():
    """Test Item Tower initialization."""
    config = load_config()
    model_config = config['model']
    
    item_tower = ItemTower(
        text_encoder_name=model_config['item_tower']['text_encoder'],
        embedding_dim=model_config['embedding_dim'],
        use_categorical_features=model_config['item_tower']['use_categorical_features']
    )
    
    assert item_tower.embedding_dim == model_config['embedding_dim']
    print("✓ Item Tower initialization test passed")


def test_item_tower_forward():
    """Test Item Tower forward pass."""
    config = load_config()
    model_config = config['model']
    
    item_tower = ItemTower(
        text_encoder_name=model_config['item_tower']['text_encoder'],
        embedding_dim=model_config['embedding_dim'],
        use_categorical_features=False  # Simpler test
    )
    item_tower.eval()
    
    texts = ["خاتم ذهب", "سلسال ذهب"]
    
    with torch.no_grad():
        embeddings = item_tower(texts)
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == model_config['embedding_dim']
    
    # Check L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    print("✓ Item Tower forward pass test passed")


if __name__ == "__main__":
    test_item_tower_initialization()
    test_item_tower_forward()
    print("All Item Tower tests passed!")


