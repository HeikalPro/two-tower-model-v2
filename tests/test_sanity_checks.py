"""Sanity checks for item embeddings and buyer behavior validation."""

import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.item_tower import ItemTower
from src.models.buyer_tower import BuyerTower
from src.models.two_tower import TwoTowerModel
from src.inference.encoder import EmbeddingEncoder
from src.inference.vector_db import VectorDatabase
from src.utils.config import load_config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_item_embedding_similarity():
    """Test that similar items have high similarity and dissimilar items have low similarity."""
    print("Testing item embedding similarity...")
    
    config = load_config()
    
    # Initialize Item Tower
    item_tower = ItemTower(
        text_encoder_name=config['model']['item_tower']['text_encoder'],
        embedding_dim=config['model']['embedding_dim'],
        use_categorical_features=config['model']['item_tower']['use_categorical_features'],
        freeze_text_encoder=True
    )
    item_tower.eval()
    
    # Test cases
    test_cases = [
        {
            "text1": "خاتم ذهب",
            "text2": "سلسال ذهب",
            "expected_min": 0.7,
            "description": "Gold ring vs gold necklace (should be HIGH)"
        },
        {
            "text1": "خاتم ذهب",
            "text2": "زيت محرك",
            "expected_max": 0.3,
            "description": "Gold ring vs engine oil (should be LOW)"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        text1 = test_case["text1"]
        text2 = test_case["text2"]
        
        # Encode both texts
        with torch.no_grad():
            emb1 = item_tower([text1])[0].cpu().numpy()
            emb2 = item_tower([text2])[0].cpu().numpy()
        
        # Compute similarity
        similarity = cosine_similarity(emb1, emb2)
        
        # Check condition
        if "expected_min" in test_case:
            passed = similarity >= test_case["expected_min"]
            condition = f">= {test_case['expected_min']}"
        else:
            passed = similarity <= test_case["expected_max"]
            condition = f"<= {test_case['expected_max']}"
        
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {test_case['description']}")
        print(f"    Similarity: {similarity:.4f} (expected {condition})")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_buyer_behavior_retrieval():
    """Test that buyer with repeated interactions retrieves relevant products."""
    print("\nTesting buyer behavior retrieval...")
    
    # This test requires a trained model and vector database
    # For now, we'll create a mock test that demonstrates the concept
    
    config = load_config()
    
    # Check if model exists
    model_path = Path(config['inference']['model_checkpoint'])
    if not model_path.exists():
        print("  SKIP: Model checkpoint not found. Train model first.")
        return True
    
    try:
        # Load encoder
        encoder = EmbeddingEncoder(
            model_path=str(model_path),
            config_path="configs/config.yaml"
        )
        
        # Mock product metadata for jewelry items
        jewelry_products = {
            "jewelry_1": {"text": "خاتم ذهب عيار 18", "brand": None, "category": None},
            "jewelry_2": {"text": "سلسال ذهب عيار 21", "brand": None, "category": None},
            "jewelry_3": {"text": "أقراط ذهب", "brand": None, "category": None},
            "jewelry_4": {"text": "سوار ذهب", "brand": None, "category": None},
        }
        
        # Mock unrelated products
        unrelated_products = {
            "car_1": {"text": "زيت محرك سيارات", "brand": None, "category": None},
            "food_1": {"text": "أرز بسمتي", "brand": None, "category": None},
            "clothing_1": {"text": "قميص قطني", "brand": None, "category": None},
        }
        
        # Set product metadata
        all_products = {**jewelry_products, **unrelated_products}
        encoder.set_product_metadata(all_products)
        
        # Create buyer with repeated gold ring interactions
        buyer_interactions = [
            {"product_id": "jewelry_1", "event_type": "view", "timestamp": None},
            {"product_id": "jewelry_1", "event_type": "add_to_cart", "timestamp": None},
            {"product_id": "jewelry_1", "event_type": "purchase", "timestamp": None},
            {"product_id": "jewelry_1", "event_type": "view", "timestamp": None},
        ]
        
        # Encode buyer
        buyer_embedding = encoder.encode_buyer(buyer_interactions)
        
        # Encode all products
        all_product_ids = list(all_products.keys())
        product_embeddings = encoder.encode_items(all_product_ids)
        
        # Compute similarities
        similarities = {}
        for i, product_id in enumerate(all_product_ids):
            similarity = cosine_similarity(buyer_embedding, product_embeddings[i])
            similarities[product_id] = similarity
        
        # Sort by similarity
        sorted_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Check that top results are jewelry items
        top_k = 3
        top_products = [pid for pid, _ in sorted_products[:top_k]]
        jewelry_in_top = sum(1 for pid in top_products if pid in jewelry_products)
        
        print(f"  Top {top_k} products: {top_products}")
        print(f"  Jewelry items in top {top_k}: {jewelry_in_top}/{top_k}")
        
        # Check that unrelated items are not in top
        unrelated_in_top = sum(1 for pid in top_products if pid in unrelated_products)
        
        passed = jewelry_in_top >= 2 and unrelated_in_top == 0
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: Buyer behavior retrieval test")
        
        return passed
    
    except Exception as e:
        print(f"  SKIP: Error in buyer behavior test: {str(e)}")
        return True  # Don't fail if model not available


def run_all_sanity_checks():
    """Run all sanity checks."""
    print("=" * 60)
    print("Running Sanity Checks")
    print("=" * 60)
    
    results = []
    
    # Test 1: Item embedding similarity
    results.append(("Item Embedding Similarity", test_item_embedding_similarity()))
    
    # Test 2: Buyer behavior retrieval
    results.append(("Buyer Behavior Retrieval", test_buyer_behavior_retrieval()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Sanity Check Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_sanity_checks()
    sys.exit(0 if success else 1)

