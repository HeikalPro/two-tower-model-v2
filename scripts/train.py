"""Training script for Two-Tower model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.processor import DataProcessor
from src.data.dataset import TwoTowerDataset, collate_fn
from src.models.item_tower import ItemTower
from src.models.buyer_tower import BuyerTower
from src.models.two_tower import TwoTowerModel
from src.training.trainer import Trainer
from src.utils.config import load_config


def main():
    """Main training function."""
    print("=" * 60)
    print("Two-Tower Model Training")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load and process data
    print("\n1. Loading and processing data...")
    processor = DataProcessor()
    
    print("   Loading events...")
    events_df = processor.load_events()
    print(f"   Loaded {len(events_df)} events")
    
    print("   Loading products...")
    products_df = processor.load_products()
    print(f"   Loaded {len(products_df)} products")
    
    # Create interaction sequences
    print("   Creating interaction sequences...")
    buyer_sequences = processor.create_interaction_sequences(events_df)
    print(f"   Created sequences for {len(buyer_sequences)} buyers")
    
    # Create positive pairs
    print("   Creating positive pairs...")
    positive_pairs = processor.create_positive_pairs(events_df)
    print(f"   Created {len(positive_pairs)} positive pairs")
    
    # Get product metadata
    product_metadata = processor.get_product_metadata(products_df)
    print(f"   Loaded metadata for {len(product_metadata)} products")
    
    # Get all product IDs
    all_product_ids = list(product_metadata.keys())
    
    # Split data
    print("\n2. Splitting data...")
    val_split = config['training']['validation_split']
    train_pairs, val_pairs = train_test_split(
        positive_pairs,
        test_size=val_split,
        random_state=42
    )
    print(f"   Train pairs: {len(train_pairs)}")
    print(f"   Val pairs: {len(val_pairs)}")
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = TwoTowerDataset(
        positive_pairs=train_pairs,
        product_metadata=product_metadata,
        buyer_sequences=buyer_sequences,
        all_product_ids=all_product_ids,
        num_negatives=config['training']['num_negatives']
    )
    
    val_dataset = TwoTowerDataset(
        positive_pairs=val_pairs,
        product_metadata=product_metadata,
        buyer_sequences=buyer_sequences,
        all_product_ids=all_product_ids,
        num_negatives=config['training']['num_negatives']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("\n4. Initializing model...")
    model_config = config['model']
    
    item_tower = ItemTower(
        text_encoder_name=model_config['item_tower']['text_encoder'],
        embedding_dim=model_config['embedding_dim'],
        use_categorical_features=model_config['item_tower']['use_categorical_features'],
        categorical_embedding_dim=model_config['item_tower']['categorical_embedding_dim'],
        projection_hidden_dim=model_config['item_tower']['projection_hidden_dim'],
        freeze_text_encoder=config['training']['freeze_text_encoder']
    )
    
    # Initialize categorical embeddings if needed
    if model_config['item_tower']['use_categorical_features']:
        brands = [p.get('brand') for p in product_metadata.values() if p.get('brand')]
        categories = [p.get('category') for p in product_metadata.values() if p.get('category')]
        item_tower.initialize_categorical_embeddings(
            brand_vocab=brands,
            category_vocab=categories
        )
    
    buyer_tower = BuyerTower(
        embedding_dim=model_config['embedding_dim'],
        aggregation_method=model_config['buyer_tower']['aggregation_method'],
        attention_hidden_dim=model_config['buyer_tower']['attention_hidden_dim']
    )
    
    model = TwoTowerModel(item_tower, buyer_tower)
    
    print(f"   Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    print("\n5. Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path="configs/config.yaml"
    )
    
    # Set product metadata for trainer
    trainer.set_product_metadata(product_metadata)
    
    # Train
    print("\n6. Starting training...")
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

