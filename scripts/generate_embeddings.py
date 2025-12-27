"""Script to generate item embeddings for all products."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.processor import DataProcessor
from src.inference.encoder import EmbeddingEncoder
from src.utils.config import load_config


def main():
    """Generate embeddings for all products."""
    print("=" * 60)
    print("Generating Item Embeddings")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load products
    print("\n1. Loading products...")
    processor = DataProcessor()
    products_df = processor.load_products()
    print(f"   Loaded {len(products_df)} products")
    
    # Get product metadata
    product_metadata = processor.get_product_metadata(products_df)
    product_ids = list(product_metadata.keys())
    
    # Load encoder
    print("\n2. Loading model...")
    model_path = config['inference']['model_checkpoint']
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    encoder = EmbeddingEncoder(
        model_path=model_path,
        config_path="configs/config.yaml"
    )
    encoder.set_product_metadata(product_metadata)
    
    # Generate embeddings
    print("\n3. Generating embeddings...")
    print(f"   Processing {len(product_ids)} products...")
    
    batch_size = 64
    embeddings = encoder.encode_items(product_ids, batch_size=batch_size)
    
    print(f"   Generated embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    print("\n4. Saving embeddings...")
    output_dir = config['inference']['embeddings_dir']
    encoder.save_item_embeddings(product_ids, embeddings, output_dir)
    
    print("\n" + "=" * 60)
    print("Embedding generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

