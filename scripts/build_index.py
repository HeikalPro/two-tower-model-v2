"""Script to build FAISS index from product embeddings."""

import sys
from pathlib import Path

# Add project root to path (resolve to absolute path for reliability)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from src.inference.vector_db import VectorDatabase
from src.utils.config import load_config


def main():
    """Build FAISS index from embeddings."""
    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load embeddings
    print("\n1. Loading embeddings...")
    embeddings_dir = Path(config['inference']['embeddings_dir'])
    embeddings_path = embeddings_dir / "product_embeddings.npy"
    product_ids_path = embeddings_dir / "product_ids.npy"
    mapping_path = embeddings_dir / "product_id_to_index.json"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    product_ids = np.load(product_ids_path, allow_pickle=True).tolist()
    
    print(f"   Loaded {len(product_ids)} embeddings")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Build index
    print("\n2. Building FAISS index...")
    vector_db = VectorDatabase(embedding_dim=config['model']['embedding_dim'])
    vector_db.build_index(embeddings, product_ids)
    
    # Save index
    print("\n3. Saving index...")
    index_dir = Path(config['inference']['index_dir'])
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = index_dir / "product_index.faiss"
    product_ids_save_path = index_dir / "product_ids.npy"
    mapping_save_path = index_dir / "product_id_to_index.json"
    
    vector_db.save_index(
        str(index_path),
        str(product_ids_save_path),
        str(mapping_save_path)
    )
    
    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

