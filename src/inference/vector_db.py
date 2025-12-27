"""FAISS vector database for product retrieval."""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


class VectorDatabase:
    """FAISS-based vector database for product retrieval."""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize vector database.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.product_ids = None
        self.id_to_index = None
        self.index_to_id = None
    
    def build_index(
        self,
        embeddings: np.ndarray,
        product_ids: List[str]
    ):
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: Product embeddings [n_products, embedding_dim]
            product_ids: List of product IDs
        """
        n_products, dim = embeddings.shape
        
        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {dim}"
            )
        
        # Ensure embeddings are L2-normalized (required for cosine similarity with inner product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Create FAISS index (IndexFlatIP for inner product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Convert to float32 (required by FAISS)
        embeddings_normalized = embeddings_normalized.astype(np.float32)
        
        # Add embeddings to index
        self.index.add(embeddings_normalized)
        
        # Store product IDs and mappings
        self.product_ids = product_ids
        self.id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}
        self.index_to_id = {idx: pid for idx, pid in enumerate(product_ids)}
        
        print(f"Built FAISS index with {n_products} products")
    
    def load_index(
        self,
        index_path: str,
        product_ids_path: Optional[str] = None,
        mapping_path: Optional[str] = None
    ):
        """Load FAISS index from disk.
        
        Args:
            index_path: Path to FAISS index file
            product_ids_path: Path to product IDs file
            mapping_path: Path to product ID to index mapping file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load product IDs
        if product_ids_path:
            product_ids_array = np.load(product_ids_path, allow_pickle=True)
            self.product_ids = product_ids_array.tolist()
        else:
            # Infer from index size
            n_products = self.index.ntotal
            self.product_ids = [f"product_{i}" for i in range(n_products)]
        
        # Load mapping if available
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.id_to_index = json.load(f)
            self.index_to_id = {v: k for k, v in self.id_to_index.items()}
        else:
            # Create default mapping
            self.id_to_index = {pid: idx for idx, pid in enumerate(self.product_ids)}
            self.index_to_id = {idx: pid for idx, pid in enumerate(self.product_ids)}
        
        print(f"Loaded FAISS index with {len(self.product_ids)} products")
    
    def save_index(
        self,
        index_path: str,
        product_ids_path: Optional[str] = None,
        mapping_path: Optional[str] = None
    ):
        """Save FAISS index to disk.
        
        Args:
            index_path: Path to save FAISS index
            product_ids_path: Path to save product IDs
            mapping_path: Path to save product ID to index mapping
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save product IDs
        if product_ids_path:
            np.save(product_ids_path, np.array(self.product_ids))
        
        # Save mapping
        if mapping_path and self.id_to_index:
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.id_to_index, f, ensure_ascii=False, indent=2)
        
        print(f"Saved FAISS index to {index_path}")
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Retrieve top-K products for a query embedding.
        
        Args:
            query_embedding: Query embedding [embedding_dim] or [1, embedding_dim]
            k: Number of products to retrieve
            
        Returns:
            List of (product_id, score) tuples, sorted by score (descending)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure L2-normalized
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / (norm + 1e-8)
        
        # Convert to float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert to product IDs and scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                results.append((product_id, float(score)))
        
        return results
    
    def retrieve_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """Retrieve top-K products for multiple query embeddings.
        
        Args:
            query_embeddings: Query embeddings [n_queries, embedding_dim]
            k: Number of products to retrieve per query
            
        Returns:
            List of lists of (product_id, score) tuples for each query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        # Ensure L2-normalized
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (norms + 1e-8)
        
        # Convert to float32
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embeddings, k)
        
        # Convert to product IDs and scores
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for idx, score in zip(query_indices, query_scores):
                if idx < len(self.product_ids):
                    product_id = self.product_ids[idx]
                    results.append((product_id, float(score)))
            all_results.append(results)
        
        return all_results
    
    def get_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific product ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product embedding or None if not found
        """
        if self.index is None or self.id_to_index is None:
            return None
        
        if product_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[product_id]
        
        # Reconstruct embedding from index (FAISS doesn't support direct retrieval)
        # This is a limitation - we'd need to store embeddings separately for this
        # For now, return None and suggest using the saved embeddings file
        return None

