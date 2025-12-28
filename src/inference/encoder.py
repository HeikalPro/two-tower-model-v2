"""Embedding generation for inference."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from src.models.two_tower import TwoTowerModel
from src.models.item_tower import ItemTower
from src.models.buyer_tower import BuyerTower
from src.utils.config import load_config, get_event_weight


class EmbeddingEncoder:
    """Encoder for generating embeddings at inference time."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config.yaml",
        device: Optional[str] = None
    ):
        """Initialize encoder.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = load_config(config_path)
        self.device = torch.device(
            device or self.config['inference']['device']
            if torch.cuda.is_available() else 'cpu'
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Product metadata cache
        self.product_metadata = None
    
    def _load_model(self, model_path: str) -> TwoTowerModel:
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded Two-Tower model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use default
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
        else:
            model_config = self.config['model']
        
        # Check if categorical embeddings exist in checkpoint
        state_dict = checkpoint['model_state_dict']
        brand_embedding_key = 'item_tower.brand_embedding.weight'
        category_embedding_key = 'item_tower.category_embedding.weight'
        
        # Initialize Item Tower
        item_tower = ItemTower(
            text_encoder_name=model_config['item_tower']['text_encoder'],
            embedding_dim=model_config['embedding_dim'],
            use_categorical_features=model_config['item_tower']['use_categorical_features'],
            categorical_embedding_dim=model_config['item_tower']['categorical_embedding_dim'],
            projection_hidden_dim=model_config['item_tower']['projection_hidden_dim'],
            freeze_text_encoder=self.config['training']['freeze_text_encoder']
        )
        
        # Initialize categorical embeddings if they exist in checkpoint
        if model_config['item_tower']['use_categorical_features']:
            # Try to load vocabs from checkpoint (preferred method)
            brand_vocab = None
            category_vocab = None
            
            if 'brand_vocab' in checkpoint:
                # Reconstruct brand vocab list from dict (dict maps brand -> index)
                brand_vocab_dict = checkpoint['brand_vocab']
                brand_vocab = [''] * len(brand_vocab_dict)
                for brand, idx in brand_vocab_dict.items():
                    brand_vocab[idx] = brand
            
            if 'category_vocab' in checkpoint:
                # Reconstruct category vocab list from dict (dict maps category -> index)
                category_vocab_dict = checkpoint['category_vocab']
                category_vocab = [''] * len(category_vocab_dict)
                for category, idx in category_vocab_dict.items():
                    category_vocab[idx] = category
            
            # If vocabs not in checkpoint, infer sizes from state_dict and create dummy vocabs
            # Note: initialize_categorical_embeddings adds '<UNK>' automatically, so we need size-1
            if brand_vocab is None and brand_embedding_key in state_dict:
                brand_vocab_size = state_dict[brand_embedding_key].shape[0]
                # Create dummy vocab of size-1 (UNK will be added by initialize_categorical_embeddings)
                # This ensures final size matches checkpoint: ['<UNK>'] + [dummy_1, ..., dummy_N-1] = N total
                brand_vocab = [f'brand_{i}' for i in range(1, brand_vocab_size)]
            
            if category_vocab is None and category_embedding_key in state_dict:
                category_vocab_size = state_dict[category_embedding_key].shape[0]
                # Create dummy vocab of size-1 (UNK will be added by initialize_categorical_embeddings)
                category_vocab = [f'category_{i}' for i in range(1, category_vocab_size)]
            
            # Initialize embeddings if we have vocab info
            if brand_vocab is not None or category_vocab is not None:
                item_tower.initialize_categorical_embeddings(
                    brand_vocab=brand_vocab,
                    category_vocab=category_vocab
                )
        
        # Initialize Buyer Tower
        buyer_tower = BuyerTower(
            embedding_dim=model_config['embedding_dim'],
            aggregation_method=model_config['buyer_tower']['aggregation_method'],
            attention_hidden_dim=model_config['buyer_tower']['attention_hidden_dim']
        )
        
        # Create Two-Tower model
        model = TwoTowerModel(item_tower, buyer_tower)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def set_product_metadata(self, product_metadata: Dict):
        """Set product metadata for encoding.
        
        Args:
            product_metadata: Dictionary mapping product_id to metadata
        """
        self.product_metadata = product_metadata
        
        # If categorical embeddings were loaded with dummy vocabs, reconstruct from metadata
        item_tower = self.model.item_tower
        if (item_tower.use_categorical_features and 
            hasattr(item_tower, 'brand_embedding') and 
            item_tower.brand_embedding is not None):
            
            # Check if we're using dummy vocabs (vocabs that start with 'brand_' or 'category_')
            needs_reconstruction = False
            if hasattr(item_tower, 'brand_vocab'):
                # Check if first non-UNK entry is a dummy
                vocab_list = sorted([k for k in item_tower.brand_vocab.keys() if k != '<UNK>'])
                if vocab_list and vocab_list[0].startswith('brand_'):
                    needs_reconstruction = True
            
            if needs_reconstruction:
                # Reconstruct vocabs from product metadata (same way as training)
                brands = [p.get('brand') for p in product_metadata.values() if p.get('brand')]
                categories = [p.get('category') for p in product_metadata.values() if p.get('category')]
                
                # Get exact vocab sizes from embedding layers (these match the checkpoint)
                brand_vocab_size = item_tower.brand_embedding.num_embeddings if item_tower.brand_embedding is not None else None
                category_vocab_size = item_tower.category_embedding.num_embeddings if item_tower.category_embedding is not None else None
                
                # Reconstruct vocabs in the same order as training: ['<UNK>'] + sorted(unique_values)
                # IMPORTANT: Must match exact size from checkpoint (embedding layer size)
                if brands and brand_vocab_size:
                    unique_brands = sorted(set(brands))
                    # Create vocab list: UNK at 0, then actual brands, pad/truncate to exact size
                    # The embedding layer already exists with size brand_vocab_size, we just update the mapping
                    if len(unique_brands) + 1 <= brand_vocab_size:
                        # We have enough or fewer brands - pad with dummies if needed
                        reconstructed_brand_vocab = ['<UNK>'] + unique_brands
                        # Pad with dummy values to match exact embedding size
                        while len(reconstructed_brand_vocab) < brand_vocab_size:
                            reconstructed_brand_vocab.append(f'brand_dummy_{len(reconstructed_brand_vocab)}')
                    else:
                        # Too many brands - truncate to match exact size
                        reconstructed_brand_vocab = ['<UNK>'] + unique_brands[:brand_vocab_size-1]
                    
                    # Ensure exact size match
                    assert len(reconstructed_brand_vocab) == brand_vocab_size, \
                        f"Brand vocab size mismatch: {len(reconstructed_brand_vocab)} != {brand_vocab_size}"
                    
                    # Update vocab dictionary (this doesn't change embedding layer, just the string->index mapping)
                    item_tower.brand_vocab = {brand: idx for idx, brand in enumerate(reconstructed_brand_vocab)}
                
                if categories and category_vocab_size:
                    unique_categories = sorted(set(categories))
                    # Create vocab list: UNK at 0, then actual categories, pad/truncate to exact size
                    if len(unique_categories) + 1 <= category_vocab_size:
                        # We have enough or fewer categories - pad with dummies if needed
                        reconstructed_category_vocab = ['<UNK>'] + unique_categories
                        # Pad with dummy values to match exact embedding size
                        while len(reconstructed_category_vocab) < category_vocab_size:
                            reconstructed_category_vocab.append(f'category_dummy_{len(reconstructed_category_vocab)}')
                    else:
                        # Too many categories - truncate to match exact size
                        reconstructed_category_vocab = ['<UNK>'] + unique_categories[:category_vocab_size-1]
                    
                    # Ensure exact size match
                    assert len(reconstructed_category_vocab) == category_vocab_size, \
                        f"Category vocab size mismatch: {len(reconstructed_category_vocab)} != {category_vocab_size}"
                    
                    # Update vocab dictionary (this doesn't change embedding layer, just the string->index mapping)
                    item_tower.category_vocab = {cat: idx for idx, cat in enumerate(reconstructed_category_vocab)}
    
    def encode_items(
        self,
        product_ids: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode products into embeddings (offline).
        
        Args:
            product_ids: List of product IDs
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings [n_products, embedding_dim]
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before encoding items")
        
        # Get texts and metadata
        texts = []
        brands = []
        categories = []
        
        for product_id in product_ids:
            metadata = self.product_metadata.get(product_id, {})
            texts.append(metadata.get('text', ''))
            brands.append(metadata.get('brand'))
            categories.append(metadata.get('category'))
        
        # Encode in batches
        embeddings = self.model.item_tower.encode_batch(
            texts,
            brands if self.config['model']['item_tower']['use_categorical_features'] else None,
            categories if self.config['model']['item_tower']['use_categorical_features'] else None,
            batch_size=batch_size
        )
        
        return embeddings
    
    def encode_buyer(
        self,
        interactions: List[Dict[str, any]]
    ) -> np.ndarray:
        """Encode buyer from recent interactions (online).
        
        Args:
            interactions: List of interaction dicts with keys:
                - product_id: str
                - event_type: str (e.g., 'view', 'add_to_cart', 'purchase')
                - timestamp: str (optional)
        
        Returns:
            Buyer embedding [embedding_dim]
        """
        if self.product_metadata is None:
            raise ValueError("Product metadata must be set before encoding buyers")
        
        # Sort by timestamp if available
        if all('timestamp' in interaction for interaction in interactions):
            interactions = sorted(interactions, key=lambda x: x['timestamp'])
        
        # Limit to max history
        max_history = self.config['model']['buyer_tower']['max_interaction_history']
        interactions = interactions[-max_history:]
        
        # Get product texts and weights
        product_ids = [interaction['product_id'] for interaction in interactions]
        event_types = [interaction['event_type'] for interaction in interactions]
        weights = [get_event_weight(et, self.config) for et in event_types]
        
        # Get product texts
        texts = []
        brands = []
        categories = []
        
        for product_id in product_ids:
            metadata = self.product_metadata.get(product_id, {})
            texts.append(metadata.get('text', ''))
            brands.append(metadata.get('brand'))
            categories.append(metadata.get('category'))
        
        # Encode items
        with torch.no_grad():
            item_embeddings = self.model.item_tower(
                texts,
                brands if self.config['model']['item_tower']['use_categorical_features'] else None,
                categories if self.config['model']['item_tower']['use_categorical_features'] else None
            )  # [seq_len, embedding_dim]
        
        # Convert weights to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Add batch dimension
        item_embeddings = item_embeddings.unsqueeze(0)  # [1, seq_len, embedding_dim]
        weights_tensor = weights_tensor.unsqueeze(0)  # [1, seq_len]
        
        # Encode buyer
        with torch.no_grad():
            buyer_embedding = self.model.buyer_tower(item_embeddings, weights_tensor)
        
        return buyer_embedding.squeeze(0).cpu().numpy()
    
    def save_item_embeddings(
        self,
        product_ids: List[str],
        embeddings: np.ndarray,
        output_dir: str
    ):
        """Save item embeddings to disk.
        
        Args:
            product_ids: List of product IDs
            embeddings: Embedding array [n_products, embedding_dim]
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_path = output_dir / "product_embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Save product IDs
        ids_path = output_dir / "product_ids.npy"
        np.save(ids_path, np.array(product_ids))
        
        # Save mapping
        mapping_path = output_dir / "product_id_to_index.json"
        id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(id_to_index, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(product_ids)} item embeddings to {output_dir}")

