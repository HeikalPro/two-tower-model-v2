"""Item Tower model for content-based product embeddings."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, List
import numpy as np


class ItemTower(nn.Module):
    """Item Tower for encoding product content into embeddings."""
    
    def __init__(
        self,
        text_encoder_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: int = 384,
        use_categorical_features: bool = True,
        categorical_embedding_dim: int = 64,
        projection_hidden_dim: int = 256,
        freeze_text_encoder: bool = True
    ):
        """Initialize Item Tower.
        
        Args:
            text_encoder_name: Name of sentence-transformer model
            embedding_dim: Output embedding dimension
            use_categorical_features: Whether to use brand/category features
            categorical_embedding_dim: Dimension for categorical embeddings
            projection_hidden_dim: Hidden dimension for projection layer
            freeze_text_encoder: Whether to freeze pretrained text encoder
        """
        super(ItemTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_categorical_features = use_categorical_features
        
        # Load sentence transformer
        self.text_encoder = SentenceTransformer(text_encoder_name)
        text_embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Categorical feature embeddings
        if use_categorical_features:
            # These will be initialized dynamically based on vocab size
            self.brand_embedding = None
            self.category_embedding = None
            self.categorical_embedding_dim = categorical_embedding_dim
            
            # Projection layer combining text and categorical features
            input_dim = text_embedding_dim + 2 * categorical_embedding_dim
        else:
            input_dim = text_embedding_dim
        
        # Projection to final embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden_dim, embedding_dim)
        )
        
        # Initialize categorical embeddings (will be set during training)
        self._categorical_vocabs = {'brand': set(), 'category': set()}
    
    def initialize_categorical_embeddings(
        self,
        brand_vocab: Optional[List[str]] = None,
        category_vocab: Optional[List[str]] = None
    ):
        """Initialize categorical embedding layers.
        
        Args:
            brand_vocab: List of unique brand values
            category_vocab: List of unique category values
        """
        if not self.use_categorical_features:
            return
        
        if brand_vocab is not None:
            brand_vocab = ['<UNK>'] + sorted(set(brand_vocab))
            self.brand_embedding = nn.Embedding(
                len(brand_vocab),
                self.categorical_embedding_dim,
                padding_idx=0
            )
            self.brand_vocab = {brand: idx for idx, brand in enumerate(brand_vocab)}
        
        if category_vocab is not None:
            category_vocab = ['<UNK>'] + sorted(set(category_vocab))
            self.category_embedding = nn.Embedding(
                len(category_vocab),
                self.categorical_embedding_dim,
                padding_idx=0
            )
            self.category_vocab = {cat: idx for idx, cat in enumerate(category_vocab)}
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using sentence transformer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings tensor [batch_size, text_embedding_dim]
        """
        # Handle empty texts
        texts = [text if text and len(text.strip()) > 0 else " " for text in texts]
        
        # Get device from model parameters
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        with torch.no_grad() if not self.text_encoder.training else torch.enable_grad():
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=False,
                device=str(device)
            )
        
        return embeddings
    
    def encode_categorical(
        self,
        brands: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> Optional[torch.Tensor]:
        """Encode categorical features.
        
        Args:
            brands: List of brand strings
            categories: List of category strings
            
        Returns:
            Categorical embeddings tensor [batch_size, 2 * categorical_embedding_dim] or None
        """
        if not self.use_categorical_features:
            return None
        
        if self.brand_embedding is None or self.category_embedding is None:
            return None
        
        # Get device from model parameters
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        batch_size = len(brands) if brands else len(categories) if categories else 1
        
        # Encode brands
        if brands:
            brand_indices = [
                self.brand_vocab.get(brand, 0) if brand else 0
                for brand in brands
            ]
            brand_emb = self.brand_embedding(torch.tensor(brand_indices, dtype=torch.long, device=device))
        else:
            brand_emb = torch.zeros(batch_size, self.categorical_embedding_dim, device=device)
        
        # Encode categories
        if categories:
            category_indices = [
                self.category_vocab.get(cat, 0) if cat else 0
                for cat in categories
            ]
            category_emb = self.category_embedding(torch.tensor(category_indices, dtype=torch.long, device=device))
        else:
            category_emb = torch.zeros(batch_size, self.categorical_embedding_dim, device=device)
        
        # Concatenate
        return torch.cat([brand_emb, category_emb], dim=1)
    
    def forward(
        self,
        texts: List[str],
        brands: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            texts: List of product text (title + description)
            brands: Optional list of brand strings
            categories: Optional list of category strings
            
        Returns:
            Item embeddings [batch_size, embedding_dim], L2-normalized
        """
        # Encode text
        text_emb = self.encode_text(texts)
        
        # Encode categorical features
        if self.use_categorical_features:
            cat_emb = self.encode_categorical(brands, categories)
            if cat_emb is not None:
                # Concatenate text and categorical embeddings
                combined = torch.cat([text_emb, cat_emb], dim=1)
            else:
                # Fallback to text only if categorical embeddings not initialized
                combined = text_emb
        else:
            combined = text_emb
        
        # Project to final dimension
        embeddings = self.projection(combined)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        brands: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode a batch of products.
        
        Args:
            texts: List of product texts
            brands: Optional list of brands
            categories: Optional list of categories
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings [n_products, embedding_dim]
        """
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_brands = brands[i:i+batch_size] if brands else None
                batch_categories = categories[i:i+batch_size] if categories else None
                
                embeddings = self.forward(batch_texts, batch_brands, batch_categories)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

