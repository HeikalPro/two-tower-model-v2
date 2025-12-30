"""Data processing and preprocessing utilities."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from src.utils.config import load_config, get_event_weight


class DataProcessor:
    """Process events and products data for training."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize data processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.event_weights = self.config['event_weights']
        
    def load_events(self, chunk_size: int = 100000) -> pd.DataFrame:
        """Load events data in chunks to handle large files.
        
        Args:
            chunk_size: Number of rows to read per chunk
            
        Returns:
            DataFrame with events data
        """
        events_path = Path(self.config['data']['events_path'])
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        
        chunks = []
        for chunk in pd.read_csv(events_path, chunksize=chunk_size):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Standardize column names
        column_mapping = {
            'distinct_id': 'buyer_id',
            'event_name': 'event_type',
            'created_at': 'timestamp'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Ensure required columns exist
        required_cols = ['buyer_id', 'product_id', 'event_type', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['buyer_id', 'product_id', 'event_type'])
        
        # Normalize event type names
        df['event_type'] = df['event_type'].str.lower().str.replace(' ', '_')
        
        return df
    
    def load_products(self) -> pd.DataFrame:
        """Load products data.
        
        Returns:
            DataFrame with products data
        """
        products_path = Path(self.config['data']['products_path'])
        if not products_path.exists():
            raise FileNotFoundError(f"Products file not found: {products_path}")
        
        # Read products in chunks if file is large
        chunks = []
        chunk_size = 50000
        try:
            for chunk in pd.read_csv(products_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        except:
            # If chunking fails, read directly
            df = pd.read_csv(products_path)
        
        # Ensure product_id column exists
        if 'id' in df.columns and 'product_id' not in df.columns:
            df['product_id'] = df['id']
        
        # Extract metadata
        if 'metadata' in df.columns:
            df = self._extract_metadata(df)
        
        # Combine title and description
        df['text'] = df.apply(
            lambda row: self._combine_text(row.get('title', ''), row.get('description', '')),
            axis=1
        )
        
        # Remove products with missing text
        df = df[df['text'].str.len() > 0]
        
        # Deduplicate products based on content (before embeddings are generated)
        df = self._deduplicate_products(df)
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract brand and category from metadata JSON.
        
        Args:
            df: Products DataFrame with metadata column
            
        Returns:
            DataFrame with extracted brand and category columns
        """
        def extract_field(metadata_str, field):
            try:
                if pd.isna(metadata_str):
                    return None
                metadata = json.loads(metadata_str)
                return metadata.get(field)
            except:
                return None
        
        df['brand'] = df['metadata'].apply(lambda x: extract_field(x, 'brand'))
        df['category'] = df['metadata'].apply(lambda x: extract_field(x, 'catalog_id'))
        
        return df
    
    def _combine_text(self, title: str, description: str) -> str:
        """Combine title and description into single text.
        
        Args:
            title: Product title
            description: Product description
            
        Returns:
            Combined text
        """
        title = str(title) if pd.notna(title) else ""
        description = str(description) if pd.notna(description) else ""
        
        # Clean and combine
        title = title.strip()
        description = description.strip()
        
        if title and description:
            return f"{title} {description}"
        elif title:
            return title
        elif description:
            return description
        else:
            return ""
    
    def create_interaction_sequences(self, events_df: pd.DataFrame) -> Dict[str, List[Tuple[str, int, pd.Timestamp]]]:
        """Create interaction sequences per buyer.
        
        Args:
            events_df: Events DataFrame
            
        Returns:
            Dictionary mapping buyer_id to list of (product_id, weight, timestamp) tuples
        """
        # Sort by timestamp
        events_df = events_df.sort_values('timestamp')
        
        # Group by buyer
        buyer_sequences = defaultdict(list)
        
        for _, row in events_df.iterrows():
            buyer_id = str(row['buyer_id'])
            product_id = str(row['product_id'])
            event_type = row['event_type']
            timestamp = row['timestamp']
            
            # Get event weight
            weight = get_event_weight(event_type, self.config)
            
            buyer_sequences[buyer_id].append((product_id, weight, timestamp))
        
        # Limit sequence length
        max_history = self.config['model']['buyer_tower']['max_interaction_history']
        for buyer_id in buyer_sequences:
            sequences = buyer_sequences[buyer_id]
            if len(sequences) > max_history:
                # Keep most recent interactions
                buyer_sequences[buyer_id] = sequences[-max_history:]
        
        return dict(buyer_sequences)
    
    def create_positive_pairs(self, events_df: pd.DataFrame) -> List[Tuple[str, str, int]]:
        """Create positive (buyer, product) pairs from events.
        
        Args:
            events_df: Events DataFrame
            
        Returns:
            List of (buyer_id, product_id, weight) tuples
        """
        positive_pairs = []
        
        for _, row in events_df.iterrows():
            buyer_id = str(row['buyer_id'])
            product_id = str(row['product_id'])
            event_type = row['event_type']
            weight = get_event_weight(event_type, self.config)
            
            positive_pairs.append((buyer_id, product_id, weight))
        
        return positive_pairs
    
    def get_product_metadata(self, products_df: pd.DataFrame) -> Dict[str, Dict]:
        """Create product metadata dictionary.
        
        Args:
            products_df: Products DataFrame
            
        Returns:
            Dictionary mapping product_id to metadata
        """
        metadata_dict = {}
        
        for _, row in products_df.iterrows():
            product_id = str(row['product_id'])
            metadata_dict[product_id] = {
                'text': row.get('text', ''),
                'brand': row.get('brand'),
                'category': row.get('category'),
                'title': row.get('title', ''),
                'description': row.get('description', '')
            }
        
        return metadata_dict
    
    def _deduplicate_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate products based on content similarity.
        
        Identifies duplicates based on normalized title + description + brand.
        Keeps the first occurrence (the one that appears first in the dataset).
        
        This happens before embeddings are generated, ensuring the FAISS index
        doesn't contain duplicate products with different IDs.
        
        Args:
            df: Products DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        
        # Create a normalized key for comparison (lowercase, stripped, combined title+description+brand)
        df['dedup_key'] = df.apply(
            lambda row: self._create_dedup_key(
                row.get('title', ''),
                row.get('description', ''),
                row.get('brand', '')
            ),
            axis=1
        )
        
        # Keep first occurrence of each duplicate group
        # Sort by dedup_key to ensure consistent results
        df = df.sort_values('dedup_key').drop_duplicates(
            subset=['dedup_key'],
            keep='first'
        )
        
        # Drop the temporary dedup_key column
        df = df.drop(columns=['dedup_key'])
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"Deduplicated products: Removed {removed_count} duplicates ({original_count} -> {len(df)})")
        
        return df
    
    def _create_dedup_key(self, title: str, description: str, brand: str = None) -> str:
        """Create a normalized key for duplicate detection.
        
        Normalizes the product information to identify products that are
        essentially the same despite having different IDs.
        
        Args:
            title: Product title
            description: Product description
            brand: Product brand (optional)
            
        Returns:
            Normalized key string for duplicate detection
        """
        title = str(title) if pd.notna(title) else ""
        description = str(description) if pd.notna(description) else ""
        brand = str(brand) if pd.notna(brand) else ""
        
        # Normalize: lowercase, strip whitespace, remove extra spaces
        title = ' '.join(title.lower().strip().split())
        description = ' '.join(description.lower().strip().split())
        brand = ' '.join(brand.lower().strip().split())
        
        # Combine fields
        key_parts = []
        if title:
            key_parts.append(title)
        # Only add description if it's different from title (to avoid duplication)
        if description and description != title:
            key_parts.append(description)
        if brand:
            key_parts.append(brand)
        
        # Join with separator to create unique key
        return ' || '.join(key_parts)

