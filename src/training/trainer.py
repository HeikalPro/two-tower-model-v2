"""Training pipeline for Two-Tower model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import json

from src.models.two_tower import TwoTowerModel
from src.models.item_tower import ItemTower
from src.models.buyer_tower import BuyerTower
from src.training.losses import InfoNCELoss
from src.data.dataset import TwoTowerDataset, collate_fn
from src.utils.config import load_config


class Trainer:
    """Trainer for Two-Tower model."""
    
    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config_path: str = "configs/config.yaml"
    ):
        """Initialize trainer.
        
        Args:
            model: Two-Tower model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            config_path: Path to configuration file
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = load_config(config_path)
        
        # Setup device
        self.device = torch.device(self.config['inference']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        train_config = self.config['training']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config['learning_rate']
        )
        
        # Setup loss
        self.criterion = InfoNCELoss(temperature=train_config['temperature'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(train_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Product metadata cache for encoding sequences
        self.product_metadata = None
    
    def set_product_metadata(self, product_metadata: Dict):
        """Set product metadata for encoding buyer sequences.
        
        Args:
            product_metadata: Dictionary mapping product_id to metadata
        """
        self.product_metadata = product_metadata
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # Move batch to device
            buyer_ids = batch['buyer_ids']
            positive_product_ids = batch['positive_product_ids']
            negative_product_ids = batch['negative_product_ids']
            buyer_sequences = batch['buyer_sequences']
            positive_texts = batch['positive_product_texts']
            negative_texts = batch['negative_product_texts']
            weights = batch['weights'].to(self.device)
            
            # Encode buyer sequences
            # For each buyer, get item embeddings from their sequence
            buyer_item_embeddings_list = []
            buyer_weights_list = []
            
            for seq, seq_weights in zip(buyer_sequences, weights):
                # Get product texts for sequence
                seq_texts = []
                seq_weights_tensor = []
                
                # Handle tuples: (product_id, weight) or (product_id, weight, timestamp)
                for item in seq:
                    if len(item) == 3:
                        product_id, weight, _ = item  # Unpack (product_id, weight, timestamp)
                    elif len(item) == 2:
                        product_id, weight = item  # Unpack (product_id, weight)
                    else:
                        continue  # Skip invalid entries
                    
                    if self.product_metadata and product_id in self.product_metadata:
                        seq_texts.append(self.product_metadata[product_id]['text'])
                        seq_weights_tensor.append(weight)
                
                if len(seq_texts) == 0:
                    # Fallback: use positive product text
                    seq_texts = [positive_texts[len(buyer_item_embeddings_list)]]
                    seq_weights_tensor = [1.0]
                
                # Encode sequence items
                with torch.no_grad():
                    seq_item_emb = self.model.item_tower.encode_text(seq_texts)
                
                # Pad or truncate to fixed length
                max_seq_len = self.config['model']['buyer_tower']['max_interaction_history']
                if len(seq_item_emb) > max_seq_len:
                    seq_item_emb = seq_item_emb[-max_seq_len:]
                    seq_weights_tensor = seq_weights_tensor[-max_seq_len:]
                
                # Pad if needed
                if len(seq_item_emb) < max_seq_len:
                    padding = torch.zeros(max_seq_len - len(seq_item_emb), seq_item_emb.shape[1])
                    seq_item_emb = torch.cat([seq_item_emb, padding])
                    seq_weights_tensor.extend([0.0] * (max_seq_len - len(seq_weights_tensor)))
                
                buyer_item_embeddings_list.append(seq_item_emb)
                buyer_weights_list.append(torch.tensor(seq_weights_tensor, dtype=torch.float32))
            
            buyer_item_embeddings = torch.stack(buyer_item_embeddings_list).to(self.device)
            buyer_weights = torch.stack(buyer_weights_list).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get positive and negative brands/categories if available
            positive_brands = None
            positive_categories = None
            negative_brands = None
            negative_categories = None
            
            if self.product_metadata:
                positive_brands = [
                    self.product_metadata.get(pid, {}).get('brand') 
                    for pid in positive_product_ids
                ]
                positive_categories = [
                    self.product_metadata.get(pid, {}).get('category')
                    for pid in positive_product_ids
                ]
                
                negative_brands = [
                    [self.product_metadata.get(pid, {}).get('brand') for pid in neg_list]
                    for neg_list in negative_product_ids
                ]
                negative_categories = [
                    [self.product_metadata.get(pid, {}).get('category') for pid in neg_list]
                    for neg_list in negative_product_ids
                ]
            
            outputs = self.model.forward_simplified(
                buyer_item_embeddings,
                buyer_weights,
                positive_texts,
                negative_texts,
                positive_brands,
                positive_categories,
                negative_brands,
                negative_categories
            )
            
            # Compute loss
            loss = self.criterion(
                outputs['buyer_embeddings'],
                outputs['positive_embeddings'],
                outputs['negative_embeddings']
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """Validate model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Similar to train_epoch but without gradient computation
                buyer_ids = batch['buyer_ids']
                positive_product_ids = batch['positive_product_ids']
                negative_product_ids = batch['negative_product_ids']
                buyer_sequences = batch['buyer_sequences']
                positive_texts = batch['positive_product_texts']
                negative_texts = batch['negative_product_texts']
                weights = batch['weights'].to(self.device)
                
                # Encode buyer sequences (same as training)
                buyer_item_embeddings_list = []
                buyer_weights_list = []
                
                for seq, seq_weights in zip(buyer_sequences, weights):
                    seq_texts = []
                    seq_weights_tensor = []
                    
                    # Handle tuples: (product_id, weight) or (product_id, weight, timestamp)
                    for item in seq:
                        if len(item) == 3:
                            product_id, weight, _ = item  # Unpack (product_id, weight, timestamp)
                        elif len(item) == 2:
                            product_id, weight = item  # Unpack (product_id, weight)
                        else:
                            continue  # Skip invalid entries
                        
                        if self.product_metadata and product_id in self.product_metadata:
                            seq_texts.append(self.product_metadata[product_id]['text'])
                            seq_weights_tensor.append(weight)
                    
                    if len(seq_texts) == 0:
                        seq_texts = [positive_texts[len(buyer_item_embeddings_list)]]
                        seq_weights_tensor = [1.0]
                    
                    seq_item_emb = self.model.item_tower.encode_text(seq_texts)
                    
                    max_seq_len = self.config['model']['buyer_tower']['max_interaction_history']
                    if len(seq_item_emb) > max_seq_len:
                        seq_item_emb = seq_item_emb[-max_seq_len:]
                        seq_weights_tensor = seq_weights_tensor[-max_seq_len:]
                    
                    if len(seq_item_emb) < max_seq_len:
                        padding = torch.zeros(max_seq_len - len(seq_item_emb), seq_item_emb.shape[1])
                        seq_item_emb = torch.cat([seq_item_emb, padding])
                        seq_weights_tensor.extend([0.0] * (max_seq_len - len(seq_weights_tensor)))
                    
                    buyer_item_embeddings_list.append(seq_item_emb)
                    buyer_weights_list.append(torch.tensor(seq_weights_tensor, dtype=torch.float32))
                
                buyer_item_embeddings = torch.stack(buyer_item_embeddings_list).to(self.device)
                buyer_weights = torch.stack(buyer_weights_list).to(self.device)
                
                # Get metadata
                positive_brands = None
                positive_categories = None
                negative_brands = None
                negative_categories = None
                
                if self.product_metadata:
                    positive_brands = [
                        self.product_metadata.get(pid, {}).get('brand')
                        for pid in positive_product_ids
                    ]
                    positive_categories = [
                        self.product_metadata.get(pid, {}).get('category')
                        for pid in positive_product_ids
                    ]
                    
                    negative_brands = [
                        [self.product_metadata.get(pid, {}).get('brand') for pid in neg_list]
                        for neg_list in negative_product_ids
                    ]
                    negative_categories = [
                        [self.product_metadata.get(pid, {}).get('category') for pid in neg_list]
                        for neg_list in negative_product_ids
                    ]
                
                outputs = self.model.forward_simplified(
                    buyer_item_embeddings,
                    buyer_weights,
                    positive_texts,
                    negative_texts,
                    positive_brands,
                    positive_categories,
                    negative_brands,
                    negative_categories
                )
                
                loss = self.criterion(
                    outputs['buyer_embeddings'],
                    outputs['positive_embeddings'],
                    outputs['negative_embeddings']
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config['training']['num_epochs']
        save_every = self.config['training']['save_every_n_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print("Training completed!")

