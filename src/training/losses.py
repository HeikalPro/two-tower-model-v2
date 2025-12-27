"""Loss functions for Two-Tower model training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (Contrastive) loss for retrieval training."""
    
    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling logits
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        buyer_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            buyer_embeddings: Buyer embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive product embeddings [batch_size, embedding_dim]
            negative_embeddings: Negative product embeddings [batch_size, num_negatives, embedding_dim]
            
        Returns:
            Loss scalar
        """
        batch_size = buyer_embeddings.shape[0]
        num_negatives = negative_embeddings.shape[1]
        
        # Compute positive similarities
        positive_sim = (buyer_embeddings * positive_embeddings).sum(dim=1)  # [batch_size]
        positive_sim = positive_sim / self.temperature
        
        # Compute negative similarities
        # buyer_embeddings: [batch_size, embedding_dim]
        # negative_embeddings: [batch_size, num_negatives, embedding_dim]
        negative_sim = torch.bmm(
            buyer_embeddings.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            negative_embeddings.transpose(1, 2)  # [batch_size, embedding_dim, num_negatives]
        ).squeeze(1) / self.temperature  # [batch_size, num_negatives]
        
        # Combine positive and negatives
        # In-batch negatives: use other positives in batch as negatives
        # Expand positive_embeddings to [batch_size, batch_size, embedding_dim]
        # Each row i contains all positive embeddings for computing similarity with buyer i
        all_positive_embeddings = positive_embeddings.unsqueeze(0).expand(
            batch_size, batch_size, -1
        )  # [batch_size, batch_size, embedding_dim]
        in_batch_sim = torch.bmm(
            buyer_embeddings.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            all_positive_embeddings.transpose(1, 2)  # [batch_size, embedding_dim, batch_size]
        ).squeeze(1) / self.temperature  # [batch_size, batch_size]
        
        # Mask out the positive (diagonal)
        mask = torch.eye(batch_size, device=buyer_embeddings.device, dtype=torch.bool)
        in_batch_sim = in_batch_sim.masked_fill(mask, float('-inf'))
        
        # Concatenate all negatives
        all_negatives = torch.cat([negative_sim, in_batch_sim], dim=1)  # [batch_size, num_negatives + batch_size - 1]
        
        # Compute logits: [positive, negatives]
        logits = torch.cat([positive_sim.unsqueeze(1), all_negatives], dim=1)  # [batch_size, 1 + num_negatives + batch_size - 1]
        
        # Labels: 0 is the positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=buyer_embeddings.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss

