"""
Loss functions for multimodal vision-language training.

This module implements loss functions for joint training of vision
classification and language generation tasks.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalLoss(nn.Module):
    """
    Combined loss function for multimodal vision-language tasks.
    
    Combines classification loss for receipt counting and
    language modeling loss for response generation.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        language_weight: float = 1.0,
        ignore_index: int = -100,
    ):
        """
        Initialize multimodal loss function.
        
        Args:
            classification_weight: Weight for classification loss component
            language_weight: Weight for language modeling loss component
            ignore_index: Token value to ignore in language modeling loss
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.language_weight = language_weight
        self.ignore_index = ignore_index
        
        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        classification_labels: torch.Tensor,
        language_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss for multimodal training.
        
        Args:
            model_outputs: Dictionary containing model outputs
            classification_labels: Labels for receipt counting classification
            language_labels: Labels for language generation (token IDs)
            attention_mask: Attention mask for language labels
            
        Returns:
            Dictionary with loss components and total loss
        """
        # Extract outputs
        classification_logits = model_outputs.get("logits")
        language_logits = model_outputs.get("response_logits")
        
        # Initialize losses
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=classification_labels.device)
        
        # Calculate classification loss
        if classification_logits is not None:
            classification_loss = self.classification_loss_fn(classification_logits, classification_labels)
            loss_dict["classification_loss"] = classification_loss
            total_loss += self.classification_weight * classification_loss
        
        # Calculate language modeling loss if labels provided
        if language_logits is not None and language_labels is not None:
            # Check dimensions
            if language_logits.ndim == 3:  # [batch_size, seq_len, vocab_size]
                # Reshape for CrossEntropyLoss
                vocab_size = language_logits.size(-1)
                language_logits_flat = language_logits.contiguous().view(-1, vocab_size)
                language_labels_flat = language_labels.contiguous().view(-1)
                
                # Create or adjust mask to ignore padding tokens
                if attention_mask is not None:
                    # Create mask for tokens to ignore
                    ignore_mask = ~attention_mask.bool()
                    ignore_mask = ignore_mask.view(-1)
                    ignore_indices = ignore_mask.nonzero(as_tuple=True)[0]
                    language_labels_flat[ignore_indices] = self.ignore_index
                
                # Calculate loss using CrossEntropyLoss
                language_loss = F.cross_entropy(
                    language_logits_flat,
                    language_labels_flat,
                    ignore_index=self.ignore_index,
                    reduction="mean",
                )
                
                loss_dict["language_loss"] = language_loss
                total_loss += self.language_weight * language_loss
        
        loss_dict["total_loss"] = total_loss
        return loss_dict