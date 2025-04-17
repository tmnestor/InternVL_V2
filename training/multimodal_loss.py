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
        detail_weight: float = 0.5,
        contrastive_weight: float = 0.2,
        ignore_index: int = -100,
        use_dynamic_weighting: bool = False,
    ):
        """
        Initialize multimodal loss function.
        
        Args:
            classification_weight: Weight for classification loss component
            language_weight: Weight for language modeling loss component
            detail_weight: Weight for detail extraction loss component
            contrastive_weight: Weight for contrastive learning component
            ignore_index: Token value to ignore in language modeling loss
            use_dynamic_weighting: Whether to use dynamic loss weighting based on training progress
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.language_weight = language_weight
        self.detail_weight = detail_weight
        self.contrastive_weight = contrastive_weight
        self.ignore_index = ignore_index
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Track iteration steps for dynamic weighting
        self.steps = 0
        self.warmup_steps = 1000
        
        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.detail_loss_fn = nn.BCEWithLogitsLoss()
    
    def _calculate_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Calculate contrastive loss to improve separation between different classes.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            temperature: Temperature parameter for scaling
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings to unit sphere
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.t()) / temperature
        
        # Create label mask (1 if same class, 0 otherwise)
        labels_expanded = labels.expand(labels.size(0), labels.size(0))
        labels_match = (labels_expanded == labels_expanded.t()).float()
        
        # Ignore self-similarity
        mask_self = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        
        # Apply masks
        similarity_matrix = similarity_matrix.masked_fill(~mask_self, -1e9)
        
        # Calculate positive and negative similarities
        positive_similarity = (similarity_matrix * labels_match).sum(dim=1)
        negative_similarity = torch.logsumexp(similarity_matrix * (1 - labels_match) + 
                                               -1e9 * labels_match, dim=1)
        
        # Calculate loss (larger positive similarity, smaller negative similarity)
        contrastive_loss = -(positive_similarity - negative_similarity).mean()
        
        return contrastive_loss
        
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        classification_labels: torch.Tensor,
        language_labels: Optional[torch.Tensor] = None,
        detail_labels: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_type_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss for multimodal training.
        
        Args:
            model_outputs: Dictionary containing model outputs
            classification_labels: Labels for receipt counting classification
            language_labels: Labels for language generation (token IDs)
            detail_labels: Labels for detail extraction tasks
            attention_mask: Attention mask for language labels
            question_type_labels: Labels for question type classification
            
        Returns:
            Dictionary with loss components and total loss
        """
        # Extract outputs
        classification_logits = model_outputs.get("logits")
        language_logits = model_outputs.get("response_logits")
        embeddings = model_outputs.get("embeddings")
        detail_logits = model_outputs.get("detail_logits")
        question_type_logits = model_outputs.get("question_type_logits")
        
        # Update step counter for dynamic weighting
        if self.use_dynamic_weighting:
            self.steps += 1
        
        # Initialize losses
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=classification_labels.device)
        
        # Calculate classification loss
        if classification_logits is not None:
            classification_loss = self.classification_loss_fn(classification_logits, classification_labels)
            loss_dict["classification_loss"] = classification_loss
            
            # Apply weight (either static or dynamic)
            cls_weight = self.classification_weight
            if self.use_dynamic_weighting:
                # Linearly decrease weight after warmup
                progress = min(1.0, self.steps / self.warmup_steps)
                cls_weight = max(0.5, self.classification_weight * (1.1 - 0.1 * progress))
                
            total_loss += cls_weight * classification_loss
        
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
                
                # Apply weight (either static or dynamic)
                lang_weight = self.language_weight
                if self.use_dynamic_weighting:
                    # Linearly increase weight after warmup
                    progress = min(1.0, self.steps / self.warmup_steps)
                    lang_weight = min(1.5, self.language_weight * (0.9 + 0.1 * progress))
                    
                total_loss += lang_weight * language_loss
        
        # Calculate detail extraction loss if labels provided
        if detail_logits is not None and detail_labels is not None:
            detail_losses = []
            for detail_type, logits in detail_logits.items():
                if detail_type in detail_labels:
                    detail_loss = self.detail_loss_fn(logits, detail_labels[detail_type])
                    detail_losses.append(detail_loss)
            
            if detail_losses:
                avg_detail_loss = torch.stack(detail_losses).mean()
                loss_dict["detail_loss"] = avg_detail_loss
                total_loss += self.detail_weight * avg_detail_loss
        
        # Calculate question type classification loss if labels provided
        if question_type_logits is not None and question_type_labels is not None:
            question_type_loss = self.classification_loss_fn(question_type_logits, question_type_labels)
            loss_dict["question_type_loss"] = question_type_loss
            total_loss += self.detail_weight * question_type_loss  # Use same weight as detail extraction
        
        # Calculate contrastive loss if embeddings are provided
        if embeddings is not None and self.contrastive_weight > 0:
            contrastive_loss = self._calculate_contrastive_loss(embeddings, classification_labels)
            loss_dict["contrastive_loss"] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        # Check if we need to track high loss statistics
        if not hasattr(self, '_high_loss_count'):
            self._high_loss_count = 0
            self._total_batches = 0
            self._last_report_batch = 0
            
        self._total_batches += 1
        
        # Check for numerical issues with loss - completely silent replacement
        if not torch.isfinite(total_loss):
            self._high_loss_count += 1
            
            # Replace with a reasonable default loss value that maintains gradient connection
            if classification_logits is not None:
                # Use the classification logits to maintain gradient flow
                dummy_loss = 100.0 * (classification_logits.sum() / classification_logits.numel()).tanh()
                total_loss = dummy_loss.clone()
            else:
                # Fallback with a dummy tensor that requires grad
                dummy = torch.ones(1, device=total_loss.device, requires_grad=True)
                total_loss = 100.0 * dummy.sum()
        
        # Clamp loss to prevent extremely large values that can destabilize training
        # A reasonable upper bound for a total loss is 100 - reduced from 1000
        max_loss = 100.0
        if total_loss > max_loss:
            self._high_loss_count += 1
            
            # Scale the loss down instead of replacing it to maintain gradient flow
            scale_factor = max_loss / total_loss.item()
            total_loss = scale_factor * total_loss
            
        # Print statistics every 100 batches, completely silent otherwise
        if self._total_batches - self._last_report_batch >= 100:
            self._last_report_batch = self._total_batches
            
            # Only print if we're actually having issues
            if self._high_loss_count > 0:
                high_loss_percent = (self._high_loss_count / 100) * 100
                if high_loss_percent > 90:
                    print(f"LOSS STATS: {high_loss_percent:.0f}% of recent batches had high loss. Consider lowering learning rate.")
                
            # Reset counter after reporting
            self._high_loss_count = 0
            
        loss_dict["total_loss"] = total_loss
        return loss_dict