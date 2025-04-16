"""
Focal loss implementation for training classification models.

This module provides an implementation of focal loss to address class imbalance
and focus the model on hard examples during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    
    This loss function helps focus training on hard examples and address class imbalance.
    Introduced in paper: "Focal Loss for Dense Object Detection" - Lin et al. 2017.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the true class
    - alpha_t is a weighting factor for class imbalance
    - gamma is a focusing parameter (gamma > 0)
    """
    
    def __init__(
        self,
        alpha: torch.Tensor = None,  # Class weights
        gamma: float = 2.0,          # Focusing parameter
        reduction: str = 'mean',     # Reduction method
        label_smoothing: float = 0.0 # Label smoothing factor
    ):
        """
        Initialize focal loss with parameters.
        
        Args:
            alpha: Weighting factor for classes, helps with class imbalance
            gamma: Focusing parameter that adjusts rate at which easy examples are down-weighted
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor for preventing overconfidence
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for focal loss computation.
        
        Args:
            inputs: Model predictions (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Loss value
        """
        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Get probs from log_probs
        probs = torch.exp(log_probs)
        
        # Gather the probability for the target class
        target_probs = probs.gather(1, targets.unsqueeze(1))
        
        # Calculate focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Make sure alpha is on the same device as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_weight = self.alpha.gather(0, targets)
            focal_weight = alpha_weight * focal_weight
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            # Create smoothed targets
            num_classes = inputs.size(-1)
            smoothed_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1
            )
            smoothed_targets = smoothed_targets * (1 - self.label_smoothing) + \
                              self.label_smoothing / num_classes
            
            # Calculate focal loss with smoothed targets
            loss = -focal_weight.squeeze() * torch.sum(
                smoothed_targets * log_probs, dim=-1
            )
        else:
            # Calculate focal loss
            loss = -focal_weight.squeeze() * log_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class FocalLossWithLogits(nn.Module):
    """
    Focal loss with raw logits input (no softmax).
    
    Useful when the model doesn't apply softmax internally.
    """
    
    def __init__(
        self,
        alpha: torch.Tensor = None,  # Class weights
        gamma: float = 2.0,          # Focusing parameter
        reduction: str = 'mean',     # Reduction method
        label_smoothing: float = 0.0 # Label smoothing factor
    ):
        """
        Initialize focal loss with parameters.
        
        Args:
            alpha: Weighting factor for classes, helps with class imbalance
            gamma: Focusing parameter that adjusts rate at which easy examples are down-weighted
            reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor for preventing overconfidence
        """
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for focal loss computation.
        
        Args:
            logits: Raw model predictions (before softmax), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Loss value
        """
        # Get log softmax outputs
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            # Create one-hot encoding of targets
            num_classes = logits.size(-1)
            with torch.no_grad():
                targets_one_hot = torch.zeros_like(logits)
                targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                # Apply label smoothing
                targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                                 self.label_smoothing / num_classes
            
            # Calculate the loss with smoothed targets
            loss = -targets_one_hot * log_probs
            
            # Get the probability for the target class
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Calculate the focal weight
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply class weights if provided
            if self.alpha is not None:
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            
            # Apply focal weight
            loss = focal_weight.unsqueeze(1) * loss
        else:
            # Get the probability for the target class
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Calculate the focal weight
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply class weights if provided
            if self.alpha is not None:
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            
            # Calculate the focal loss
            loss = -focal_weight * log_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss.sum(dim=-1) if self.label_smoothing > 0 else loss
        elif self.reduction == 'mean':
            return loss.mean() if self.label_smoothing == 0 else loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum() if self.label_smoothing == 0 else loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


def get_class_weights(
    class_counts: dict,
    num_classes: int = 5,
    weight_strategy: str = 'inverse_frequency',
    beta: float = 0.999
) -> torch.Tensor:
    """
    Calculate class weights based on various strategies.
    
    Args:
        class_counts: Dictionary mapping class indices to their counts
        num_classes: Total number of classes
        weight_strategy: Strategy to use ('inverse_frequency', 'effective_samples', or 'manual')
        beta: Parameter for effective number of samples strategy
        
    Returns:
        Tensor of class weights
    """
    # Ensure all classes are represented
    counts = torch.ones(num_classes)
    for cls_idx, count in class_counts.items():
        counts[cls_idx] = count
    
    if weight_strategy == 'inverse_frequency':
        # Inverse frequency weighting
        weights = torch.zeros(num_classes)
        total_samples = counts.sum()
        for i in range(num_classes):
            weights[i] = total_samples / (counts[i] * num_classes)
        
    elif weight_strategy == 'effective_samples':
        # Effective number of samples weighting (from paper "Class-Balanced Loss")
        # Formula: (1-beta)/(1-beta^n)
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            effective_num = 1.0 - beta ** counts[i]
            weights[i] = (1.0 - beta) / effective_num
        
    elif weight_strategy == 'manual':
        # Manual weights based on our observations
        weights = torch.tensor([2.0, 1.0, 0.5, 0.2, 2.0])
        
    else:
        raise ValueError(f"Unsupported weight strategy: {weight_strategy}")
    
    # Normalize weights to have a mean of 1
    weights = weights * (num_classes / weights.sum())
    
    return weights