"""
Projection head components for classification tasks and multimodal processing.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Flexible classification head for vision models.
    
    Supports multiple hidden layers, dropout, batch normalization,
    and different activation functions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rates: Optional[List[float]] = None,
        use_batchnorm: bool = True,
        activation: str = "gelu",
    ):
        """
        Initialize a custom classification head.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (number of classes)
            dropout_rates: Optional list of dropout rates for each layer
            use_batchnorm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        # Set up activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu" or activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Configure dropout rates
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_dims)
        elif len(dropout_rates) != len(hidden_dims):
            raise ValueError("Number of dropout rates must match number of hidden layers")
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates, strict=False)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(act_fn)
            
            # Dropout
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.mlp = nn.Sequential(*layers)
        
        # Convert model to bfloat16 if using GPU with bfloat16 support
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Get majority dtype of tensors in the model
            self.mlp = self.mlp.to(torch.bfloat16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Logits tensor of shape [batch_size, output_dim]
        """
        # Check for dtype compatibility with first linear layer
        if len(self.mlp) > 0 and hasattr(self.mlp[0], 'weight') and x.dtype != self.mlp[0].weight.dtype:
            # Ensure input has same dtype as weights without logging
            x = x.to(self.mlp[0].weight.dtype)
        
        return self.mlp(x)


class CrossAttention(nn.Module):
    """
    Cross-attention module for fusing vision and language features.
    
    This module implements multi-head attention where queries come from 
    one modality (text) and keys/values come from another (vision).
    """
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention module.
        
        Args:
            query_dim: Dimension of query features (typically text)
            key_value_dim: Dimension of key/value features (typically vision)
            embed_dim: Output dimension after attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(key_value_dim, embed_dim)
        self.v_proj = nn.Linear(key_value_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, query_dim]
            key: Key tensor [batch_size, seq_len_k, key_value_dim]
            value: Value tensor [batch_size, seq_len_v, key_value_dim]
            attention_mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Output tensor after cross-attention [batch_size, seq_len_q, embed_dim]
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Project inputs
        q = self.q_proj(query)  # [B, Lq, E]
        k = self.k_proj(key)    # [B, Lk, E]
        v = self.v_proj(value)  # [B, Lv, E]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lq, D]
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lk, D]
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, Lv, D]
        
        # Compute scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Lq, Lk]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # expand mask for multi-head: [B, 1, Lq, Lk]
            expanded_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(expanded_mask == 0, -1e10)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v)  # [B, H, Lq, D]
        
        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [B, Lq, H, D]
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)  # [B, Lq, E]
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class ResponseGenerator(nn.Module):
    """
    Response generator head for multimodal tasks.
    
    Takes in cross-modal features and generates text responses.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        vocab_size: int,
        max_length: int = 128,
        dropout_rates: Optional[List[float]] = None,
        use_batchnorm: bool = True,
        activation: str = "gelu",
    ):
        """
        Initialize response generator.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            vocab_size: Size of vocabulary for text generation
            max_length: Maximum sequence length for generation
            dropout_rates: Optional list of dropout rates for each layer
            use_batchnorm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'gelu', etc.)
        """
        super().__init__()
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Set up activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu" or activation == "swish":
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Configure dropout rates
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_dims)
        elif len(dropout_rates) != len(hidden_dims):
            raise ValueError("Number of dropout rates must match number of hidden layers")
        
        # Build feature transformer
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates, strict=False)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Optional batch normalization
            if use_batchnorm:
                layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm is more typical for NLP tasks
            
            # Activation
            layers.append(act_fn)
            
            # Dropout
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            
            prev_dim = hidden_dim
        
        # Feature transformer (processes multimodal features)
        self.mlp = nn.Sequential(*layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(prev_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the response generator.
        
        Args:
            x: Input tensor of multimodal features [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary with logits tensor [batch_size, seq_len, vocab_size]
        """
        # Process features
        transformed = self.mlp(x)  # [B, L, H]
        
        # Generate logits for each token position
        logits = self.output_proj(transformed)  # [B, L, V]
        
        return {
            "logits": logits,
            "features": transformed,
        }
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        multimodal_context: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text responses autoregressively.
        
        Args:
            start_tokens: Initial token IDs [batch_size, prefix_len]
            multimodal_context: Multimodal features to condition on [batch_size, context_len, dim]
            temperature: Sampling temperature (1.0 = no change, <1.0 = sharper, >1.0 = more random)
            top_k: If set, only sample from the top k most likely tokens
            top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
            
        Returns:
            Generated token sequence [batch_size, max_length]
        """
        # This is a simplified autoregressive decoding implementation
        # In a real implementation, you would:
        # 1. Use cached key-value pairs for efficiency
        # 2. Support beam search
        # 3. Handle stopping criteria more elegantly
        
        batch_size = start_tokens.shape[0]
        device = start_tokens.device
        
        # Initialize generation with start tokens
        generated = start_tokens.clone()
        seq_len = generated.shape[1]
        
        # Generate tokens autoregressively
        for i in range(seq_len, self.max_length):
            # Get features for current sequence
            inputs = generated  # [B, i]
            
            # TODO: This is a simplified version; in a full implementation
            # you would have a proper decoder that attends to the multimodal_context
            
            # For now, just concatenate the last token embedding with the context
            # and use the feature transformer to predict the next token
            
            # Get features for the last position
            if i > seq_len:
                # Use just the last generated token
                last_token_pos = -1
                transformed = self.mlp(
                    torch.cat([multimodal_context[:, 0, :], inputs[:, last_token_pos].unsqueeze(1)], dim=1)
                )  # [B, H]
            else:
                # Initial case, use the multimodal context directly
                transformed = self.mlp(multimodal_context[:, 0, :])  # [B, H]
            
            # Generate logits
            logits = self.output_proj(transformed)  # [B, V]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted indices to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated