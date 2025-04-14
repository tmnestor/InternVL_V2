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
        
        # Keep model in float32 for maximum compatibility
        # We'll handle precision dynamically in forward method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Logits tensor of shape [batch_size, output_dim]
        """
        # Ensure input is in float32 for maximum compatibility
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Ensure all MLP layers are in float32
        for module in self.mlp:
            if hasattr(module, 'weight') and module.weight.dtype != torch.float32:
                module.to(torch.float32)
        
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
        # [B, H, Lq, D]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, H, Lk, D]
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, H, Lv, D]
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
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
        self.feature_transformer = nn.Sequential(*layers)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(prev_dim, vocab_size)
        
        # Enable gradient checkpointing to save memory
        self.use_gradient_checkpointing = True
    
    def _feature_transform_fn(self, x):
        """Helper function for gradient checkpointing"""
        return self.feature_transformer(x)
    
    def _lm_head_fn(self, x):
        """Helper function for gradient checkpointing"""
        return self.lm_head(x)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the response generator.
        
        Args:
            x: Input tensor of multimodal features [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary with logits tensor [batch_size, seq_len, vocab_size]
        """
        # Ensure input is in float32 for maximum compatibility
        input_dtype = x.dtype
        if input_dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Ensure feature_transformer modules are in float32
        for module in self.feature_transformer:
            if hasattr(module, 'weight') and module.weight.dtype != torch.float32:
                module.to(torch.float32)
        
        # Ensure lm_head is in float32
        if hasattr(self.lm_head, 'weight') and self.lm_head.weight.dtype != torch.float32:
            self.lm_head.to(torch.float32)
        
        # Process features with gradient checkpointing to reduce memory usage
        if self.use_gradient_checkpointing and self.training and torch.is_grad_enabled() and x.requires_grad:
            # Use gradient checkpointing to save memory
            transformed = torch.utils.checkpoint.checkpoint(
                self._feature_transform_fn, 
                x,
                use_reentrant=False  # Per CLAUDE.md instructions to avoid warnings in PyTorch 2.0+
            )
            
            # Low-rank adaptation for large vocabulary output
            batch_size, seq_len, hidden_dim = transformed.shape
            
            # Process in smaller chunks if sequence is too long
            if seq_len > 64:
                # Split processing into chunks to reduce memory usage
                logits_list = []
                chunk_size = 32  # Process 32 tokens at a time
                
                for i in range(0, seq_len, chunk_size):
                    end_idx = min(i + chunk_size, seq_len)
                    chunk = transformed[:, i:end_idx, :]
                    
                    # Apply LM head with gradient checkpointing
                    if torch.is_grad_enabled() and chunk.requires_grad:
                        chunk_logits = torch.utils.checkpoint.checkpoint(
                            self._lm_head_fn,
                            chunk,
                            use_reentrant=False
                        )
                    else:
                        # Fallback if not tracking gradients
                        chunk_logits = self.lm_head(chunk)
                    logits_list.append(chunk_logits)
                
                # Concatenate chunks back together
                logits = torch.cat(logits_list, dim=1)
            else:
                # If sequence is short enough, process all at once with checkpointing
                if torch.is_grad_enabled() and transformed.requires_grad:
                    logits = torch.utils.checkpoint.checkpoint(
                        self._lm_head_fn,
                        transformed,
                        use_reentrant=False
                    )
                else:
                    # Fallback if not tracking gradients
                    logits = self.lm_head(transformed)
        else:
            # Standard forward pass without checkpointing for inference
            transformed = self.feature_transformer(x)  # [B, L, H]
            logits = self.lm_head(transformed)  # [B, L, V]
        
        return {
            "logits": logits,
            "features": transformed,
        }
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        multimodal_context: torch.Tensor,
        temperature: float = 0.7,  # Lower temperature for more focused generation
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> List[List[int]]:
        """
        Generate text responses autoregressively.
        
        Args:
            start_tokens: Initial token IDs [batch_size, prefix_len]
            multimodal_context: Multimodal features to condition on [batch_size, context_len, dim]
            temperature: Sampling temperature (1.0 = no change, <1.0 = sharper, >1.0 = more random)
            top_k: If set, only sample from the top k most likely tokens
            top_p: If set, sample from the smallest set of tokens whose cumulative probability exceeds p
            
        Returns:
            List of generated token sequences, each as a list of token IDs.
        """
        batch_size = start_tokens.shape[0]
        device = start_tokens.device
        
        # Initialize generation with start tokens
        generated = start_tokens.clone()
        seq_len = generated.shape[1]
        
        # Define EOS token IDs to stop generation (common values in many tokenizers)
        eos_token_ids = {0, 1, 2, 50256}  # Common EOS token IDs in many tokenizers
        
        # Convert all tensors to float32 for consistency
        multimodal_context = multimodal_context.to(torch.float32)
        
        # Pre-process multimodal context once
        context_features = self.feature_transformer(multimodal_context.mean(dim=1))
        
        # Generate tokens autoregressively
        for i in range(seq_len, min(self.max_length, seq_len + 50)):  # Limit to 50 new tokens
            # Generate logits using the context features
            # This is a more focused approach than the previous implementation
            logits = self.lm_head(context_features)  # [B, V]
            
            # Apply temperature scaling
            logits = logits / max(0.1, temperature)  # Prevent division by very small values
            
            # Apply top-k sampling
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))  # Cannot sample more than vocabulary size
                topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, topk_indices, topk_values)
                logits = logits_filtered
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least the top token
                sorted_indices_to_remove[..., 0] = 0
                # Shift to remove tokens below threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                
                # Scatter sorted indices to original indices
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices_to_remove[batch_idx].scatter(
                        0, sorted_indices[batch_idx], sorted_indices_to_remove[batch_idx]
                    )
                    logits[batch_idx][indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Clamp to valid token range to prevent errors
            next_token = torch.clamp(next_token, min=0, max=self.vocab_size-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if we've hit EOS tokens for all sequences
            if all((generated[:, -1] == t).any().item() for t in eos_token_ids):
                break
        
        # Convert tensor to list of token lists
        generated_lists = []
        for i in range(generated.shape[0]):
            # Only keep tokens after the initial prompt (start_tokens)
            valid_tokens = torch.clamp(generated[i, seq_len:], min=0, max=self.vocab_size-1)
            generated_lists.append(valid_tokens.tolist())
        
        return generated_lists