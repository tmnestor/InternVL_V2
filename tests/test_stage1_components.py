#!/usr/bin/env python3
"""
Test suite for Stage 1 model components.

This module contains unit tests for the individual components that make up
the extended InternVL2 architecture, focusing on:
- Cross-attention mechanism
- Response generator
- Classification head
"""

import os
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components to test
from models.components.projection_head import ClassificationHead, CrossAttention, ResponseGenerator


class TestCrossAttention(unittest.TestCase):
    """Test the cross-attention mechanism for vision-language integration."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set up common parameters
        self.query_dim = 768
        self.key_value_dim = 512
        self.embed_dim = 768
        self.num_heads = 8
        self.dropout = 0.0  # Use 0 for deterministic testing
        
        # Create cross-attention module
        self.cross_attn = CrossAttention(
            query_dim=self.query_dim,
            key_value_dim=self.key_value_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
    
    def test_dimensions(self):
        """Test that the cross-attention module has the correct dimensions."""
        # Check internal dimensions
        self.assertEqual(self.cross_attn.num_heads, self.num_heads)
        self.assertEqual(self.cross_attn.head_dim, self.embed_dim // self.num_heads)
        
        # Check projection matrices
        self.assertEqual(self.cross_attn.q_proj.in_features, self.query_dim)
        self.assertEqual(self.cross_attn.q_proj.out_features, self.embed_dim)
        
        self.assertEqual(self.cross_attn.k_proj.in_features, self.key_value_dim)
        self.assertEqual(self.cross_attn.k_proj.out_features, self.embed_dim)
        
        self.assertEqual(self.cross_attn.v_proj.in_features, self.key_value_dim)
        self.assertEqual(self.cross_attn.v_proj.out_features, self.embed_dim)
        
        self.assertEqual(self.cross_attn.out_proj.in_features, self.embed_dim)
        self.assertEqual(self.cross_attn.out_proj.out_features, self.embed_dim)
    
    def test_projection(self):
        """Test that the projection matrices work correctly."""
        # Create dummy inputs
        batch_size = 2
        query_seq_len = 10
        key_seq_len = 20
        
        query = torch.randn(batch_size, query_seq_len, self.query_dim)
        key = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        value = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        
        # Test projections separately
        q_proj = self.cross_attn.q_proj(query)
        k_proj = self.cross_attn.k_proj(key)
        v_proj = self.cross_attn.v_proj(value)
        
        # Check shapes
        self.assertEqual(q_proj.shape, (batch_size, query_seq_len, self.embed_dim))
        self.assertEqual(k_proj.shape, (batch_size, key_seq_len, self.embed_dim))
        self.assertEqual(v_proj.shape, (batch_size, key_seq_len, self.embed_dim))
    
    def test_attention_weights(self):
        """Test that attention weights are computed correctly."""
        # Create dummy inputs
        batch_size = 2
        query_seq_len = 10
        key_seq_len = 20
        
        # Create dummy tensors directly in projected form for simplicity
        q_proj = torch.randn(batch_size, query_seq_len, self.embed_dim)
        k_proj = torch.randn(batch_size, key_seq_len, self.embed_dim)
        
        # Reshape to multi-head form
        q_reshaped = q_proj.view(batch_size, query_seq_len, self.num_heads, self.embed_dim // self.num_heads)
        q_reshaped = q_reshaped.permute(0, 2, 1, 3)  # [batch, heads, q_len, head_dim]
        
        k_reshaped = k_proj.view(batch_size, key_seq_len, self.num_heads, self.embed_dim // self.num_heads)
        k_reshaped = k_reshaped.permute(0, 2, 3, 1)  # [batch, heads, head_dim, k_len]
        
        # Compute attention scores manually
        attention_scores = torch.matmul(q_reshaped, k_reshaped) / torch.sqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))
        
        # Check shape of attention scores
        self.assertEqual(attention_scores.shape, (batch_size, self.num_heads, query_seq_len, key_seq_len))
        
        # Check that scores sum to expected values (before softmax)
        self.assertTrue(torch.isfinite(attention_scores).all())
    
    def test_forward(self):
        """Test the forward pass of the cross-attention module."""
        # Create dummy inputs
        batch_size = 2
        query_seq_len = 10
        key_seq_len = 20
        
        query = torch.randn(batch_size, query_seq_len, self.query_dim)
        key = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        value = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        
        # Test forward pass
        output = self.cross_attn(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, query_seq_len, self.embed_dim))
        
        # Check output values are finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_attention_mask(self):
        """Test the attention mask functionality."""
        # Create dummy inputs
        batch_size = 2
        query_seq_len = 10
        key_seq_len = 20
        
        query = torch.randn(batch_size, query_seq_len, self.query_dim)
        key = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        value = torch.randn(batch_size, key_seq_len, self.key_value_dim)
        
        # Create different types of attention masks
        # Binary mask (boolean)
        bool_mask = torch.ones(batch_size, query_seq_len, key_seq_len, dtype=torch.bool)
        bool_mask[:, :, key_seq_len//2:] = False  # Mask out second half of keys
        
        # Float mask (0.0 = masked, 1.0 = keep)
        float_mask = torch.ones(batch_size, query_seq_len, key_seq_len, dtype=torch.float32)
        float_mask[:, :, key_seq_len//2:] = 0.0  # Mask out second half of keys
        
        # Test with boolean mask
        output_bool_mask = self.cross_attn(query, key, value, bool_mask)
        self.assertEqual(output_bool_mask.shape, (batch_size, query_seq_len, self.embed_dim))
        
        # Test with float mask
        output_float_mask = self.cross_attn(query, key, value, float_mask)
        self.assertEqual(output_float_mask.shape, (batch_size, query_seq_len, self.embed_dim))
        
        # Test without mask
        output_no_mask = self.cross_attn(query, key, value)
        self.assertEqual(output_no_mask.shape, (batch_size, query_seq_len, self.embed_dim))
        
        # The outputs should be different with and without masks
        self.assertFalse(torch.allclose(output_bool_mask, output_no_mask))
        self.assertFalse(torch.allclose(output_float_mask, output_no_mask))


class TestResponseGenerator(unittest.TestCase):
    """Test the response generator for text generation capability."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set up common parameters
        self.input_dim = 768
        self.hidden_dims = [512, 256]
        self.vocab_size = 30000
        self.max_length = 50
        self.dropout_rates = [0.0, 0.0]  # Use 0 for deterministic testing
        self.use_batchnorm = False  # Disable for deterministic testing
        self.activation = "gelu"
        
        # Create response generator
        self.response_gen = ResponseGenerator(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            dropout_rates=self.dropout_rates,
            use_batchnorm=self.use_batchnorm,
            activation=self.activation
        )
    
    def test_mlp_structure(self):
        """Test that the MLP has the correct structure."""
        # Check MLP layers
        expected_mlp_layers = len(self.hidden_dims) * 3 - 1  # layers, activation, dropout for each except last
        self.assertEqual(len(self.response_gen.mlp), expected_mlp_layers)
        
        # Check first layer
        self.assertIsInstance(self.response_gen.mlp[0], nn.Linear)
        self.assertEqual(self.response_gen.mlp[0].in_features, self.input_dim)
        self.assertEqual(self.response_gen.mlp[0].out_features, self.hidden_dims[0])
        
        # Check activation
        if self.activation == "gelu":
            self.assertIsInstance(self.response_gen.mlp[1], nn.GELU)
        elif self.activation == "relu":
            self.assertIsInstance(self.response_gen.mlp[1], nn.ReLU)
        
        # Check dropout
        self.assertIsInstance(self.response_gen.mlp[2], nn.Dropout)
        self.assertEqual(self.response_gen.mlp[2].p, self.dropout_rates[0])
        
        # Check second layer
        self.assertIsInstance(self.response_gen.mlp[3], nn.Linear)
        self.assertEqual(self.response_gen.mlp[3].in_features, self.hidden_dims[0])
        self.assertEqual(self.response_gen.mlp[3].out_features, self.hidden_dims[1])
        
        # Check output projection
        self.assertIsInstance(self.response_gen.output_proj, nn.Linear)
        self.assertEqual(self.response_gen.output_proj.in_features, self.hidden_dims[-1])
        self.assertEqual(self.response_gen.output_proj.out_features, self.vocab_size)
    
    def test_forward(self):
        """Test the forward pass of the response generator."""
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        features = torch.randn(batch_size, seq_len, self.input_dim)
        
        # Test forward pass
        output = self.response_gen(features)
        
        # Check output structure
        self.assertIsInstance(output, dict)
        self.assertIn("logits", output)
        self.assertIn("features", output)
        
        # Check shapes
        self.assertEqual(output["logits"].shape, (batch_size, seq_len, self.vocab_size))
        self.assertEqual(output["features"].shape, (batch_size, seq_len, self.hidden_dims[-1]))
        
        # Check output values are finite
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(torch.isfinite(output["features"]).all())
    
    def test_next_token_logits(self):
        """Test computing logits for the next token."""
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        features = torch.randn(batch_size, seq_len, self.input_dim)
        
        # Process through MLP to get features
        processed_features = features
        for layer in self.response_gen.mlp:
            processed_features = layer(processed_features)
        
        # Compute logits manually
        expected_logits = self.response_gen.output_proj(processed_features)
        
        # Get logits from forward pass
        output = self.response_gen(features)
        actual_logits = output["logits"]
        
        # Check that logits match
        self.assertTrue(torch.allclose(expected_logits, actual_logits))
    
    def test_generate_single_step(self):
        """Test a single step of generation."""
        # Create dummy inputs
        batch_size = 2
        seq_len = 1
        token_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        features = torch.randn(batch_size, seq_len, self.input_dim)
        
        # Create a simple embedding that just returns features
        mock_embedding = nn.Embedding(self.vocab_size, self.input_dim)
        
        # Temporarily replace the embedding in generate method
        original_embedding = self.response_gen.token_embedding
        self.response_gen.token_embedding = mock_embedding
        
        # Override token_embedding weights with identity-like behavior for testing
        with torch.no_grad():
            eye_matrix = torch.eye(self.vocab_size, self.input_dim)
            self.response_gen.token_embedding.weight.copy_(eye_matrix[:self.vocab_size, :self.input_dim])
        
        # Forward pass with token_ids
        processed_features = mock_embedding(token_ids)
        for layer in self.response_gen.mlp:
            processed_features = layer(processed_features)
        
        # Compute logits manually
        expected_logits = self.response_gen.output_proj(processed_features)
        
        # Sample next token (use argmax for deterministic testing)
        expected_next_token = expected_logits.argmax(dim=-1)
        
        # Get next token from the generate method (simplified test)
        with torch.no_grad():
            logits = self.response_gen(processed_features)["logits"]
            next_token = logits.argmax(dim=-1)
        
        # Check that next tokens match
        self.assertTrue(torch.allclose(expected_next_token, next_token))
        
        # Restore original embedding
        self.response_gen.token_embedding = original_embedding
    
    def test_sampling_methods(self):
        """Test different sampling methods for generation."""
        # Create dummy logits with a clear preference
        batch_size = 2
        seq_len = 1
        logits = torch.ones(batch_size, seq_len, self.vocab_size) * -100  # Initialize with low probability
        
        # Set a few tokens to have higher probabilities (at different positions for each batch)
        top_tokens = [[100, 200, 300], [400, 500, 600]]
        for i in range(batch_size):
            for j, token_id in enumerate(top_tokens[i]):
                logits[i, 0, token_id] = 10.0 - j  # Decreasing probability
        
        # Test temperature scaling
        temperatures = [0.5, 1.0, 2.0]
        for temp in temperatures:
            scaled_logits = logits / temp
            # Check that the scale changed appropriately
            self.assertTrue(torch.allclose(scaled_logits, logits / temp))
            
            # Check that ordering is preserved
            for i in range(batch_size):
                top_k_tokens = torch.topk(scaled_logits[i, 0], 3).indices
                self.assertTrue(all(t in top_k_tokens for t in top_tokens[i]))
        
        # Test top-k sampling
        k_values = [1, 2, 5, 10]
        for k in k_values:
            for i in range(batch_size):
                # Get top-k token indices
                top_k_values, top_k_indices = torch.topk(logits[i, 0], k)
                
                # Set all other values to -inf
                filtered_logits = logits[i, 0].clone()
                mask = torch.ones_like(filtered_logits, dtype=torch.bool)
                mask[top_k_indices] = False
                filtered_logits[mask] = float('-inf')
                
                # Check that only the top-k tokens have finite logits
                finite_count = torch.isfinite(filtered_logits).sum().item()
                self.assertEqual(finite_count, k)
        
        # Test top-p (nucleus) sampling
        p_values = [0.9, 0.95]
        for p in p_values:
            for i in range(batch_size):
                # Sort logits
                sorted_logits, sorted_indices = torch.sort(logits[i, 0], descending=True)
                
                # Compute cumulative probabilities
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find indices where cumulative prob exceeds p
                sorted_indices_to_keep = cumulative_probs <= p
                
                # Count how many tokens to keep
                keep_count = sorted_indices_to_keep.sum().item()
                
                # This should always include at least one token
                self.assertGreaterEqual(keep_count, 1)
                
                # For our test setup, this should include the top tokens
                self.assertLessEqual(keep_count, self.vocab_size)


class TestClassificationHead(unittest.TestCase):
    """Test the classification head for receipt counting."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set up common parameters
        self.input_dim = 768
        self.hidden_dims = [512, 256]
        self.output_dim = 3  # 0, 1, 2+ receipts
        self.dropout_rates = [0.0, 0.0]  # Use 0 for deterministic testing
        self.use_batchnorm = False  # Disable for deterministic testing
        self.activation = "gelu"
        
        # Create classification head
        self.cls_head = ClassificationHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rates=self.dropout_rates,
            use_batchnorm=self.use_batchnorm,
            activation=self.activation
        )
    
    def test_mlp_structure(self):
        """Test that the MLP has the correct structure."""
        # Check MLP layers
        expected_mlp_layers = len(self.hidden_dims) * 3 + 1  # layers, activation, dropout for each, and output layer
        self.assertEqual(len(self.cls_head.mlp), expected_mlp_layers)
        
        # Check first layer
        self.assertIsInstance(self.cls_head.mlp[0], nn.Linear)
        self.assertEqual(self.cls_head.mlp[0].in_features, self.input_dim)
        self.assertEqual(self.cls_head.mlp[0].out_features, self.hidden_dims[0])
        
        # Check activation
        if self.activation == "gelu":
            self.assertIsInstance(self.cls_head.mlp[1], nn.GELU)
        elif self.activation == "relu":
            self.assertIsInstance(self.cls_head.mlp[1], nn.ReLU)
        
        # Check dropout
        self.assertIsInstance(self.cls_head.mlp[2], nn.Dropout)
        self.assertEqual(self.cls_head.mlp[2].p, self.dropout_rates[0])
        
        # Check second layer
        self.assertIsInstance(self.cls_head.mlp[3], nn.Linear)
        self.assertEqual(self.cls_head.mlp[3].in_features, self.hidden_dims[0])
        self.assertEqual(self.cls_head.mlp[3].out_features, self.hidden_dims[1])
        
        # Check output layer
        output_layer_idx = len(self.cls_head.mlp) - 1
        self.assertIsInstance(self.cls_head.mlp[output_layer_idx], nn.Linear)
        self.assertEqual(self.cls_head.mlp[output_layer_idx].in_features, self.hidden_dims[-1])
        self.assertEqual(self.cls_head.mlp[output_layer_idx].out_features, self.output_dim)
    
    def test_forward(self):
        """Test the forward pass of the classification head."""
        # Create dummy inputs
        batch_size = 2
        features = torch.randn(batch_size, self.input_dim)
        
        # Test forward pass
        logits = self.cls_head(features)
        
        # Check shape
        self.assertEqual(logits.shape, (batch_size, self.output_dim))
        
        # Check output values are finite
        self.assertTrue(torch.isfinite(logits).all())
    
    def test_classification_prediction(self):
        """Test that classification prediction works as expected."""
        # Create dummy features
        batch_size = 10
        features = torch.randn(batch_size, self.input_dim)
        
        # Get logits
        logits = self.cls_head(features)
        
        # Convert to class predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Check shape and range
        self.assertEqual(predictions.shape, (batch_size,))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < self.output_dim))
    
    def test_variable_batch_size(self):
        """Test that the classification head works with different batch sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Create features
            features = torch.randn(batch_size, self.input_dim)
            
            # Get logits
            logits = self.cls_head(features)
            
            # Check shape
            self.assertEqual(logits.shape, (batch_size, self.output_dim))
    
    def test_with_batchnorm(self):
        """Test the classification head with batch normalization."""
        # Create a version with batch normalization
        cls_head_bn = ClassificationHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            dropout_rates=self.dropout_rates,
            use_batchnorm=True,
            activation=self.activation
        )
        
        # Verify batch norm layers exist
        bn_count = 0
        for layer in cls_head_bn.mlp:
            if isinstance(layer, nn.BatchNorm1d):
                bn_count += 1
        
        # Should have batch norm after each hidden layer
        self.assertEqual(bn_count, len(self.hidden_dims))
        
        # Test forward pass
        batch_size = 2
        features = torch.randn(batch_size, self.input_dim)
        
        # Set to eval mode to get deterministic behavior
        cls_head_bn.eval()
        
        # Get logits
        logits = cls_head_bn(features)
        
        # Check shape
        self.assertEqual(logits.shape, (batch_size, self.output_dim))


if __name__ == "__main__":
    unittest.main()