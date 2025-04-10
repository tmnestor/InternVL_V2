#!/usr/bin/env python3
"""
Test suite for Stage 1: Model Architecture Extension.

This module contains unit tests for verifying the InternVL2 model architecture
extensions for multimodal capabilities, including:
- Vision-language model architecture
- Cross-attention mechanisms
- Response generation capability
- Proper handling of both vision and language inputs
"""

import os
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components to test
from models.internvl2 import InternVL2MultimodalModel
from models.components.projection_head import CrossAttention, ResponseGenerator


class TestModelArchitecture(unittest.TestCase):
    """Test the core architecture of the InternVL2 multimodal model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Mock config for testing
        cls.config = {
            "model": {
                "pretrained_path": "mock_path",  # Will be mocked in tests
                "num_classes": 3,  # 0, 1, 2+ receipts
                "classifier": {
                    "hidden_dims": [512, 256],
                    "dropout_rates": [0.1, 0.1],
                    "batch_norm": True,
                    "activation": "gelu"
                }
            },
            "training": {
                "three_stage": {
                    "enabled": True,
                    "stage2": {
                        "start_epoch": 5,
                        "lr_multiplier": 0.1
                    },
                    "stage3": {
                        "start_epoch": 10,
                        "lr_multiplier": 0.01
                    }
                }
            }
        }
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Use CPU for tests
        cls.device = torch.device("cpu")
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Skip actual model loading for tests - we'll use mocks
        self.original_automodel_from_pretrained = None
        self.original_autotokenizer_from_pretrained = None
        
        # Create temporary directory for mock model if it doesn't exist
        self.mock_model_dir = Path("tests/mock_model")
        self.mock_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with mock path
        self.test_config = self.config.copy()
        self.test_config["model"]["pretrained_path"] = str(self.mock_model_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore original methods if they were patched
        if self.original_automodel_from_pretrained:
            import transformers
            transformers.AutoModel.from_pretrained = self.original_automodel_from_pretrained
            
        if self.original_autotokenizer_from_pretrained:
            import transformers
            transformers.AutoTokenizer.from_pretrained = self.original_autotokenizer_from_pretrained

    def test_cross_attention_initialization(self):
        """Test the initialization of the cross-attention mechanism."""
        # Create cross-attention with reasonable dimensions
        query_dim = 768
        key_value_dim = 768
        embed_dim = 768
        num_heads = 8
        dropout = 0.1
        
        cross_attn = CrossAttention(
            query_dim=query_dim,
            key_value_dim=key_value_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Check structure
        self.assertIsInstance(cross_attn, nn.Module)
        self.assertEqual(cross_attn.num_heads, num_heads)
        self.assertEqual(cross_attn.head_dim, embed_dim // num_heads)
        
        # Check that projection matrices have correct dimensions
        self.assertEqual(cross_attn.q_proj.in_features, query_dim)
        self.assertEqual(cross_attn.q_proj.out_features, embed_dim)
        
        self.assertEqual(cross_attn.k_proj.in_features, key_value_dim)
        self.assertEqual(cross_attn.k_proj.out_features, embed_dim)
        
        self.assertEqual(cross_attn.v_proj.in_features, key_value_dim)
        self.assertEqual(cross_attn.v_proj.out_features, embed_dim)
        
        self.assertEqual(cross_attn.out_proj.in_features, embed_dim)
        self.assertEqual(cross_attn.out_proj.out_features, embed_dim)
    
    def test_cross_attention_forward(self):
        """Test the forward pass of the cross-attention mechanism."""
        # Set dimensions
        batch_size = 2
        query_seq_len = 10
        key_seq_len = 20
        query_dim = 768
        key_value_dim = 512
        embed_dim = 768
        num_heads = 8
        
        # Create inputs
        query = torch.randn(batch_size, query_seq_len, query_dim)
        key = torch.randn(batch_size, key_seq_len, key_value_dim)
        value = torch.randn(batch_size, key_seq_len, key_value_dim)
        
        # Create attention mask (optional)
        attn_mask = torch.ones(batch_size, query_seq_len, key_seq_len, dtype=torch.bool)
        
        # Create cross-attention
        cross_attn = CrossAttention(
            query_dim=query_dim,
            key_value_dim=key_value_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0  # Use 0 for deterministic testing
        )
        
        # Test forward pass without mask
        output = cross_attn(query, key, value)
        self.assertEqual(output.shape, (batch_size, query_seq_len, embed_dim))
        
        # Test forward pass with mask
        output_masked = cross_attn(query, key, value, attn_mask)
        self.assertEqual(output_masked.shape, (batch_size, query_seq_len, embed_dim))
        
        # Outputs should be different with and without mask
        self.assertFalse(torch.allclose(output, output_masked))
    
    def test_response_generator_initialization(self):
        """Test the initialization of the response generator."""
        # Set parameters
        input_dim = 768
        hidden_dims = [512, 256]
        vocab_size = 30000
        max_length = 50
        dropout_rates = [0.1, 0.1]
        use_batchnorm = True
        activation = "gelu"
        
        # Create response generator
        response_gen = ResponseGenerator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            vocab_size=vocab_size,
            max_length=max_length,
            dropout_rates=dropout_rates,
            use_batchnorm=use_batchnorm,
            activation=activation
        )
        
        # Check structure
        self.assertIsInstance(response_gen, nn.Module)
        self.assertEqual(response_gen.vocab_size, vocab_size)
        self.assertEqual(response_gen.max_length, max_length)
        
        # Check MLP structure
        self.assertEqual(len(response_gen.mlp), len(hidden_dims) * 3 - 1)  # layers, activation, dropout for each except last
        
        # Check output projection
        self.assertEqual(response_gen.output_proj.in_features, hidden_dims[-1])
        self.assertEqual(response_gen.output_proj.out_features, vocab_size)
    
    def test_response_generator_forward(self):
        """Test the forward pass of the response generator."""
        # Set parameters
        batch_size = 2
        seq_len = 10
        input_dim = 768
        hidden_dims = [512, 256]
        vocab_size = 30000
        
        # Create inputs
        features = torch.randn(batch_size, seq_len, input_dim)
        
        # Create response generator
        response_gen = ResponseGenerator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            vocab_size=vocab_size,
            max_length=50,
            dropout_rates=[0.0, 0.0],  # Use 0 for deterministic testing
            use_batchnorm=False,  # Disable for deterministic testing
            activation="gelu"
        )
        
        # Test forward pass
        output = response_gen(features)
        
        # Check output structure
        self.assertIsInstance(output, dict)
        self.assertIn("logits", output)
        self.assertIn("features", output)
        
        # Check shapes
        self.assertEqual(output["logits"].shape, (batch_size, seq_len, vocab_size))
        self.assertEqual(output["features"].shape, (batch_size, seq_len, hidden_dims[-1]))
    
    def test_response_generator_generate(self):
        """Test the text generation capability of the response generator."""
        # Set parameters
        batch_size = 2
        seq_len = 1  # Starting with a single token
        input_dim = 768
        context_len = 10
        hidden_dims = [512, 256]
        vocab_size = 30000
        
        # Create inputs (start tokens and context)
        start_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        context = torch.randn(batch_size, context_len, input_dim)
        
        # Create response generator
        response_gen = ResponseGenerator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            vocab_size=vocab_size,
            max_length=5,  # Short for testing
            dropout_rates=[0.0, 0.0],
            use_batchnorm=False,
            activation="gelu"
        )
        
        # Test generation
        generated_ids = response_gen.generate(
            start_tokens=start_tokens,
            multimodal_context=context,
            temperature=1.0,
            top_k=50,
            top_p=1.0
        )
        
        # Check output
        self.assertIsInstance(generated_ids, list)
        self.assertEqual(len(generated_ids), batch_size)
        
        # Each sequence should have tokens and be longer than the input
        for seq in generated_ids:
            self.assertIsInstance(seq, list)
            self.assertGreaterEqual(len(seq), seq_len)
            self.assertLessEqual(len(seq), response_gen.max_length)
    
    def test_multimodal_model_structure(self):
        """Test that the multimodal model has the correct structure and components."""
        # Mock the required modules to avoid actual model loading
        import transformers
        from unittest.mock import MagicMock
        
        # Save original methods
        self.original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
        self.original_autotokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.vision_model = MagicMock()
        mock_model.vision_model.config.hidden_size = 768
        mock_model.language_model = MagicMock()
        mock_model.language_model.config.hidden_size = 768
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 30000
        
        # Replace with mocks
        transformers.AutoModel.from_pretrained = MagicMock(return_value=mock_model)
        transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
        
        # Create multimodal model
        model = InternVL2MultimodalModel(
            config=self.test_config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=False
        )
        
        # Check main components exist
        self.assertTrue(hasattr(model, "vision_encoder"))
        self.assertTrue(hasattr(model, "language_model"))
        self.assertTrue(hasattr(model, "cross_attention"))
        self.assertTrue(hasattr(model, "classification_head"))
        self.assertTrue(hasattr(model, "response_generator"))
        
        # Check that the cross-attention is correct
        self.assertIsInstance(model.cross_attention, CrossAttention)
        
        # Check that the response generator is correct
        self.assertIsInstance(model.response_generator, ResponseGenerator)
    
    def test_multimodal_model_forward_vision_only(self):
        """Test forward pass with only vision input (backward compatibility)."""
        # Mock the required modules
        import transformers
        from unittest.mock import MagicMock
        
        # Save original methods
        self.original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
        self.original_autotokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
        
        # Define vision encoder output
        vision_hidden_size = 768
        batch_size = 2
        seq_len = 197  # Typical for ViT with 224x224 input
        
        # Create mock vision outputs
        class MockVisionOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(batch_size, seq_len, vision_hidden_size)
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.vision_model = MagicMock()
        mock_model.vision_model.config.hidden_size = vision_hidden_size
        mock_model.vision_model.return_value = MockVisionOutput()
        
        mock_model.language_model = MagicMock()
        mock_model.language_model.config.hidden_size = 768
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 30000
        
        # Replace with mocks
        transformers.AutoModel.from_pretrained = MagicMock(return_value=mock_model)
        transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
        
        # Create multimodal model
        model = InternVL2MultimodalModel(
            config=self.test_config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Create a mock for the vision encoder
        model.vision_encoder = mock_model.vision_model
        
        # Create input
        pixel_values = torch.randn(batch_size, 3, 224, 224)  # [B, C, H, W]
        
        # Test forward pass with vision only
        output = model(pixel_values=pixel_values)
        
        # Check outputs
        self.assertIsInstance(output, dict)
        self.assertIn("logits", output)
        self.assertIn("embeddings", output)
        
        # Check shapes
        self.assertEqual(output["logits"].shape, (batch_size, self.test_config["model"]["num_classes"]))
        self.assertEqual(output["embeddings"].shape, (batch_size, vision_hidden_size))
    
    def test_multimodal_model_forward_with_text(self):
        """Test forward pass with both vision and text inputs."""
        # Mock the required modules
        import transformers
        from unittest.mock import MagicMock
        
        # Save original methods
        self.original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
        self.original_autotokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
        
        # Define sizes
        vision_hidden_size = 768
        language_hidden_size = 768
        batch_size = 2
        vision_seq_len = 197  # Typical for ViT
        text_seq_len = 20
        
        # Create mock outputs
        class MockVisionOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(batch_size, vision_seq_len, vision_hidden_size)
        
        class MockLanguageOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(batch_size, text_seq_len, language_hidden_size)
                self.hidden_states = [torch.randn(batch_size, text_seq_len, language_hidden_size)]
        
        # Create mock model components
        mock_vision_encoder = MagicMock()
        mock_vision_encoder.config.hidden_size = vision_hidden_size
        mock_vision_encoder.return_value = MockVisionOutput()
        
        mock_language_model = MagicMock()
        mock_language_model.config.hidden_size = language_hidden_size
        mock_language_model.return_value = MockLanguageOutput()
        
        mock_model = MagicMock()
        mock_model.vision_model = mock_vision_encoder
        mock_model.language_model = mock_language_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 30000
        
        # Replace with mocks
        transformers.AutoModel.from_pretrained = MagicMock(return_value=mock_model)
        transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
        
        # Create multimodal model
        model = InternVL2MultimodalModel(
            config=self.test_config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Replace model components with mocks
        model.vision_encoder = mock_vision_encoder
        model.language_model = mock_language_model
        
        # Create inputs
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        text_input_ids = torch.randint(0, 30000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        
        # Test forward pass with both vision and text
        output = model(
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask
        )
        
        # Check outputs
        self.assertIsInstance(output, dict)
        self.assertIn("logits", output)
        self.assertIn("embeddings", output)
        self.assertIn("multimodal_embeddings", output)
        self.assertIn("response_logits", output)
        self.assertIn("response_features", output)
        
        # Check shapes
        self.assertEqual(output["logits"].shape, (batch_size, self.test_config["model"]["num_classes"]))
        self.assertEqual(output["embeddings"].shape, (batch_size, vision_hidden_size))
        self.assertEqual(output["multimodal_embeddings"].shape, (batch_size, text_seq_len, language_hidden_size))
        self.assertEqual(output["response_logits"].shape, (batch_size, text_seq_len, 30000))
    
    def test_prepare_inputs(self):
        """Test the input preparation for the model."""
        # Mock the required modules
        import transformers
        from unittest.mock import MagicMock
        
        # Save original methods
        self.original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
        self.original_autotokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.vision_model = MagicMock()
        mock_model.vision_model.config.hidden_size = 768
        mock_model.language_model = MagicMock()
        mock_model.language_model.config.hidden_size = 768
        
        # Create a tokenizer that returns predictable outputs
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 30000
            
            def __call__(self, text_prompts, padding, truncation, max_length, return_tensors):
                # Create predictable tokenizer outputs based on inputs
                batch_size = len(text_prompts)
                seq_len = max_length
                
                # Create simple mock tensors
                input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                
                # Create a container that mimics the tokenizer output
                class TokenizerOutput:
                    def __init__(self, input_ids, attention_mask):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                
                return TokenizerOutput(input_ids, attention_mask)
        
        mock_tokenizer = MockTokenizer()
        
        # Replace with mocks
        transformers.AutoModel.from_pretrained = MagicMock(return_value=mock_model)
        transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
        
        # Create multimodal model
        model = InternVL2MultimodalModel(
            config=self.test_config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Set the tokenizer
        model.tokenizer = mock_tokenizer
        
        # Create inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_prompts = ["How many receipts are in this image?", "What is the total value?"]
        
        # Test prepare_inputs method
        inputs = model.prepare_inputs(images, text_prompts)
        
        # Check outputs
        self.assertIsInstance(inputs, dict)
        self.assertIn("pixel_values", inputs)
        self.assertIn("text_input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        
        # Check shapes
        self.assertEqual(inputs["pixel_values"].shape, (batch_size, 3, 224, 224))
        self.assertEqual(inputs["text_input_ids"].shape, (batch_size, 128))  # 128 is default max_length
        self.assertEqual(inputs["attention_mask"].shape, (batch_size, 128))
        
        # Check device consistency
        self.assertEqual(inputs["pixel_values"].device, images.device)
        self.assertEqual(inputs["text_input_ids"].device, images.device)
        self.assertEqual(inputs["attention_mask"].device, images.device)
    
    def test_generate_response(self):
        """Test the response generation functionality."""
        # Mock the required modules
        import transformers
        from unittest.mock import MagicMock
        
        # Save original methods
        self.original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
        self.original_autotokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
        
        # Define sizes
        vision_hidden_size = 768
        language_hidden_size = 768
        batch_size = 2
        vision_seq_len = 197
        text_seq_len = 20
        vocab_size = 30000
        
        # Create mock tokenizer that decodes predictably
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = vocab_size
            
            def decode(self, ids, skip_special_tokens=True):
                # Just return a fixed string for testing
                return "This is a mock response."
        
        # Create mock model with forward method that returns the expected structure
        class MockMultimodalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokenizer = MockTokenizer()
                
                # Create a simple response generator for testing
                self.response_generator = MagicMock()
                self.response_generator.generate.return_value = [
                    [101, 102, 103, 104, 105],  # Mock token IDs
                    [201, 202, 203, 204, 205]
                ]
            
            def forward(self, pixel_values, text_input_ids, attention_mask=None):
                # Return a structure matching the expected output of the forward method
                batch_size = pixel_values.shape[0]
                text_seq_len = text_input_ids.shape[1]
                
                return {
                    "logits": torch.randn(batch_size, 3),  # 3 classes for receipt counting
                    "embeddings": torch.randn(batch_size, vision_hidden_size),
                    "multimodal_embeddings": torch.randn(batch_size, text_seq_len, language_hidden_size),
                    "response_logits": torch.randn(batch_size, text_seq_len, vocab_size),
                    "response_features": torch.randn(batch_size, text_seq_len, language_hidden_size // 2)
                }
        
        # Create the model
        model = MockMultimodalModel()
        
        # Create inputs
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        text_input_ids = torch.randint(0, vocab_size, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        
        # Test generate_response method
        generated_ids, decoded_texts = model.generate_response(
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            max_length=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        
        # Check outputs
        self.assertIsInstance(generated_ids, list)
        self.assertIsInstance(decoded_texts, list)
        self.assertEqual(len(generated_ids), batch_size)
        self.assertEqual(len(decoded_texts), batch_size)
        
        # Check content
        for text in decoded_texts:
            self.assertEqual(text, "This is a mock response.")


if __name__ == "__main__":
    unittest.main()