#!/usr/bin/env python3
"""
Integration tests for Stage 1 model architecture.

This module tests the end-to-end functionality of the multimodal model architecture,
focusing on the integration between vision and language components.
"""

import os
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock transformers imports to avoid actual model loading
sys.modules['transformers'] = MagicMock()
import transformers

# Import model after mocking
from models.vision_language.internvl2 import InternVL2MultimodalModel
from models.components.projection_head import CrossAttention, ResponseGenerator


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the multimodal model architecture."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Mock config for testing
        cls.config = {
            "model": {
                "pretrained_path": "mock_path",
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
        
        # Create vision hidden state dimensions
        cls.vision_hidden_size = 768
        cls.vision_seq_len = 197  # ViT sequence length
        
        # Create language hidden state dimensions
        cls.language_hidden_size = 768
        cls.language_seq_len = 20
        
        # Setup model dimensions
        cls.batch_size = 2
        cls.vocab_size = 30000
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create mock vision and language outputs
        class MockVisionOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(
                    self.batch_size, self.vision_seq_len, self.vision_hidden_size
                )
                self.pooler_output = torch.randn(self.batch_size, self.vision_hidden_size)
        
        class MockLanguageOutput:
            def __init__(self):
                self.last_hidden_state = torch.randn(
                    self.batch_size, self.language_seq_len, self.language_hidden_size
                )
                self.hidden_states = [
                    torch.randn(self.batch_size, self.language_seq_len, self.language_hidden_size)
                ]
        
        # Create mock model components
        self.mock_vision_encoder = MagicMock()
        self.mock_vision_encoder.config = MagicMock()
        self.mock_vision_encoder.config.hidden_size = self.vision_hidden_size
        self.mock_vision_encoder.return_value = MockVisionOutput()
        
        self.mock_language_model = MagicMock()
        self.mock_language_model.config = MagicMock()
        self.mock_language_model.config.hidden_size = self.language_hidden_size
        self.mock_language_model.return_value = MockLanguageOutput()
        
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = self.vocab_size
        
        # Mock the transformers module
        self.patcher = patch.multiple(
            'transformers',
            AutoModel=MagicMock(),
            AutoTokenizer=MagicMock()
        )
        
        # Start patches
        self.patcher.start()
        
        # Configure the mock objects
        transformers.AutoModel.from_pretrained.return_value = MagicMock(
            vision_model=self.mock_vision_encoder,
            language_model=self.mock_language_model
        )
        transformers.AutoTokenizer.from_pretrained.return_value = self.mock_tokenizer
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.patcher.stop()
    
    def test_end_to_end_vision_only(self):
        """Test the end-to-end vision-only path for backward compatibility."""
        # Create a model instance
        model = InternVL2MultimodalModel(
            config=self.config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Replace components with our mocks
        model.vision_encoder = self.mock_vision_encoder
        model.language_model = self.mock_language_model
        model.tokenizer = self.mock_tokenizer
        
        # Create a dummy image batch
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Set the model to eval mode
        model.eval()
        
        # Run a forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        # Check that the output has the expected structure
        self.assertIn("logits", outputs)
        self.assertIn("embeddings", outputs)
        
        # Check that output shapes are correct
        self.assertEqual(outputs["logits"].shape, (batch_size, self.config["model"]["num_classes"]))
        self.assertEqual(outputs["embeddings"].shape, (batch_size, self.vision_hidden_size))
        
        # Vision encoder should be called exactly once
        self.mock_vision_encoder.assert_called_once()
        
        # Language model should not be called
        self.mock_language_model.assert_not_called()
    
    def test_end_to_end_multimodal(self):
        """Test the end-to-end multimodal path with both vision and language inputs."""
        # Create a model instance
        model = InternVL2MultimodalModel(
            config=self.config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Replace components with our mocks
        model.vision_encoder = self.mock_vision_encoder
        model.language_model = self.mock_language_model
        model.tokenizer = self.mock_tokenizer
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        text_input_ids = torch.randint(0, self.vocab_size, (batch_size, self.language_seq_len))
        attention_mask = torch.ones((batch_size, self.language_seq_len))
        
        # Set the model to eval mode
        model.eval()
        
        # Run a forward pass
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask
            )
        
        # Check that the output has the expected structure
        self.assertIn("logits", outputs)
        self.assertIn("embeddings", outputs)
        self.assertIn("multimodal_embeddings", outputs)
        self.assertIn("response_logits", outputs)
        self.assertIn("response_features", outputs)
        
        # Check that output shapes are correct
        self.assertEqual(outputs["logits"].shape, (batch_size, self.config["model"]["num_classes"]))
        self.assertEqual(outputs["embeddings"].shape, (batch_size, self.vision_hidden_size))
        self.assertEqual(outputs["multimodal_embeddings"].shape, (batch_size, self.language_seq_len, self.language_hidden_size))
        self.assertEqual(outputs["response_logits"].shape, (batch_size, self.language_seq_len, self.vocab_size))
        
        # Both vision encoder and language model should be called exactly once
        self.mock_vision_encoder.assert_called_once()
        self.mock_language_model.assert_called_once()
    
    def test_prepare_inputs_flow(self):
        """Test the input preparation and tokenization flow."""
        # Create a model instance
        model = InternVL2MultimodalModel(
            config=self.config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Create a simple tokenizer mock that returns predictable outputs
        class MockTokenizerSimple:
            def __init__(self):
                self.vocab_size = 30000
            
            def __call__(self, text_prompts, padding, truncation, max_length, return_tensors):
                # Create predictable tokenizer outputs based on inputs
                batch_size = len(text_prompts)
                seq_len = max_length
                
                # Create mock tensors
                input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                
                # Create a container that mimics the tokenizer output
                class TokenizerOutput:
                    def __init__(self, input_ids, attention_mask):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                
                return TokenizerOutput(input_ids, attention_mask)
        
        # Replace tokenizer with our simple mock
        model.tokenizer = MockTokenizerSimple()
        
        # Create inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_prompts = ["How many receipts are in this image?", "What is the total value?"]
        
        # Prepare inputs
        inputs = model.prepare_inputs(images, text_prompts)
        
        # Check the structure of the prepared inputs
        self.assertIn("pixel_values", inputs)
        self.assertIn("text_input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        
        # Check shapes
        self.assertEqual(inputs["pixel_values"].shape, (batch_size, 3, 224, 224))
        self.assertEqual(inputs["text_input_ids"].shape, (batch_size, 128))  # Default max_length
        self.assertEqual(inputs["attention_mask"].shape, (batch_size, 128))
        
        # Check device consistency
        self.assertEqual(inputs["pixel_values"].device, images.device)
        self.assertEqual(inputs["text_input_ids"].device, images.device)
        self.assertEqual(inputs["attention_mask"].device, images.device)
    
    def test_generate_response_flow(self):
        """Test the response generation flow from end to end."""
        # Create a model instance
        model = InternVL2MultimodalModel(
            config=self.config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Replace components with our mocks
        model.vision_encoder = self.mock_vision_encoder
        model.language_model = self.mock_language_model
        
        # Create a simple tokenizer mock for decoding
        class MockTokenizerWithDecode:
            def __init__(self):
                self.vocab_size = 30000
            
            def __call__(self, text_prompts, padding, truncation, max_length, return_tensors):
                # Create predictable tokenizer outputs
                batch_size = len(text_prompts)
                seq_len = max_length
                
                # Create mock tensors
                input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                
                class TokenizerOutput:
                    def __init__(self, input_ids, attention_mask):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                
                return TokenizerOutput(input_ids, attention_mask)
            
            def decode(self, token_ids, skip_special_tokens=True):
                # Return a fixed response for testing
                return f"This is a response for tokens: {token_ids[:5]}"
        
        # Replace tokenizer with our mock
        model.tokenizer = MockTokenizerWithDecode()
        
        # Create a mock response generator
        mock_response_generator = MagicMock()
        mock_response_generator.generate.return_value = [
            [101, 102, 103, 104, 105],  # First batch response
            [201, 202, 203, 204, 205]   # Second batch response
        ]
        
        # Replace the response generator
        model.response_generator = mock_response_generator
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        text_input_ids = torch.randint(0, self.vocab_size, (batch_size, self.language_seq_len))
        attention_mask = torch.ones(batch_size, self.language_seq_len)
        
        # Mock the model's forward method to return a specific structure
        original_forward = model.forward
        
        def mock_forward(*args, **kwargs):
            return {
                "logits": torch.randn(batch_size, self.config["model"]["num_classes"]),
                "embeddings": torch.randn(batch_size, self.vision_hidden_size),
                "multimodal_embeddings": torch.randn(batch_size, self.language_seq_len, self.language_hidden_size),
                "response_logits": torch.randn(batch_size, self.language_seq_len, self.vocab_size),
                "response_features": torch.randn(batch_size, self.language_seq_len, self.language_hidden_size // 2)
            }
        
        # Replace forward method
        model.forward = mock_forward
        
        # Set the model to eval mode
        model.eval()
        
        # Generate response
        with torch.no_grad():
            token_ids, decoded_texts = model.generate_response(
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                max_length=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
        
        # Check the output structure
        self.assertIsInstance(token_ids, list)
        self.assertIsInstance(decoded_texts, list)
        self.assertEqual(len(token_ids), batch_size)
        self.assertEqual(len(decoded_texts), batch_size)
        
        # Check that the response generator was called with appropriate parameters
        mock_response_generator.generate.assert_called_once()
        
        # Restore original forward method
        model.forward = original_forward
    
    def test_vision_language_integration(self):
        """Test the integration between vision and language components."""
        # Create real components for cross-attention
        cross_attention = CrossAttention(
            query_dim=self.language_hidden_size,
            key_value_dim=self.vision_hidden_size,
            embed_dim=self.language_hidden_size,
            num_heads=8,
            dropout=0.0
        )
        
        # Create a model instance
        model = InternVL2MultimodalModel(
            config=self.config,
            pretrained=True,
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        
        # Replace components with our mocks
        model.vision_encoder = self.mock_vision_encoder
        model.language_model = self.mock_language_model
        model.cross_attention = cross_attention  # Use real cross-attention
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        text_input_ids = torch.randint(0, self.vocab_size, (batch_size, self.language_seq_len))
        attention_mask = torch.ones(batch_size, self.language_seq_len)
        
        # Create mock outputs for vision and language models
        vision_output = torch.randn(batch_size, self.vision_seq_len, self.vision_hidden_size)
        language_output = torch.randn(batch_size, self.language_seq_len, self.language_hidden_size)
        
        # Mock the model's encoding methods to return these outputs
        model.vision_encoder.return_value.last_hidden_state = vision_output
        model.language_model.return_value.last_hidden_state = language_output
        
        # Set the model to eval mode
        model.eval()
        
        # Run a forward pass
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask
            )
        
        # Compute the expected cross-modal attention output
        expected_cross_attn = cross_attention(
            query=language_output,
            key=vision_output,
            value=vision_output,
            attention_mask=None
        )
        
        # The multimodal embeddings should be the result of cross-attention
        self.assertEqual(outputs["multimodal_embeddings"].shape, expected_cross_attn.shape)
        
        # Check that logits can be computed from these embeddings
        self.assertEqual(outputs["logits"].shape, (batch_size, self.config["model"]["num_classes"]))


class TestComponentInteractions(unittest.TestCase):
    """Test interactions between different components of the architecture."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Define dimensions
        self.vision_dim = 768
        self.language_dim = 768
        self.vocab_size = 30000
        self.batch_size = 2
        self.vision_seq_len = 197
        self.language_seq_len = 20
        
        # Create components
        self.cross_attention = CrossAttention(
            query_dim=self.language_dim,
            key_value_dim=self.vision_dim,
            embed_dim=self.language_dim,
            num_heads=8,
            dropout=0.0
        )
        
        self.response_generator = ResponseGenerator(
            input_dim=self.language_dim,
            hidden_dims=[512, 256],
            vocab_size=self.vocab_size,
            max_length=50,
            dropout_rates=[0.0, 0.0],
            use_batchnorm=False,
            activation="gelu"
        )
    
    def test_cross_attention_to_response_generator(self):
        """Test the flow from cross-attention to response generator."""
        # Create dummy inputs
        vision_features = torch.randn(self.batch_size, self.vision_seq_len, self.vision_dim)
        language_features = torch.randn(self.batch_size, self.language_seq_len, self.language_dim)
        
        # Apply cross-attention
        cross_modal_features = self.cross_attention(
            query=language_features,
            key=vision_features,
            value=vision_features
        )
        
        # Check shape
        self.assertEqual(cross_modal_features.shape, (self.batch_size, self.language_seq_len, self.language_dim))
        
        # Generate response from cross-modal features
        response_output = self.response_generator(cross_modal_features)
        
        # Check output
        self.assertIn("logits", response_output)
        self.assertIn("features", response_output)
        
        # Check shapes
        self.assertEqual(response_output["logits"].shape, (self.batch_size, self.language_seq_len, self.vocab_size))
        self.assertEqual(response_output["features"].shape, (self.batch_size, self.language_seq_len, 256))
    
    def test_response_generation_from_cross_attention(self):
        """Test generating text tokens from cross-attention output."""
        # Create dummy inputs
        vision_features = torch.randn(self.batch_size, self.vision_seq_len, self.vision_dim)
        language_features = torch.randn(self.batch_size, self.language_seq_len, self.language_dim)
        
        # Apply cross-attention
        cross_modal_features = self.cross_attention(
            query=language_features,
            key=vision_features,
            value=vision_features
        )
        
        # Create starting tokens for generation
        start_tokens = torch.randint(0, self.vocab_size, (self.batch_size, 1))
        
        # Set up mocks for the token embedding
        original_embedding = self.response_generator.token_embedding
        mock_embedding = MagicMock()
        mock_embedding.return_value = torch.randn(self.batch_size, 1, self.language_dim)
        self.response_generator.token_embedding = mock_embedding
        
        # Generate tokens
        try:
            generated_ids = self.response_generator.generate(
                start_tokens=start_tokens,
                multimodal_context=cross_modal_features,
                temperature=1.0,
                top_k=50,
                top_p=1.0
            )
            
            # Check output
            self.assertIsInstance(generated_ids, list)
            self.assertEqual(len(generated_ids), self.batch_size)
            
            # Each item should be a list of token IDs
            for token_ids in generated_ids:
                self.assertIsInstance(token_ids, list)
                self.assertGreaterEqual(len(token_ids), 1)
                self.assertLessEqual(len(token_ids), self.response_generator.max_length)
        finally:
            # Restore original embedding
            self.response_generator.token_embedding = original_embedding


if __name__ == "__main__":
    unittest.main()