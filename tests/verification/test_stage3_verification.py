#!/usr/bin/env python3
"""
Verification tests for Stage 3 (Training Pipeline).

This module contains tests to verify the correctness of the existing
implementation of Stage 3, which focuses on the training pipeline
for both vision-only and multimodal models.
"""

import os
import unittest
import tempfile
from pathlib import Path
import sys
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from training.trainer import InternVL2Trainer
from training.multimodal_trainer import MultimodalTrainer
from training.multimodal_loss import MultimodalLoss
from models.internvl2 import InternVL2ReceiptClassifier, InternVL2MultimodalModel


class MockVisionEncoder(nn.Module):
    """Mock vision encoder for testing."""
    def __init__(self, hidden_size=512):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(32 * 4 * 4, hidden_size)
        self.config = type('', (), {})()
        self.config.hidden_size = hidden_size
        
    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        # Reshape to match expected output shape [batch_size, seq_len, hidden_size]
        x = x.unsqueeze(1).expand(-1, 16, -1)  # Expand to 16 sequence length
        return type('', (), {'last_hidden_state': x})()


class MockLanguageModel(nn.Module):
    """Mock language model for testing."""
    def __init__(self, hidden_size=512, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.config = type('', (), {})()
        self.config.hidden_size = hidden_size
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
        x = self.embedding(input_ids)
        outputs, _ = self.lstm(x)
        
        if return_dict:
            return {'last_hidden_state': outputs, 'hidden_states': [outputs]}
        return type('', (), {'last_hidden_state': outputs, 'hidden_states': [outputs]})()


class TestTrainingPipeline(unittest.TestCase):
    """Test the implementation of the training pipeline for both models."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)
        
        # Device for testing (use CPU for consistent testing)
        cls.device = torch.device("cpu")
        
        # Create configurations for testing
        cls.create_test_configs()
        
        # Create mock data for testing
        cls.create_mock_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up the temporary directory
        cls.temp_dir.cleanup()
    
    def tearDown(self):
        """Clean up after each test."""
        # Make sure to clean up CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @classmethod
    def create_test_configs(cls):
        """Create test configurations for trainers."""
        # Vision-only model config
        cls.vision_config = {
            "model": {
                "pretrained_path": str(cls.test_dir / "mock_model"),
                "num_classes": 3,
                "classifier": {
                    "hidden_dims": [256, 128],
                    "dropout_rates": [0.1, 0.1],
                    "batch_norm": True,
                    "activation": "gelu"
                }
            },
            "training": {
                "epochs": 2,
                "loss": {
                    "name": "cross_entropy",
                    "label_smoothing": 0.1
                },
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "backbone_lr_multiplier": 0.1,
                    "gradient_clip": 1.0
                },
                "scheduler": {
                    "name": "cosine",
                    "min_lr_factor": 0.1
                },
                "mixed_precision": False,
                "three_stage": {
                    "enabled": True,
                    "mlp_warmup_epochs": 1,
                    "vision_tuning_epochs": 1
                },
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.01
                }
            },
            "output": {
                "model_dir": str(cls.test_dir / "outputs"),
                "results_dir": str(cls.test_dir / "results"),
                "log_dir": str(cls.test_dir / "logs"),
                "tensorboard": False,
                "checkpoint_frequency": 1,
                "save_best_only": False
            }
        }
        
        # Multimodal model config
        cls.multimodal_config = {
            "model": {
                "pretrained_path": str(cls.test_dir / "mock_model"),
                "num_classes": 3,
                "classifier": {
                    "hidden_dims": [256, 128],
                    "dropout_rates": [0.1, 0.1],
                    "batch_norm": True,
                    "activation": "gelu"
                },
                "multimodal": True
            },
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "fp16": False,
                "gradient_clip": 1.0,
                "loss_weights": {
                    "classification": 1.0,
                    "language": 1.0
                },
                "scheduler": {
                    "name": "cosine",
                    "min_lr_factor": 0.1
                },
                "three_stage": {
                    "enabled": True,
                    "stage2": {
                        "start_epoch": 1,
                        "lr_multiplier": 0.1
                    },
                    "stage3": {
                        "start_epoch": 2,
                        "lr_multiplier": 0.1
                    }
                },
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.01
                },
                "language_lr_multiplier": 0.1
            },
            "output": {
                "model_dir": str(cls.test_dir / "outputs"),
                "results_dir": str(cls.test_dir / "results"),
                "log_dir": str(cls.test_dir / "logs"),
                "tensorboard": False,
                "save_frequency": 1,
                "save_best_only": False
            },
            "data": {
                "max_text_length": 32
            }
        }
    
    @classmethod
    def create_mock_data(cls):
        """Create mock data for training."""
        # Create mock datasets for vision-only model
        batch_size = 4
        
        # Vision data: images and labels
        images = torch.randn(batch_size * 2, 3, 224, 224)
        labels = torch.randint(0, 3, (batch_size * 2,))
        
        # Create TensorDatasets
        vision_dataset = TensorDataset(images, labels)
        
        # Create DataLoaders
        cls.vision_dataloaders = {
            "train": DataLoader(vision_dataset, batch_size=batch_size, shuffle=True),
            "val": DataLoader(vision_dataset, batch_size=batch_size, shuffle=False)
        }
        
        # Create mock multimodal data
        # Simulate a multimodal batch structure
        cls.multimodal_batch = {
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
            "text_input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "text_attention_mask": torch.ones(batch_size, 32),
            "labels": torch.randint(0, 1000, (batch_size, 32)),
            "labels_attention_mask": torch.ones(batch_size, 32),
            "classification_labels": torch.randint(0, 3, (batch_size,)),
            "receipt_count": torch.randint(0, 3, (batch_size,))
        }
    
    def create_mock_vision_model(self):
        """Create a mock vision model for testing."""
        # Create a proper mock model that inherits from nn.Module
        class MockVisionModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.logger = type('', (), {'info': print, 'warning': print, 'error': print})()
                self.vision_encoder = MockVisionEncoder()
                self.classification_head = nn.Linear(512, 3)
            
            def forward(self, pixel_values):
                vision_outputs = self.vision_encoder(pixel_values)
                embeddings = vision_outputs.last_hidden_state.mean(dim=1)
                logits = self.classification_head(embeddings)
                return {"logits": logits, "embeddings": embeddings}
            
            def unfreeze_vision_encoder(self, lr_multiplier=0.1):
                return [{'params': self.classification_head.parameters()},
                        {'params': self.vision_encoder.parameters(), 'lr': 1e-5}]
                
        # Create an instance of our mock model
        model = MockVisionModel(self.vision_config)
        
        # Freeze the vision encoder initially (to match expected behavior)
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
            
        return model
    
    def create_mock_multimodal_model(self):
        """Create a mock multimodal model for testing."""
        # Create a proper mock model that inherits from nn.Module
        class MockMultimodalModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.logger = type('', (), {'info': print, 'warning': print, 'error': print})()
                
                # Model components
                self.vision_encoder = MockVisionEncoder(hidden_size=512)
                self.language_model = MockLanguageModel(hidden_size=512, vocab_size=1000)
                self.classification_head = nn.Linear(512, 3)
                self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
                self.response_generator = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.Linear(512, 1000)
                )
                
                # Mock tokenizer
                class MockTokenizer:
                    def __init__(self):
                        self.vocab_size = 1000
                    
                    def decode(self, token_ids, skip_special_tokens=True):
                        return "Test response"
                
                self.tokenizer = MockTokenizer()
            
            def forward(self, pixel_values, text_input_ids=None, attention_mask=None):
                vision_outputs = self.vision_encoder(pixel_values)
                image_embeds = vision_outputs.last_hidden_state
                
                # Classification logits from image embeddings
                pooled_vision = image_embeds.mean(dim=1)
                classification_logits = self.classification_head(pooled_vision)
                
                if text_input_ids is not None:
                    # Text encoding
                    text_outputs = self.language_model(text_input_ids, attention_mask)
                    text_embeds = text_outputs.last_hidden_state
                    
                    # Cross-modal attention
                    multimodal_embeds, _ = self.cross_attention(
                        text_embeds, image_embeds, image_embeds
                    )
                    
                    # Create mock response logits
                    response_logits = torch.randn(text_input_ids.size(0), text_input_ids.size(1), 1000)
                    
                    return {
                        "logits": classification_logits,
                        "embeddings": pooled_vision,
                        "multimodal_embeddings": multimodal_embeds,
                        "response_logits": response_logits,
                    }
                else:
                    return {
                        "logits": classification_logits,
                        "embeddings": pooled_vision
                    }
            
            def generate_response(self, pixel_values, text_input_ids, attention_mask=None, max_length=50, **kwargs):
                batch_size = text_input_ids.size(0)
                # Return mocked generated IDs and texts
                generated_ids = [list(range(10)) for _ in range(batch_size)]
                decoded_texts = ["Generated text response"] * batch_size
                return generated_ids, decoded_texts
        
        # Create an instance of our mock model
        model = MockMultimodalModel(self.multimodal_config)
        
        # Freeze the vision encoder initially (to match expected behavior)
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
            
        return model
    
    def test_loss_function_implementation(self):
        """Verify that loss functions are implemented correctly."""
        # Test CrossEntropyLoss for vision-only model
        batch_size = 4
        num_classes = 3
        
        # Create dummy logits and targets
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Create loss function
        ce_loss = nn.CrossEntropyLoss()
        loss_value = ce_loss(logits, targets)
        
        # Verify loss shape and type
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertEqual(loss_value.ndim, 0)  # Scalar tensor
        self.assertTrue(loss_value > 0)  # Loss should be positive
        
        # Test MultimodalLoss
        multimodal_loss = MultimodalLoss(
            classification_weight=1.0,
            language_weight=1.0
        )
        
        # Create dummy model outputs
        model_outputs = {
            "logits": logits,
            "response_logits": torch.randn(batch_size, 10, 1000)  # [batch_size, seq_len, vocab_size]
        }
        
        # Create dummy targets
        classification_labels = targets
        language_labels = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        
        # Calculate loss
        loss_dict = multimodal_loss(
            model_outputs=model_outputs,
            classification_labels=classification_labels,
            language_labels=language_labels,
            attention_mask=attention_mask
        )
        
        # Verify loss structure
        self.assertIn("total_loss", loss_dict)
        self.assertIn("classification_loss", loss_dict)
        self.assertIn("language_loss", loss_dict)
        
        # Verify loss values
        self.assertTrue(loss_dict["total_loss"] > 0)
        self.assertTrue(loss_dict["classification_loss"] > 0)
        self.assertTrue(loss_dict["language_loss"] > 0)
        
        # Verify loss weighting
        expected_total = loss_dict["classification_loss"] + loss_dict["language_loss"]
        self.assertAlmostEqual(loss_dict["total_loss"].item(), expected_total.item(), places=5)
        
        # Test with different weights
        multimodal_loss_weighted = MultimodalLoss(
            classification_weight=0.3,
            language_weight=0.7
        )
        
        loss_dict_weighted = multimodal_loss_weighted(
            model_outputs=model_outputs,
            classification_labels=classification_labels,
            language_labels=language_labels,
            attention_mask=attention_mask
        )
        
        # Verify weighted loss
        expected_weighted_total = (0.3 * loss_dict_weighted["classification_loss"] + 
                                   0.7 * loss_dict_weighted["language_loss"])
        self.assertAlmostEqual(loss_dict_weighted["total_loss"].item(), 
                              expected_weighted_total.item(), places=5)
    
    def test_optimizer_configuration(self):
        """Verify that optimizers are configured correctly."""
        # Test vision-only trainer optimizer configuration
        vision_model = self.create_mock_vision_model()
        trainer = InternVL2Trainer(
            config=self.vision_config,
            model=vision_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Check optimizer type
        self.assertIsInstance(trainer.optimizer, optim.AdamW)
        
        # Check learning rate
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-4)
        
        # Test scheduler configuration
        self.assertIsInstance(trainer.scheduler, optim.lr_scheduler.CosineAnnealingLR)
        
        # Test MultimodalTrainer optimizer configuration
        multimodal_model = self.create_mock_multimodal_model()
        multimodal_trainer = MultimodalTrainer(
            config=self.multimodal_config,
            model=multimodal_model,
            dataloaders=self.vision_dataloaders,  # Reuse for simplicity
            output_dir=self.test_dir / "outputs"
        )
        
        # Check optimizer type and learning rates for different parameter groups
        self.assertIsInstance(multimodal_trainer.optimizer, optim.AdamW)
        
        # Check multiple parameter groups
        self.assertGreater(len(multimodal_trainer.optimizer.param_groups), 1)
        
        # Check scheduler
        self.assertIsInstance(multimodal_trainer.scheduler, optim.lr_scheduler.CosineAnnealingLR)
    
    def test_training_stage_transitions(self):
        """Verify that training stage transitions work correctly."""
        # Create a minimal InternVL2ReceiptClassifier with tracking capabilities
        vision_model = self.create_mock_vision_model()
        
        # Add tracking of parameter grad states
        initial_grad_states = {}
        for name, param in vision_model.vision_encoder.named_parameters():
            initial_grad_states[name] = param.requires_grad
        
        # Create trainer
        trainer = InternVL2Trainer(
            config=self.vision_config,
            model=vision_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Check initial state (Stage 1: frozen vision encoder)
        for name, param in vision_model.vision_encoder.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen in Stage 1")
        
        # Manually trigger Stage 2 transition (unfreeze vision encoder)
        param_groups = vision_model.unfreeze_vision_encoder(
            lr_multiplier=self.vision_config["training"]["optimizer"]["backbone_lr_multiplier"]
        )
        
        # Create new optimizer with unfrozen params
        trainer.optimizer = trainer._get_optimizer(param_groups)
        
        # Check parameter groups in optimizer
        self.assertEqual(len(trainer.optimizer.param_groups), 2)
        
        # Clean up
        del trainer
        
        # Test MultimodalTrainer stage transitions
        multimodal_model = self.create_mock_multimodal_model()
        
        # Freeze vision encoder for initial stage
        for param in multimodal_model.vision_encoder.parameters():
            param.requires_grad = False
        
        multimodal_trainer = MultimodalTrainer(
            config=self.multimodal_config,
            model=multimodal_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Check initial parameter states
        for param in multimodal_model.vision_encoder.parameters():
            self.assertFalse(param.requires_grad, "Vision encoder should be frozen in Stage 1")
        
        # Test stage 2 transition (internal method)
        unfrozen_params = multimodal_trainer._unfreeze_vision_encoder(lr_multiplier=0.1)
        
        # Verify some parameters were unfrozen
        self.assertGreater(len(unfrozen_params), 0)
        
        # Test stage transition by directly calling _configure_optimizer
        stage2_optimizer = multimodal_trainer._configure_optimizer(stage=2)
        
        # Verify multiple parameter groups with different learning rates
        self.assertGreater(len(stage2_optimizer.param_groups), 1)
        
        # Check that at least one vision encoder parameter is unfrozen
        any_unfrozen = False
        for param in multimodal_model.vision_encoder.parameters():
            if param.requires_grad:
                any_unfrozen = True
                break
        
        self.assertTrue(any_unfrozen, "At least one vision encoder parameter should be unfrozen in Stage 2")
    
    def test_gradient_flow(self):
        """Verify gradient flow through the models."""
        # Test gradient flow in vision-only model
        vision_model = self.create_mock_vision_model()
        
        # Enable gradients for all parameters for this test
        for param in vision_model.parameters():
            param.requires_grad = True
        
        # Forward pass
        batch = next(iter(self.vision_dataloaders["train"]))
        images, targets = [t.to(self.device) for t in batch]
        
        outputs = vision_model(images)
        loss = nn.CrossEntropyLoss()(outputs["logits"], targets)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in vision_model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient")
        
        # Test gradient flow in multimodal model
        multimodal_model = self.create_mock_multimodal_model()
        
        # Enable gradients for all parameters
        for param in multimodal_model.parameters():
            param.requires_grad = True
        
        # Mock batch
        batch = {k: v.to(self.device) for k, v in self.multimodal_batch.items()}
        
        # Forward pass
        outputs = multimodal_model(
            pixel_values=batch["pixel_values"],
            text_input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"]
        )
        
        # Calculate loss
        multimodal_loss = MultimodalLoss()
        loss_dict = multimodal_loss(
            model_outputs=outputs,
            classification_labels=batch["classification_labels"],
            language_labels=batch["labels"],
            attention_mask=batch["labels_attention_mask"]
        )
        
        # Backward pass
        loss_dict["total_loss"].backward()
        
        # Check gradients - focus on main components
        for name, component in [
            ('vision_encoder', multimodal_model.vision_encoder),
            ('language_model', multimodal_model.language_model),
            ('cross_attention', multimodal_model.cross_attention),
            ('classification_head', multimodal_model.classification_head)
        ]:
            any_grad = False
            for param in component.parameters():
                if param.requires_grad and param.grad is not None:
                    any_grad = True
                    break
            
            self.assertTrue(any_grad, f"Component {name} has no gradient")
    
    def test_trainer_epoch_methods(self):
        """Verify that training and validation methods work correctly."""
        # Test vision-only trainer
        vision_model = self.create_mock_vision_model()
        trainer = InternVL2Trainer(
            config=self.vision_config,
            model=vision_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Run one training epoch
        train_loss, train_acc = trainer.train_epoch(epoch=1)
        
        # Verify outputs
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 100.0)
        
        # Run validation
        val_loss, val_acc = trainer.validate(epoch=1)
        
        # Verify outputs
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 100.0)
        
        # Test multimodal trainer
        multimodal_model = self.create_mock_multimodal_model()
        
        # For testing, we need to mock the dataloader to return batches in the expected format
        class MockDataLoader:
            def __init__(self, batch, num_batches=2):
                self.batch = batch
                self.num_batches = num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield self.batch
            
            def __len__(self):
                return self.num_batches
        
        # Move batch tensors to the device
        device_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in self.multimodal_batch.items()}
        
        mock_dataloaders = {
            "train": MockDataLoader(device_batch),
            "val": MockDataLoader(device_batch)
        }
        
        multimodal_trainer = MultimodalTrainer(
            config=self.multimodal_config,
            model=multimodal_model,
            dataloaders=mock_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Run one training epoch
        train_metrics = multimodal_trainer.train_epoch(epoch=1)
        
        # Verify outputs
        self.assertIn("loss", train_metrics)
        self.assertIn("accuracy", train_metrics)
        self.assertIn("bleu", train_metrics)
        
        # Run validation
        val_metrics = multimodal_trainer.validate(epoch=1)
        
        # Verify outputs
        self.assertIn("loss", val_metrics)
        self.assertIn("accuracy", val_metrics)
        self.assertIn("bleu", val_metrics)
    
    def test_checkpoint_saving(self):
        """Verify that checkpoints are saved correctly."""
        # Test vision-only trainer checkpoint saving
        vision_model = self.create_mock_vision_model()
        trainer = InternVL2Trainer(
            config=self.vision_config,
            model=vision_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Create checkpoint
        trainer.save_checkpoint(epoch=1, is_best=True)
        
        # Verify checkpoint files
        checkpoint_path = Path(self.test_dir) / "outputs" / "checkpoints" / "model_epoch_1.pt"
        best_path = Path(self.test_dir) / "outputs" / "best_model.pt"
        
        self.assertTrue(checkpoint_path.exists(), "Checkpoint file should exist")
        self.assertTrue(best_path.exists(), "Best model file should exist")
        
        # Load checkpoint and verify contents
        checkpoint = torch.load(checkpoint_path)
        
        self.assertIn("epoch", checkpoint)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        
        # Test multimodal trainer checkpoint saving
        multimodal_model = self.create_mock_multimodal_model()
        multimodal_trainer = MultimodalTrainer(
            config=self.multimodal_config,
            model=multimodal_model,
            dataloaders=self.vision_dataloaders,
            output_dir=self.test_dir / "outputs"
        )
        
        # Create checkpoint
        val_metrics = {
            "loss": 1.0,
            "accuracy": 75.0,
            "bleu": 0.5
        }
        multimodal_trainer.save_checkpoint(epoch=1, metrics=val_metrics, is_best=True)
        
        # Verify checkpoint files (should overwrite previous files)
        self.assertTrue(checkpoint_path.exists(), "Checkpoint file should exist")
        self.assertTrue(best_path.exists(), "Best model file should exist")
        
        # Load checkpoint and verify contents
        checkpoint = torch.load(checkpoint_path)
        
        self.assertIn("epoch", checkpoint)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("metrics", checkpoint)
        
        # Verify metrics are saved
        self.assertEqual(checkpoint["metrics"]["loss"], 1.0)
        self.assertEqual(checkpoint["metrics"]["accuracy"], 75.0)
        self.assertEqual(checkpoint["metrics"]["bleu"], 0.5)


if __name__ == "__main__":
    unittest.main()