#!/usr/bin/env python3
"""
Verification tests for Stage 4 (Training Orchestration and Evaluation).

This module contains tests to verify the correctness of the existing
implementation of Stage 4, which focuses on training orchestration,
monitoring, and evaluation for the multimodal vision-language model.
"""

import os
import unittest
import tempfile
from pathlib import Path
import sys
import json
import yaml
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from scripts.train_orchestrator import TrainingOrchestrator
from scripts.training_monitor import TrainingMonitor
from scripts.evaluate_multimodal import evaluate_model, visualize_attention
from utils.metrics import compute_classification_metrics, compute_nlg_metrics


class MockModel(nn.Module):
    """Mock model for testing the evaluation and monitoring systems."""
    
    def __init__(self, device=None):
        super().__init__()
        # Get device or default to CPU
        self.device = device if device is not None else torch.device("cpu")
        
        # Create model components
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        ).to(self.device)
        
        self.classification_head = nn.Linear(32 * 7 * 7, 3).to(self.device)
        self.cross_attention = nn.Linear(32 * 7 * 7, 512).to(self.device)
        self.response_generator = nn.Linear(512, 1000).to(self.device)
        
        # Create tokenizer mock
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                
            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    # Just return a fixed response for testing
                    return "This is a mock response"
                return "This is a mock response"
                
            def __call__(self, text, **kwargs):
                # Return a mock encoding object
                if isinstance(text, str):
                    text = [text]
                    
                batch_size = len(text)
                seq_len = kwargs.get('max_length', 32)
                
                return {
                    'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device=self.device),
                    'attention_mask': torch.ones((batch_size, seq_len), device=self.device)
                }
        
        self.tokenizer = MockTokenizer()
        
    def to(self, device):
        """Override to method to handle device transfer properly."""
        self.device = device
        super().to(device)
        self.vision_encoder.to(device)
        self.classification_head.to(device)
        self.cross_attention.to(device)
        self.response_generator.to(device)
        return self
    
    def forward(self, pixel_values, text_input_ids=None, attention_mask=None):
        batch_size = pixel_values.shape[0]
        
        # Vision encoding
        x = self.vision_encoder(pixel_values)
        x_flat = x.view(batch_size, -1)
        
        # Classification
        logits = self.classification_head(x_flat)
        
        # Text processing if provided
        if text_input_ids is not None:
            # Mock cross-attention
            vision_features = self.cross_attention(x_flat)
            
            # Mock response generation
            response_logits = self.response_generator(vision_features).unsqueeze(1).expand(-1, text_input_ids.size(1), -1)
            
            return {
                "logits": logits,
                "response_logits": response_logits,
                "embeddings": vision_features
            }
        
        return {"logits": logits}
    
    def generate_response(self, pixel_values, text_input_ids, attention_mask=None, max_length=50, **kwargs):
        batch_size = pixel_values.shape[0]
        generated_ids = [torch.randint(0, 1000, (10,)) for _ in range(batch_size)]
        decoded_texts = ["This is a mock response"] * batch_size
        return generated_ids, decoded_texts
    
    def prepare_inputs(self, pixel_values, questions):
        text_inputs = self.tokenizer(questions, padding=True, truncation=True, max_length=32, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "text_input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"]
        }
    
    def get_attention_maps(self, pixel_values):
        # Return a mock attention map
        batch_size = pixel_values.shape[0]
        return [torch.rand(batch_size, 8, 196, 196)]  # [batch_size, num_heads, seq_len, seq_len]


class TestStage4Verification(unittest.TestCase):
    """Test suite for Stage 4 verification."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)
        
        # Create subdirectories for experiments
        cls.experiments_dir = cls.test_dir / "experiments"
        cls.experiments_dir.mkdir(parents=True)
        
        # Create mock experiment data
        cls._create_mock_experiments()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Clean up temporary directory
        cls.temp_dir.cleanup()
    
    @classmethod
    def _create_mock_experiments(cls):
        """Create mock experiment data for testing."""
        # Create two mock experiments
        exp_names = ["baseline", "improved"]
        
        for exp_name in exp_names:
            exp_dir = cls.experiments_dir / exp_name
            exp_dir.mkdir(parents=True)
            
            # Create model directory
            model_dir = exp_dir / "model"
            model_dir.mkdir(parents=True)
            
            # Create evaluation directory
            eval_dir = exp_dir / "evaluation"
            eval_dir.mkdir(parents=True)
            
            # Create tensorboard directory
            tb_dir = model_dir / "tensorboard"
            tb_dir.mkdir(parents=True)
            
            # Create config file
            config = {
                "model": {
                    "name": "InternVL2",
                    "multimodal": True,
                    "pretrained": True,
                    "num_classes": 3
                },
                "training": {
                    "epochs": 10,
                    "learning_rate": 1e-4 if exp_name == "baseline" else 2e-4,
                    "batch_size": 32,
                    "loss_weights": {
                        "classification": 1.0,
                        "language": 1.0 if exp_name == "baseline" else 1.5
                    }
                },
                "data": {
                    "dataset": "receipts",
                    "max_text_length": 32
                }
            }
            
            with open(exp_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)
            
            # Create evaluation metrics
            metrics = {
                "classification": {
                    "accuracy": 85.0 if exp_name == "baseline" else 88.5,
                    "precision": 84.2 if exp_name == "baseline" else 87.3,
                    "recall": 83.5 if exp_name == "baseline" else 86.9,
                    "f1": 83.8 if exp_name == "baseline" else 87.1
                },
                "generation": {
                    "bleu": 0.35 if exp_name == "baseline" else 0.42,
                    "rouge1_f": 0.45 if exp_name == "baseline" else 0.52,
                    "rouge2_f": 0.30 if exp_name == "baseline" else 0.38
                }
            }
            
            with open(eval_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
    
    def test_training_orchestrator_initialization(self):
        """Test that TrainingOrchestrator initializes correctly."""
        # Create a base config for testing
        config_path = self.test_dir / "test_config.yaml"
        config = {
            "model": {
                "name": "InternVL2",
                "multimodal": True
            },
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            base_config_path=str(config_path),
            experiments_dir=str(self.experiments_dir)
        )
        
        # Verify initialization
        self.assertEqual(orchestrator.base_config_path, config_path)
        self.assertEqual(orchestrator.experiments_dir, self.experiments_dir)
        self.assertEqual(orchestrator.base_config, config)
        self.assertIsNone(orchestrator.ablation_config)
        self.assertEqual(orchestrator.experiments, [])
    
    def test_run_single_experiment(self):
        """Test running a single experiment with mocked training process."""
        # Create a base config for testing
        config_path = self.test_dir / "test_config.yaml"
        config = {
            "model": {
                "name": "InternVL2",
                "multimodal": True
            },
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            base_config_path=str(config_path),
            experiments_dir=str(self.experiments_dir)
        )
        
        # Set up expected evaluation metrics
        eval_metrics = {
            "classification": {
                "accuracy": 85.0,
                "precision": 84.2
            },
            "generation": {
                "bleu": 0.35,
                "rouge1_f": 0.45
            }
        }
        
        # Mock the subprocess.run method to avoid actual execution
        with patch('subprocess.run') as mock_run:
            # Set up mock return values
            mock_process = MagicMock()
            mock_process.stdout = "Training completed successfully"
            mock_process.stderr = ""
            mock_run.return_value = mock_process
            
            # Mock the _extract_tensorboard_metrics method
            with patch.object(orchestrator, '_extract_tensorboard_metrics') as mock_extract:
                mock_extract.return_value = {
                    "train/loss": {"steps": [0, 10, 20], "values": [1.0, 0.8, 0.6]},
                    "val/loss": {"steps": [10, 20], "values": [0.9, 0.7]}
                }
                
                # Create a real metrics.json file that will be found by the code
                experiment_dir = orchestrator.experiments_dir / "test_experiment"
                experiment_dir.mkdir(exist_ok=True)
                
                eval_dir = experiment_dir / "evaluation"
                eval_dir.mkdir(exist_ok=True)
                
                # Write the actual metrics file that will be read
                metrics_path = eval_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(eval_metrics, f)
                
                # Set up appropriate patches
                def mock_path_exists_side_effect(path):
                    # Always return True for metrics.json check
                    if str(path).endswith('metrics.json'):
                        return True
                    # For other paths, do normal behavior
                    return Path(path).exists()
                
                with patch('pathlib.Path.exists', side_effect=mock_path_exists_side_effect):
                    # Run experiment
                    results = orchestrator.run_single_experiment(
                        config=config,
                        experiment_name="test_experiment",
                        seed=42
                    )
        
        # Verify that subprocess.run was called twice (once for training, once for evaluation)
        self.assertEqual(mock_run.call_count, 2)
        
        # Verify results structure
        self.assertIn("classification", results)
        self.assertIn("generation", results)
        self.assertEqual(results["classification"]["accuracy"], 85.0)
        self.assertEqual(results["generation"]["bleu"], 0.35)
    
    def test_training_monitor_initialization(self):
        """Test that TrainingMonitor initializes correctly."""
        # Create monitor
        monitor = TrainingMonitor(str(self.experiments_dir))
        
        # Verify initialization
        self.assertEqual(monitor.experiments_dir, self.experiments_dir)
        
        # Verify that experiments were found
        experiments = monitor.get_all_experiments()
        self.assertGreaterEqual(len(experiments), 2)
        self.assertIn("baseline", experiments)
        self.assertIn("improved", experiments)
    
    def test_compare_experiments(self):
        """Test comparing multiple experiments."""
        # Create monitor
        monitor = TrainingMonitor(str(self.experiments_dir))
        
        # Compare experiments
        comparison = monitor.compare_experiments(["baseline", "improved"])
        
        # Verify comparison structure
        self.assertIn("experiments", comparison)
        self.assertIn("metrics", comparison)
        self.assertIn("best", comparison)
        
        # Verify metrics included
        metrics = comparison["metrics"]
        self.assertIn("eval_accuracy", metrics)
        self.assertIn("eval_bleu", metrics)
        
        # Verify best experiment identification
        best = comparison["best"]
        self.assertEqual(best["eval_accuracy"]["experiment"], "improved")
        self.assertEqual(best["eval_bleu"]["experiment"], "improved")
    
    def test_model_evaluation(self):
        """Test model evaluation with mock model and data."""
        # Get device - use CPU to ensure consistency
        device = torch.device("cpu")
        
        # Create mock model
        model = MockModel(device=device)
        
        # Create mock dataloaders
        batch_size = 4
        
        # Create mock data batch - ensure everything is on the same device
        mock_batch = {
            "pixel_values": torch.randn(batch_size, 3, 224, 224, device=device),
            "text_input_ids": torch.randint(0, 1000, (batch_size, 32), device=device),
            "text_attention_mask": torch.ones(batch_size, 32, device=device),
            "labels": torch.randint(0, 1000, (batch_size, 32), device=device),
            "labels_attention_mask": torch.ones(batch_size, 32, device=device),
            "classification_labels": torch.randint(0, 3, (batch_size,), device=device),
            "receipt_count": torch.randint(0, 3, (batch_size,), device=device)
        }
        
        # Mock DataLoader class
        class MockDataLoader:
            def __init__(self, batch, num_batches=2):
                self.batch = batch
                self.batch_size = batch_size
                self.num_batches = num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield self.batch
            
            def __len__(self):
                return self.num_batches
        
        # Create dataloaders
        dataloaders = {
            "val": MockDataLoader(mock_batch),
            "test": MockDataLoader(mock_batch)
        }
        
        # Create output directory
        output_dir = self.test_dir / "evaluation"
        output_dir.mkdir(exist_ok=True)
        
        # Mock compute_nlg_metrics to avoid NLTK dependency
        with patch('utils.metrics.compute_nlg_metrics') as mock_compute_nlg:
            # Setup return value
            mock_compute_nlg.return_value = {
                "bleu": 0.35,
                "rouge1_f": 0.45,
                "rouge2_f": 0.30
            }
            
            # Also patch get_device to ensure CPU usage
            with patch('utils.device.get_device', return_value=device):
                # Ensure files can be written
                with patch('json.dump') as mock_json_dump:
                    # Run evaluation
                    metrics = evaluate_model(model, dataloaders, output_dir)
                    
                    # Verify metrics structure
                    self.assertIn("classification", metrics)
                    self.assertIn("generation", metrics)
                    
                    # Verify json.dump was called (metrics were saved)
                    self.assertTrue(mock_json_dump.called)
    
    def test_visualization_attention(self):
        """Test attention visualization with mock model and data."""
        # Get device - use CPU to ensure consistency
        device = torch.device("cpu")
        
        # Create mock model
        model = MockModel(device=device)
        
        # Create mock image path
        image_path = self.test_dir / "test_image.jpg"
        
        # Create a blank test image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        plt.imsave(image_path, img)
        
        # Create output directory
        output_dir = self.test_dir / "visualization"
        output_dir.mkdir(exist_ok=True)
        
        # Add get_attention_maps method to the model if it doesn't have one
        if not hasattr(model, 'get_attention_maps'):
            def get_attention_maps(pixel_values):
                batch_size = pixel_values.shape[0]
                return [torch.rand(batch_size, 8, 196, 196, device=device)]
            
            model.get_attention_maps = get_attention_maps
        
        # Multiple patches to avoid actual execution
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('torchvision.transforms.Compose.__call__', return_value=torch.randn(1, 3, 224, 224, device=device)):
                with patch('PIL.Image.open'):
                    with patch('utils.device.get_device', return_value=device):
                        # Run attention visualization
                        visualize_attention(model, str(image_path), "How many receipts are in this image?", output_dir)
                        
                        # Verify that savefig was called
                        mock_savefig.assert_called_once()
    
    def test_grid_search_config_generation(self):
        """Test generating configurations for grid search."""
        # Create a base config for testing
        config_path = self.test_dir / "test_config.yaml"
        config = {
            "model": {
                "name": "InternVL2"
            },
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            base_config_path=str(config_path),
            experiments_dir=str(self.experiments_dir)
        )
        
        # Define parameter grid
        param_grid = {
            "training.learning_rate": [1e-4, 5e-4, 1e-3],
            "training.batch_size": [16, 32]
        }
        
        # Generate grid configs
        configs = orchestrator._generate_grid_configs(param_grid)
        
        # Verify correct number of configurations
        self.assertEqual(len(configs), 6)  # 3 learning rates × 2 batch sizes
        
        # Verify configuration structure
        for cfg in configs:
            self.assertIn("training", cfg)
            self.assertIn("learning_rate", cfg["training"])
            self.assertIn("batch_size", cfg["training"])
    
    def test_hyperparameter_analysis(self):
        """Test analyzing hyperparameter optimization results."""
        # Create a base config for testing
        config_path = self.test_dir / "test_config.yaml"
        config = {
            "model": {
                "name": "InternVL2"
            },
            "training": {
                "epochs": 2
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            base_config_path=str(config_path),
            experiments_dir=str(self.experiments_dir)
        )
        
        # Create mock hyperparameter results
        results = [
            {
                "experiment": "hp_1",
                "config": {
                    "training": {
                        "learning_rate": 1e-4,
                        "batch_size": 16
                    }
                },
                "seed": 42,
                "results": {
                    "classification": {
                        "accuracy": 83.5,
                        "f1": 82.7
                    },
                    "generation": {
                        "bleu": 0.33,
                        "perplexity": 15.2
                    }
                }
            },
            {
                "experiment": "hp_2",
                "config": {
                    "training": {
                        "learning_rate": 5e-4,
                        "batch_size": 16
                    }
                },
                "seed": 42,
                "results": {
                    "classification": {
                        "accuracy": 85.1,
                        "f1": 84.2
                    },
                    "generation": {
                        "bleu": 0.38,
                        "perplexity": 12.9
                    }
                }
            },
            {
                "experiment": "hp_3",
                "config": {
                    "training": {
                        "learning_rate": 1e-3,
                        "batch_size": 32
                    }
                },
                "seed": 42,
                "results": {
                    "classification": {
                        "accuracy": 84.2,
                        "f1": 83.5
                    },
                    "generation": {
                        "bleu": 0.36,
                        "perplexity": 13.8
                    }
                }
            }
        ]
        
        # Analyze results
        analysis = orchestrator._analyze_hyperparameter_results(results)
        
        # Verify analysis structure
        self.assertIn("best_configurations", analysis)
        self.assertIn("parameter_correlations", analysis)
        
        # Verify best configurations
        best = analysis["best_configurations"]
        self.assertEqual(best["best_classification_accuracy"]["experiment"], "hp_2")
        self.assertEqual(best["best_generation_bleu"]["experiment"], "hp_2")
        self.assertEqual(best["best_generation_perplexity"]["experiment"], "hp_2")


if __name__ == "__main__":
    unittest.main()