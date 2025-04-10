#!/usr/bin/env python3
"""
Verification tests for Stage 2 (Multimodal Dataset Implementation).

This module contains tests to verify the correctness of the existing
implementation of Stage 2, which focuses on multimodal dataset creation
for vision-language integration.
"""

import os
import unittest
import tempfile
from pathlib import Path
import sys
import json
import random
import string
import pandas as pd
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components to test
from data.dataset import MultimodalReceiptDataset, ReceiptDataset, create_dataloaders, collate_fn_multimodal
from data.data_generators.create_multimodal_data import (
    generate_question_templates, 
    generate_answer_templates,
    generate_qa_pair,
    create_synthetic_multimodal_data
)


class TestMultimodalDatasetImplementation(unittest.TestCase):
    """Test the implementation of the multimodal dataset components."""
    
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
        
        # Create test images directory
        cls.img_dir = cls.test_dir / "images"
        cls.img_dir.mkdir(exist_ok=True)
        
        # Create a small test dataset for testing
        cls.create_test_dataset()
        
        # Create a mock tokenizer path (we won't actually load it in tests)
        cls.tokenizer_path = cls.test_dir / "mock_tokenizer"
        cls.tokenizer_path.mkdir(exist_ok=True)
        
        # Mock config for dataloaders
        cls.config = {
            "data": {
                "train_csv": str(cls.test_dir / "train_metadata.csv"),
                "val_csv": str(cls.test_dir / "val_metadata.csv"),
                "train_dir": str(cls.img_dir),
                "val_dir": str(cls.img_dir),
                "batch_size": 2,
                "num_workers": 0,
                "augmentation": True,
                "image_size": 224
            },
            "model": {
                "pretrained_path": str(cls.tokenizer_path),
                "multimodal": True,
                "num_classes": 3
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    @classmethod
    def create_test_dataset(cls):
        """Create a small test dataset for testing."""
        # Create train_metadata.csv
        train_data = []
        for i in range(5):
            # Create a small test image
            img = Image.new('RGB', (224, 224), color=random.randint(0, 255))
            img_path = cls.img_dir / f"test_image_{i}.png"
            img.save(img_path)
            
            receipt_count = random.randint(0, 2)
            
            # Add three QA pairs for each image
            for j in range(3):
                question = f"Test question {j} for image {i}?"
                answer = f"Test answer {j} for image {i} with {receipt_count} receipts."
                
                train_data.append({
                    "filename": f"test_image_{i}.png",
                    "receipt_count": receipt_count,
                    "question": question,
                    "answer": answer,
                    "qa_pair_idx": j
                })
        
        # Save as CSV
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(cls.test_dir / "train_metadata.csv", index=False)
        
        # Create validation set (copy of train for simplicity)
        train_df.to_csv(cls.test_dir / "val_metadata.csv", index=False)
    
    def test_question_template_generation(self):
        """Verify that question templates are generated correctly."""
        templates = generate_question_templates()
        
        # Check that all expected question types are present
        self.assertIn("counting", templates)
        self.assertIn("existence", templates)
        self.assertIn("value", templates)
        self.assertIn("detail", templates)
        
        # Check that each question type has multiple templates
        for question_type, questions in templates.items():
            self.assertGreaterEqual(len(questions), 5, f"{question_type} should have at least 5 templates")
            
            # Check that all templates are strings and end with a question mark or period
            for question in questions:
                self.assertIsInstance(question, str)
                self.assertTrue(question.endswith('?') or question.endswith('.'), 
                              f"Question should end with ? or .: {question}")
    
    def test_answer_template_generation(self):
        """Verify that answer templates are generated correctly."""
        templates = generate_answer_templates()
        
        # Check that all expected answer types are present
        self.assertIn("counting", templates)
        self.assertIn("existence_yes", templates)
        self.assertIn("existence_no", templates)
        self.assertIn("value", templates)
        
        # Check detailed answer types
        self.assertIn("detail_high_value", templates)
        self.assertIn("detail_stores", templates)
        self.assertIn("detail_date", templates)
        self.assertIn("detail_payment", templates)
        
        # Check that each answer type has multiple templates
        for answer_type, answers in templates.items():
            self.assertGreaterEqual(len(answers), 3, f"{answer_type} should have at least 3 templates")
            
            # Check that all templates are strings and end with a period
            for answer in answers:
                self.assertIsInstance(answer, str)
                # Check for placeholders in appropriate templates
                if answer_type == "counting":
                    self.assertIn("{count}", answer)
                    self.assertIn("{is_are}", answer)
                elif answer_type == "value":
                    self.assertIn("{total_value", answer)
                elif answer_type == "detail_high_value":
                    self.assertIn("{highest_value", answer)
                    self.assertIn("{store_name}", answer)
    
    def test_qa_pair_generation(self):
        """Verify that QA pairs are generated correctly."""
        # Test counting question generation
        receipt_count = 2
        qa_pair = generate_qa_pair(receipt_count)
        
        # Check structure
        self.assertIsInstance(qa_pair, dict)
        self.assertIn("question", qa_pair)
        self.assertIn("answer", qa_pair)
        
        # Test generation with additional metadata
        receipt_values = [10.50, 25.75]
        store_names = ["GROCERY WORLD", "SUPERMARKET PLUS"]
        dates = ["January 1, 2023", "February 15, 2023"]
        payment_methods = ["VISA", "CASH"]
        
        qa_pair = generate_qa_pair(
            receipt_count=receipt_count,
            receipt_values=receipt_values,
            store_names=store_names,
            dates=dates,
            payment_methods=payment_methods
        )
        
        # Check again with metadata
        self.assertIsInstance(qa_pair, dict)
        self.assertIn("question", qa_pair)
        self.assertIn("answer", qa_pair)
        
        # Test zero receipt case
        zero_qa = generate_qa_pair(0)
        self.assertIsInstance(zero_qa, dict)
        self.assertIn("question", zero_qa)
        self.assertIn("answer", zero_qa)
        
        # The answer should mention that there are no receipts
        self.assertTrue(
            "no receipts" in zero_qa["answer"].lower() or 
            "0 receipt" in zero_qa["answer"].lower(),
            f"Zero receipt answer should mention no receipts: {zero_qa['answer']}"
        )
    
    def test_multimodal_dataset(self):
        """Verify that the MultimodalReceiptDataset is implemented correctly."""
        # Mock AutoTokenizer
        class MockTokenizer:
            def __init__(self, *args, **kwargs):
                self.vocab_size = 30000
            
            def __call__(self, text, **kwargs):
                # Return a mock encoding object
                class MockEncoding:
                    def __init__(self, text_len):
                        # Create random input IDs and attention mask
                        seq_len = kwargs.get('max_length', 128)
                        self.input_ids = torch.randint(0, 30000, (1, seq_len))
                        self.attention_mask = torch.ones((1, seq_len))
                
                return MockEncoding(len(text))
            
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(*args, **kwargs)
        
        # Monkey patch AutoTokenizer
        import transformers
        original_tokenizer = transformers.AutoTokenizer
        transformers.AutoTokenizer = MockTokenizer
        
        try:
            # Create dataset
            dataset = MultimodalReceiptDataset(
                csv_file=self.test_dir / "train_metadata.csv",
                img_dir=self.img_dir,
                tokenizer_path=self.tokenizer_path,
                image_size=224
            )
            
            # Check length
            self.assertEqual(len(dataset), 15)  # 5 images × 3 QA pairs
            
            # Check getitem
            item = dataset[0]
            
            # Verify structure
            self.assertIn("pixel_values", item)
            self.assertIn("text_input_ids", item)
            self.assertIn("text_attention_mask", item)
            self.assertIn("labels", item)
            self.assertIn("labels_attention_mask", item)
            self.assertIn("classification_labels", item)
            self.assertIn("receipt_count", item)
            
            # Check types and shapes
            self.assertIsInstance(item["pixel_values"], torch.Tensor)
            self.assertIsInstance(item["text_input_ids"], torch.Tensor)
            self.assertIsInstance(item["classification_labels"], torch.Tensor)
            
            # Pixel values should be [3, 224, 224] (C, H, W)
            self.assertEqual(item["pixel_values"].shape, (3, 224, 224))
            
            # Text should be [max_length]
            self.assertEqual(item["text_input_ids"].dim(), 1)
            self.assertEqual(item["text_attention_mask"].dim(), 1)
            
        finally:
            # Restore original tokenizer
            transformers.AutoTokenizer = original_tokenizer
    
    def test_collate_fn_multimodal(self):
        """Verify that the collate function for multimodal batches works correctly."""
        # Create mock batch items
        batch = []
        for i in range(3):
            batch.append({
                "pixel_values": torch.randn(3, 224, 224),
                "text_input_ids": torch.randint(0, 30000, (128,)),
                "text_attention_mask": torch.ones(128),
                "labels": torch.randint(0, 30000, (128,)),
                "labels_attention_mask": torch.ones(128),
                "classification_labels": torch.tensor(i % 3),
                "receipt_count": torch.tensor(i % 3)
            })
        
        # Call collate function
        collated = collate_fn_multimodal(batch)
        
        # Check structure
        self.assertIn("pixel_values", collated)
        self.assertIn("text_input_ids", collated)
        self.assertIn("text_attention_mask", collated)
        self.assertIn("labels", collated)
        self.assertIn("labels_attention_mask", collated)
        self.assertIn("classification_labels", collated)
        self.assertIn("receipt_count", collated)
        
        # Check batch dimension
        self.assertEqual(collated["pixel_values"].shape[0], 3)
        self.assertEqual(collated["text_input_ids"].shape[0], 3)
        self.assertEqual(collated["classification_labels"].shape[0], 3)
        
        # Check proper stacking
        self.assertEqual(collated["pixel_values"].shape, (3, 3, 224, 224))
        self.assertEqual(collated["text_input_ids"].shape, (3, 128))
    
    def test_dataloaders_creation(self):
        """Verify that data loaders are created correctly."""
        # Mock the transformers.AutoTokenizer again
        class MockTokenizer:
            def __init__(self, *args, **kwargs):
                self.vocab_size = 30000
            
            def __call__(self, text, **kwargs):
                # Return a mock encoding object
                class MockEncoding:
                    def __init__(self, text_len):
                        # Create random input IDs and attention mask
                        seq_len = kwargs.get('max_length', 128)
                        self.input_ids = torch.randint(0, 30000, (1, seq_len))
                        self.attention_mask = torch.ones((1, seq_len))
                
                return MockEncoding(len(text))
            
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(*args, **kwargs)
        
        # Monkey patch AutoTokenizer
        import transformers
        original_tokenizer = transformers.AutoTokenizer
        transformers.AutoTokenizer = MockTokenizer
        
        try:
            # Create data loaders
            loaders = create_dataloaders(self.config)
            
            # Check structure
            self.assertIn("train", loaders)
            self.assertIn("val", loaders)
            
            # Check that loaders are properly initialized
            self.assertEqual(loaders["train"].batch_size, 2)
            self.assertEqual(loaders["val"].batch_size, 2)
            
            # Check multimodal dataset properties
            self.assertIsInstance(loaders["train"].dataset, MultimodalReceiptDataset)
            self.assertEqual(len(loaders["train"].dataset), 15)  # 5 images × 3 QA pairs
            
            # Try a forward batch
            batch = next(iter(loaders["train"]))
            self.assertIn("pixel_values", batch)
            self.assertIn("text_input_ids", batch)
            self.assertIn("classification_labels", batch)
            
            # Check batch shapes
            self.assertEqual(batch["pixel_values"].shape[0], 2)  # Batch size
            self.assertEqual(batch["pixel_values"].shape[1], 3)  # Channels
            
        finally:
            # Restore original tokenizer
            transformers.AutoTokenizer = original_tokenizer
    
    def test_dataset_distribution(self):
        """Verify that the dataset distribution matches requirements."""
        # Create a synthetic dataset in memory for testing distributions
        # Mock create_receipt_image and create_blank_image so we don't actually generate images
        from data.data_generators.create_synthetic_receipts import create_receipt_image
        from data.data_generators.receipt_processor import create_blank_image
        
        # Save original functions
        original_create_receipt = create_receipt_image
        original_create_blank = create_blank_image
        
        # Mock Image.open
        original_image_open = Image.open
        
        try:
            # Mock create_receipt_image
            def mock_create_receipt(*args, **kwargs):
                return Image.new('RGB', (300, 800), color=(255, 255, 255))
            
            # Mock create_blank_image
            def mock_create_blank(*args, **kwargs):
                return Image.new('RGB', (448, 448), color=(255, 255, 255))
            
            # Mock Image.open
            def mock_image_open(*args, **kwargs):
                return Image.new('RGB', (448, 448), color=(255, 255, 255))
            
            # Apply mocks
            import data.data_generators.create_synthetic_receipts
            import data.data_generators.receipt_processor
            data.data_generators.create_synthetic_receipts.create_receipt_image = mock_create_receipt
            data.data_generators.receipt_processor.create_blank_image = mock_create_blank
            Image.open = mock_image_open
            
            # Create output dir for this test
            output_dir = self.test_dir / "distribution_test"
            output_dir.mkdir(exist_ok=True)
            
            # Generate a small synthetic dataset
            df = create_synthetic_multimodal_data(
                num_samples=20,
                output_dir=output_dir,
                image_size=224,
                seed=42
            )
            
            # Check dataframe structure
            self.assertIn("filename", df.columns)
            self.assertIn("receipt_count", df.columns)
            self.assertIn("question", df.columns)
            self.assertIn("answer", df.columns)
            
            # Check distribution
            # Count frequencies of each receipt count
            counts = df['receipt_count'].value_counts()
            
            # Verify that we have all receipt counts
            for count in range(6):  # 0-5 receipts
                self.assertIn(count, counts.index,
                             f"Receipt count {count} should be in the dataset")
            
            # Check question diversity
            unique_questions = df['question'].nunique()
            self.assertGreaterEqual(unique_questions, 10, 
                                 "There should be at least 10 unique questions")
            
            # Check that each image has multiple QA pairs
            qa_counts = df.groupby('filename').size()
            for count in qa_counts:
                self.assertEqual(count, 3, "Each image should have 3 QA pairs")
            
            # Check that the QA pairs file was created
            qa_path = output_dir / "qa_pairs.json"
            self.assertTrue(qa_path.exists())
            
            # Verify JSON structure
            with open(qa_path, 'r') as f:
                qa_data = json.load(f)
                self.assertIsInstance(qa_data, list)
                self.assertGreaterEqual(len(qa_data), 20*3)  # At least 20 images × 3 QA pairs
            
        finally:
            # Restore original functions
            data.data_generators.create_synthetic_receipts.create_receipt_image = original_create_receipt
            data.data_generators.receipt_processor.create_blank_image = original_create_blank
            Image.open = original_image_open


if __name__ == "__main__":
    unittest.main()