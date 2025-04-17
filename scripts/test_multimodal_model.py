#!/usr/bin/env python3
"""
Test script for multimodal vision-language receipt counter model.

This script loads a trained multimodal model and tests it with example inputs.
"""
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer

from models.vision_language.internvl2 import InternVL2MultimodalModel


def test_multimodal_model(model_path, image_path, questions):
    """
    Test a multimodal model with an image and questions.
    
    Args:
        model_path: Path to model checkpoint
        image_path: Path to test image
        questions: List of questions to ask about the image
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model configuration
    config = {
        "model": {
            "pretrained_path": model_path,
            "multimodal": True,
            "classifier": {
                "hidden_dims": [256, 128],
                "dropout_rates": [0.2, 0.1],
                "batch_norm": True,
                "activation": "gelu",
            },
            "num_classes": 3,
        }
    }
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    model = InternVL2MultimodalModel(
        config=config,
        pretrained=True,
        freeze_vision_encoder=False,
        freeze_language_model=False,
    )
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Tokenize question
        text_encoding = tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        
        # Run model
        with torch.no_grad():
            # Get responses
            tokens, responses = model.generate_response(
                pixel_values=image_tensor,
                text_input_ids=text_encoding.input_ids,
                attention_mask=text_encoding.attention_mask,
                max_length=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )
            
            # Get classification result
            outputs = model(
                pixel_values=image_tensor,
                text_input_ids=text_encoding.input_ids,
                attention_mask=text_encoding.attention_mask,
            )
            logits = outputs["logits"]
            predicted_class = torch.argmax(logits, dim=1).item()
            
        # Print results
        print(f"Classification: {predicted_class} receipts")
        print(f"Response: {responses[0]}")
    
    print("\nMultimodal model test completed!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test multimodal receipt counter model")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", required=True, help="Path to test image")
    parser.add_argument("--questions", nargs="+", default=["How many receipts are in this image?"], 
                         help="Questions to ask about the image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    test_multimodal_model(
        model_path=args.model_path,
        image_path=args.image_path,
        questions=args.questions,
    )