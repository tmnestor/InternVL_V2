"""
Test script for evaluating the trained question classifier.

This script loads a trained question classifier model and evaluates
it on custom test questions to check its performance.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.components.question_classifier import QuestionClassifier
from utils.device import get_device


def test_question_classifier(
    model_path: str,
    test_questions: list,
    device=None
):
    """
    Test a trained question classifier on custom examples.
    
    Args:
        model_path: Path to the trained model
        test_questions: List of test questions to evaluate
        device: Device to run inference on
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # Get device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = QuestionClassifier(
        model_name="distilbert-base-uncased", 
        hidden_size=768,
        num_classes=5,
        device=device,
        use_custom_path=False
    )
    
    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        if "model_state_dict" in state_dict:
            # If saved with optimizer state dict
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            # If saved as just the model state dict
            model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Define question type mapping
    question_types = {
        0: "DOCUMENT_TYPE",
        1: "COUNTING",
        2: "DETAIL_EXTRACTION",
        3: "PAYMENT_INFO",
        4: "TAX_INFO"
    }
    
    # Test each question
    logger.info(f"\nTesting {len(test_questions)} questions:")
    print("\n" + "="*60)
    print(f"{'QUESTION':<40}{'PREDICTED TYPE':<20}")
    print("="*60)
    
    with torch.no_grad():
        for question in test_questions:
            # Predict question type
            predicted_type = model.predict_question_type(question)
            
            # Print results
            print(f"{question[:37] + '...' if len(question) > 40 else question:<40}{predicted_type:<20}")
    
    print("="*60)


def main():
    """Main function for testing the question classifier."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test question classifier")
    parser.add_argument("--model-path", type=str, default="models/question_classifier/best_model.pt", 
                      help="Path to trained model")
    args = parser.parse_args()
    
    # Define test questions - cover all categories with multiple examples
    test_questions = [
        # Document type questions
        "Is this a receipt?",
        "What kind of document is this?",
        "Is this a tax document?",
        "Can you identify this document?",
        
        # Counting questions
        "How many receipts are in this image?",
        "Count the number of receipts.",
        "Are there multiple receipts here?",
        "What's the count of receipts in this picture?",
        
        # Detail extraction questions
        "What store is this receipt from?",
        "What is the date on this receipt?",
        "What items were purchased?",
        "When was this purchase made?",
        "What's the GST amount on this receipt?",
        "What's the ABN listed on the receipt?",
        "What's the cashier's name on the receipt?",
        "What was the most expensive item purchased?",
        
        # Payment information questions
        "How was this purchase paid for?",
        "What payment method was used?",
        "Was this paid by credit card?",
        "Did they pay with cash or card?",
        
        # Tax information questions
        "What tax form is this?",
        "What is the ABN on this document?",
        "What tax year does this document cover?",
        "Is this an official ATO document?",
        
        # Mixed/complex questions
        "Can you tell me all the details of this receipt?",
        "When was this receipt issued and what was purchased?",
        "I need information about this document's type and date.",
        "Please extract all important information from this receipt.",
        "What's the total amount and payment method on this receipt?",
        "The address of the store on the receipt and the items bought please.",
        "Tell me when this was purchased and how many items were bought.",
        "Who issued this document and when?"
    ]
    
    # Run test
    test_question_classifier(args.model_path, test_questions)


if __name__ == "__main__":
    main()