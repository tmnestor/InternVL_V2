"""
Simple test script for evaluating the trained classifier.

This script loads the trained classifier and tests it on a few examples.
"""
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.components.question_classifier import QuestionClassifier
from transformers import AutoTokenizer
from utils.device import get_device


def main():
    """Main function to test the classifier."""
    print("Simple Question Classifier Test")
    print("==============================")
    
    # Define question types mapping
    question_types = {
        0: "DOCUMENT_TYPE",
        1: "COUNTING",
        2: "DETAIL_EXTRACTION",
        3: "PAYMENT_INFO",
        4: "TAX_INFO"
    }
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Initialize model
    model = QuestionClassifier(
        model_name="distilbert-base-uncased",
        hidden_size=768,
        num_classes=5,
        device=device,
        use_custom_path=False
    )
    
    # Try to load the best model from our enhanced training
    try:
        model_path = "models/enhanced_classifier/best_model.pt"
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_loaded = True
    except:
        # Fall back to the original model
        try:
            model_path = "models/question_classifier/best_model.pt"
            print(f"Enhanced model not found, trying original model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_loaded = True
        except:
            print("Could not load any model. Using untrained model.")
            model_loaded = False
    
    # Define test questions
    test_questions = [
        "Is this a receipt?",  # DOCUMENT_TYPE
        "What kind of document is this?",  # DOCUMENT_TYPE
        "How many receipts are in this image?",  # COUNTING
        "Count the number of receipts.",  # COUNTING
        "What store is this receipt from?",  # DETAIL_EXTRACTION
        "What is the date on this receipt?",  # DETAIL_EXTRACTION
        "How was this purchase paid for?",  # PAYMENT_INFO
        "What payment method was used?",  # PAYMENT_INFO
        "What tax form is this?",  # TAX_INFO
        "What is the ABN on this document?",  # TAX_INFO
    ]
    
    # Get predictions
    model.eval()
    
    # Print header
    print("\nTest Results:")
    print(f"{'Question':<50} {'Predicted Type':<20}")
    print("-" * 70)
    
    with torch.no_grad():
        for question in test_questions:
            # Tokenize
            inputs = tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Get prediction
            logits = model(inputs.input_ids, inputs.attention_mask)
            pred_id = torch.argmax(logits, dim=1).item()
            pred_type = question_types[pred_id]
            
            # Print result
            print(f"{question:<50} {pred_type:<20}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()